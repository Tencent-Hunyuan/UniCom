import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder.utils.helpers import to_2tuple
from decoder.runtime_context import get_runtime_args


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding

    Image to Patch Embedding using Conv2d

    A convolution based approach to patchifying a 2D image w/ embedding projection.

    Based on the impl in https://github.com/google-research/vision_transformer

    Hacked together by / Copyright 2020 Ross Wightman

    Remove the _assert function in forward function to be compatible with multi-resolution images.
    """
    def __init__(
            self,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            multitask_mask_training_type=None,
            norm_layer=None,
            flatten=True,
            bias=True,
            dtype=None,
            device=None
    ):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        if multitask_mask_training_type == "concat":
            orig_in_chans = in_chans
            in_chans = in_chans * 2 + 1

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias,
                              **factory_kwargs)

        if multitask_mask_training_type == "concat":
            nn.init.xavier_uniform_(self.proj.weight[:, :orig_in_chans].view(self.proj.weight[:, :orig_in_chans].size(0), -1))
            nn.init.zeros_(self.proj.weight[:, orig_in_chans:].view(self.proj.weight[:, orig_in_chans:].size(0), -1))
        else:
            nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))

        if bias:
            nn.init.zeros_(self.proj.bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class TextProjection(nn.Module):
    """Minimal linear projection kept for compatibility."""

    def __init__(self, in_channels, hidden_size, act_layer, dtype=None, device=None):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        args = get_runtime_args()
        self.aba_mode = args.aba_mode
        input_dim = 3584 if "qwen" in args.vision_encoder_name.lower() else 1152
        self.linear_0 = nn.Linear(in_features=input_dim, out_features=in_channels, bias=True, **factory_kwargs)
        self.act_0 = act_layer()
        self.linear_1 = nn.Linear(in_features=in_channels, out_features=hidden_size, bias=True, **factory_kwargs)
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True, **factory_kwargs)

    def forward(self, caption, generator=None):
        del generator
        hidden_states = self.linear_0(caption)
        hidden_states = self.act_0(hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.

    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self,
                 hidden_size,
                 act_layer,
                 frequency_embedding_size=256,
                 max_period=10000,
                 out_size=None,
                 dtype=None,
                 device=None
                 ):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class PlainAttention(nn.Module):
    """Plain attention module for channel compression/expansion."""
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if in_dim > out_dim:
            # Channel compression case
            # Make sure head_dim equals out_dim for exact division
            self.head_dim = out_dim
            assert in_dim % self.head_dim == 0, f"in_dim ({in_dim}) must be divisible by head_dim ({self.head_dim})"
            # Recompute num_heads based on head_dim
            self.num_heads = in_dim // self.head_dim
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(in_dim))
            self.v_bias = nn.Parameter(torch.zeros(in_dim))
            self.register_buffer('zero_k_bias', torch.zeros(in_dim))
        else:
            # Channel expansion case
            self.num_heads = num_heads
            self.head_dim = out_dim // num_heads
            assert out_dim % num_heads == 0, f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
            self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False) # 1152，3584*3
            self.q_bias = nn.Parameter(torch.zeros(out_dim))
            self.v_bias = nn.Parameter(torch.zeros(out_dim))
            self.register_buffer('zero_k_bias', torch.zeros(out_dim))

        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))) # [B, N, 3*C]
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0) #  [B, head, N, C/head]

        x = F.scaled_dot_product_attention(q, k, v)

        if self.in_dim > self.out_dim:
            # Channel compression: average over heads
            # Now head_dim == out_dim, so after mean we get exactly out_dim
            x = torch.mean(x, dim=1)  # [B, N, head_dim] where head_dim == out_dim
            if self.in_dim // self.num_heads != self.out_dim:
                x = nn.functional.adaptive_avg_pool1d(x, self.out_dim)
        else:
            # Channel expansion: reshape
            x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        return x


class GeGluMLP(nn.Module):
    """Gated MLP implementation for attention projection with configurable activation."""
    def __init__(self, in_features, hidden_features, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttnProjection(nn.Module):
    """Attention projection module for channel compression/expansion."""
    def __init__(self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2, act_layer=nn.SiLU):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = norm_layer(in_dim)
        self.attn = PlainAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = norm_layer(in_dim)

        self.norm2 = norm_layer(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMLP(
            in_features=out_dim,
            hidden_features=hidden_dim,
            act_layer=act_layer
        )

    def forward(self, x):
        # First residual connection: attention + linear projection
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        # Second residual connection: + MLP
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionTextProjection(nn.Module):
    """Attention-based projection for the exported SigLIP compression path."""

    def __init__(self, in_channels, hidden_size, act_layer, compress_dim=64, freeze_compress=False, dtype=None, device=None):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()

        args = get_runtime_args()
        self.aba_mode = args.aba_mode
        supported_aba_modes = {"eval_compression_64_siglip", "compression_64_siglip"}
        if self.aba_mode not in supported_aba_modes:
            raise ValueError(
                f"Only aba_mode in {sorted(supported_aba_modes)} is supported, got {self.aba_mode}"
            )

        if "qwen" in args.vision_encoder_name.lower():
            self.original_dim = 3584
        else:
            self.original_dim = 1152

        self.compress_dim = compress_dim
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.freeze_compress = freeze_compress


        assert self.original_dim % self.compress_dim == 0, \
            f"original_dim ({self.original_dim}) must be divisible by compress_dim ({self.compress_dim})"

        self.compress_heads = self.original_dim // self.compress_dim
        self.expand_heads = self.original_dim // self.compress_dim

        # Compression weights are kept for checkpoint compatibility even though
        # inference only uses the exported compressed features.
        self.compress_attn = AttnProjection(
            in_dim=self.original_dim,
            out_dim=self.compress_dim,
            num_heads=self.compress_heads,
            norm_layer=nn.LayerNorm,
            mlp_ratio=2,
            act_layer=act_layer
        )

        if self.freeze_compress:
            for param in self.compress_attn.parameters():
                param.requires_grad = False

        self.expand_attn = AttnProjection(
            in_dim=self.compress_dim,
            out_dim=self.original_dim,
            num_heads=self.expand_heads,
            norm_layer=nn.LayerNorm,
            mlp_ratio=2,
            act_layer=act_layer
        )

        # Final projection to target dimension
        self.final_proj = nn.Linear(self.original_dim, in_channels, **factory_kwargs)
        self.act_final = act_layer()

        # Output projection
        self.linear_1 = nn.Linear(in_features=in_channels, out_features=hidden_size, bias=True, **factory_kwargs)
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True, **factory_kwargs)

    def forward(self, caption, generator=None):
        del generator
        if self.aba_mode == "compression_64_siglip":
            if caption.shape[-1] == self.original_dim:
                caption = self.compress_attn(caption)
            elif caption.shape[-1] != self.compress_dim:
                raise ValueError(
                    f"compression_64_siglip expects caption dim {self.original_dim} or {self.compress_dim}, "
                    f"got {caption.shape[-1]}"
                )
        expanded = self.expand_attn(caption)
        hidden_states = self.final_proj(expanded)
        hidden_states = self.act_final(hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
