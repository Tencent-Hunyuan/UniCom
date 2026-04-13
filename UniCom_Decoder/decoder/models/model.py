from typing import List, Tuple, Optional, Union, Dict
from einops import rearrange
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from decoder.runtime_context import get_runtime_args, print_rank_0
from decoder.models.attention_adapter import (
    DotProductAttention,
)

from decoder.models.modules import (
    get_activation_layer,
    get_norm_layer,
    timestep_embedding,
    modulate,
    apply_gate,
    FP32_Layernorm,
    ModulateDiT,
    TimestepEmbedder,
    PatchEmbed,
    MLP,
    MLPEmbedder,
    FinalLayer,
)

from decoder.models.modules.embed_layers import AttentionTextProjection, TextProjection


def reshape_for_broadcast(freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]], x: torch.Tensor, head_first=False):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Notes:
        When using FlashMHAModified, head_first should be False.
        When using Attention, head_first should be True.

    Args:
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (x.shape[-2], x.shape[-1]), f'freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}'
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis[0].shape == (x.shape[1], x.shape[-1]), f'freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}'
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (x.shape[-2], x.shape[-1]), f'freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}'
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f'freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}'
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (torch.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (torch.Tensor or tuple): Precomputed frequency tensor for complex exponential.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)    # [S, D]
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        # real * cos - imag * sin
        # imag * cos + real * sin
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(xq.device)   # [S, D//2] --> [1, S, 1, D//2]
        # (real, imag) * (cos, sin) = (real * cos - imag * sin, imag * cos + real * sin)
        # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # [B, S, H, D//2]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


def attention(q, k, v, drop_rate=0, attn_mask=None, causal=False):
    # [b, s, h, d] -> [b, h, s, d]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.to(q.dtype)

    # [b, h, s, d]
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)

    # [b, h, s, d] -> [b, s, h, d]
    x = x.transpose(1, 2)
    b, s, a, d = x.shape
    # [b, s, h, d] -> [b, s, hd]
    out = x.reshape(b, s, -1)
    return out


def get_cu_seqlens(text_mask, img_len):
    """ 根据 text_mask 计算 cu_seqlens_q, cu_seqlens_kv
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


def get_cu_seqlens_fused(mask: torch.Tensor) -> torch.Tensor:
    """
    Given a padding mask of shape [batch_size, max_seqlen], returns an int32
    tensor of shape [batch_size + 1] containing the cumulative sequence lengths of
    the samples in a batch.
    """
    reduced_mask = mask.logical_not().sum(dim=1)
    cu_seqlens = reduced_mask.cumsum(dim=0).to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.cat((zero, cu_seqlens))

    return cu_seqlens


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_act_type: str = 'gelu_tanh',
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        attn_mode: str = "flash",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.attn_mode = attn_mode
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        ## ------------------ img -------------------
        self.img_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        args = get_runtime_args()
        if args.window_size == -1:
            self.window_size = None
        else:
            self.window_size = (args.window_size, args.window_size)
        if args.force_fp32_ops:
            self.img_norm1 = FP32_Layernorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        else:
            self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        if args.force_fp32_ops:
            self.img_norm2 = FP32_Layernorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        else:
            self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs
        )

        ## ------------------ txt -------------------
        self.txt_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        if args.force_fp32_ops:
            self.txt_norm1 = FP32_Layernorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        else:
            self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        if args.force_fp32_ops:
            self.txt_norm2 = FP32_Layernorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        else:
            self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs
        )

        ## ------------------ attn -------------------
        self.core_attention = DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=head_dim,
            num_gqa_groups=num_heads,
            attention_dropout=0.0,
            qkv_format="sbhd",
            tp_size=1,
            get_rng_state_tracker=None,
            sequence_parallel=False,
            tp_group=None,
            layer_number=None,
            attention_type="self",
        )

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        attn_mask: torch.Tensor,
        self_attn_mask_type: str = "no_mask",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        cu_seqlens_q_padded: Optional[torch.Tensor] = None,
        cu_seqlens_kv_padded: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        ## ------
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        img: [b, s, h]
        txt: [b, l, h]
        vec: [b, h]
        attn_mask: [b, 1, s+l, s+l]
        self_attn_mask_type: "no_mask"
        cu_seqlens_q: [2b+1]
        cu_seqlens_kv: [2b+1]
        freqs_cis: [s, d]

        NOTE:
            b - batch_size
            s - img_len
            h - hidden_size
            l - txt_len
            a - num_heads
            d - head_dim
        """

        # [b, h]
        img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = (
            self.img_mod(vec).chunk(6, dim=-1)
        )
        # [b, h]
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = (
            self.txt_mod(vec).chunk(6, dim=-1)
        )

        # ---------------------------- Prepare image for attention. ----------------------------
        # [b, s, h]
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)
        # [b, s, 3h]
        img_qkv = self.img_attn_qkv(img_modulated)
        # [b, s, (3 a d)] -> [3, b, s, a, d] -> 3 * [b, s, a, d]
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)

        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, \
                f'img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}'
            img_q, img_k = img_qq, img_kk

        # ---------------------------- Prepare txt for attention. ----------------------------
        # [b, l, h]
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        # [b, l, 3h]
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        # [b, l, (3 a d)] -> [3, b, l, a, d] -> 3 * [b, l, a, d]
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)

        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # ---------------------------- Run actual attention. ----------------------------
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)

        if self.attn_mode == "torch":

            # qkv: [b, s+l, a, d]
            # attn: [b, s+l, h]
            attn = attention(q, k, v, attn_mask=attn_mask)

        elif self.attn_mode == "flash":
            assert cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1

            # [b, s+l, a, d] -> [s+l, b, a, d], DotProductAttention need input tensor `q`, `k`, `v` must be of shape 'sbhd'.
            q = q.transpose(0, 1).contiguous()
            k = k.transpose(0, 1).contiguous()
            v = v.transpose(0, 1).contiguous()

            # qkv: [s+l, b, a, d]
            # attn: [s+l, b, h]
            attn = self.core_attention(
                q,
                k,
                v,
                qkv_format="sbhd",
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                attention_mask=attn_mask,
                attn_mask_type=self_attn_mask_type,
                window_size=self.window_size,
            )
            # [s, b, h] -> [b, s, h]
            attn = attn.transpose(0, 1).contiguous()

        # img_attn: [b, s, h]
        # txt_attn: [b, l, h]
        img_attn, txt_attn = attn[:, :img.shape[1], :], attn[:, img.shape[1]:, :]

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)), gate=img_mod2_gate)

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)), gate=txt_mod2_gate)

        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        mlp_act_type: str = 'gelu_tanh',
        qk_norm: bool = True,
        qk_norm_type: str = 'rms',
        qk_scale: float = None,
        attn_mode: str = "flash",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.deterministic = False
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim**-0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        args = get_runtime_args()
        if args.window_size == -1:
            window_size = None
        else:
            window_size = (args.window_size, args.window_size)
        if args.force_fp32_ops:
            self.pre_norm = FP32_Layernorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        else:
            self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=get_activation_layer("silu"), **factory_kwargs)

        ## ------------------ attn -------------------
        self.core_attention = DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=head_dim,
            num_gqa_groups=num_heads,
            attention_dropout=0.0,
            qkv_format="sbhd",
            tp_size=1,
            get_rng_state_tracker=None,
            sequence_parallel=False,
            tp_group=None,
            layer_number=None,
            attention_type="self",
        )

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        img_len: int,
        attn_mask: Optional[torch.Tensor],
        self_attn_mask_type: Optional[str] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        cu_seqlens_q_padded: Optional[torch.Tensor] = None,
        cu_seqlens_kv_padded: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        ## ------
    ) -> torch.Tensor:
        """
        x: [b, s+l, h]
        vec: [b, h]
        img_len: s
        attn_mask: [b, 1, s+l, s+l]
        freqs_cis: [s, d]

        NOTE:
            b - batch_size
            s - img_len
            h - hidden_size
            l - txt_len
            a - num_heads
            d - head_dim
        """

        # [b, h]
        mod_shift, mod_scale, mod_gate = (
            self.modulation(vec).chunk(3, dim=-1)
        )
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)

        # qkv: [b, s+l, 3*h]
        # mlp: [b, s+l, 4h]
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        # [b, s+l, a, d]
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)
        v = v.contiguous()

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :img_len, :, :], q[:, img_len:, :, :]
            img_k, txt_k = k[:, :img_len, :, :], k[:, img_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, \
                f'img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}'
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        # Compute attention.
        if self.attn_mode == "torch":
            attn = attention(q, k, v, attn_mask=attn_mask)

        elif self.attn_mode == "flash":
            assert cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1, f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"

            # [b, s+l, a, d] -> [s+l, b, a, d]
            q = q.transpose(0, 1).contiguous()
            k = k.transpose(0, 1).contiguous()
            v = v.transpose(0, 1).contiguous()

            # qkv: [s+l, b, a, d]
            # attn: [b, s+l, h]
            attn = self.core_attention(
                q,
                k,
                v,
                qkv_format="sbhd",
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                attention_mask=attn_mask,
                attn_mask_type=self_attn_mask_type,
            )
            attn = attn.transpose(0, 1).contiguous()

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate)


class UniComDiffusionTransformer(ModelMixin, ConfigMixin):
    """
    Transformer model for flow matching on sequences.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206,
               https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.
    """
    @register_to_config
    def __init__(
        self,
        patch_size: list = [1,2,2],
        in_channels: int = 4, # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        mlp_ratio: float = 4.0,
        mlp_act_type: str = 'gelu_tanh',
        num_heads: int = 24,
        depth_double_blocks: int = 19,
        depth_single_blocks: int = 38,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False, # For modulation.
        use_vanilla: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        multitask_mask_training_type: Optional[str] = None,
    ):
        args = get_runtime_args()
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.tp_size = 1

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.text_projection = args.text_projection
        self.text_states_dim = args.text_states_dim
        self.use_attention_mask = args.use_attention_mask

        # Text pooling. The pooled vector will be added to the timestep as a global context.
        # See more details (Lumina-Next): http://arxiv.org/abs/2406.18583
        #                  (Flux.1): https://github.com/black-forest-labs/flux
        self.text_pool_type = args.text_pool_type
        self.text_states_dim_2 = args.text_states_dim_2

        self.attn_mode = args.attn_mode
        self.aba_mode = args.aba_mode
        supported_aba_modes = {"eval_compression_64_siglip", "compression_64_siglip", None}
        if self.aba_mode not in supported_aba_modes:
            raise ValueError(
                f"Only aba_mode in {sorted(str(m) for m in supported_aba_modes)} is supported, got {self.aba_mode}"
            )

        self.patch_size = patch_size # [1, 2, 2]
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels #
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed # False
        self.rope_dim_list = rope_dim_list
        self.use_vanilla = use_vanilla # False
        self.multitask_mask_training_type = multitask_mask_training_type

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads # head_dim
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        if self.use_vanilla:
            self.img_in = nn.Linear(
                self.in_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
                self.hidden_size,
                bias=True,
                **factory_kwargs
            )
            if self.aba_mode is None:
                self.txt_in = TextProjection(
                    self.text_states_dim,
                    self.hidden_size,
                    get_activation_layer("silu"),
                    **factory_kwargs,
                )
            else:
                if "attn" not in self.text_projection:
                    raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")
                self.txt_in = AttentionTextProjection(
                    self.text_states_dim,
                    self.hidden_size,
                    get_activation_layer("silu"),
                    compress_dim=64,
                    freeze_compress=(self.text_projection == "attn_freeze"),
                    **factory_kwargs,
                )
            self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, **factory_kwargs)
            # no text_encoder under siglip
            self.vector_in = MLPEmbedder(
                self.text_states_dim_2,
                self.hidden_size,
                **factory_kwargs
            ) if self.text_pool_type is not None else None
            self.guidance_in = MLPEmbedder(
                in_dim=256,
                hidden_dim=self.hidden_size,
                **factory_kwargs
            ) if guidance_embed else nn.Identity()
        else:
            # image
            self.img_in = PatchEmbed(
                self.patch_size,
                self.in_channels,
                self.hidden_size,
                self.multitask_mask_training_type,
                **factory_kwargs
            )

            if self.aba_mode is None:
                self.txt_in = TextProjection(
                    self.text_states_dim,
                    self.hidden_size,
                    get_activation_layer("silu"),
                    **factory_kwargs,
                )
            else:
                if "attn" not in self.text_projection:
                    raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")
                self.txt_in = AttentionTextProjection(
                    self.text_states_dim,
                    self.hidden_size,
                    get_activation_layer("silu"),
                    compress_dim=64,
                    freeze_compress=(self.text_projection == "attn_freeze"),
                    **factory_kwargs,
                )

            # time modulation
            self.time_in = TimestepEmbedder(
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs
            )

            # text modulation
            self.vector_in = MLPEmbedder(
                self.text_states_dim_2,
                self.hidden_size,
                **factory_kwargs
            ) if self.text_pool_type is not None else None

            # guidance modulation
            self.guidance_in = TimestepEmbedder(
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs
            ) if guidance_embed else None

        self.self_attn_mask_type = "no_mask"
        self.use_fused_attn = False
        os.environ["NVTE_FUSED_ATTN"] = "0"
        self.use_flash_attn = True if self.attn_mode == "flash" else False
        print_rank_0(f"Standalone decoder runtime, use_flash_attn: {self.use_flash_attn}")

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    attn_mode=self.attn_mode,
                    **factory_kwargs
                )
                for _ in range(depth_double_blocks)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    attn_mode=self.attn_mode,
                    **factory_kwargs
                )
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs
        )

        self.depth_double_blocks = depth_double_blocks
        self.depth_single_blocks = depth_single_blocks

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor, # Should be in range(0, 1000).
            text_states: torch.Tensor = None,
            text_mask: torch.Tensor = None, # Now we don't use it.
            text_states_2: Optional[torch.Tensor] = None, # Text embedding for modulation.
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
            guidance: torch.Tensor = None, # Guidance for modulation, should be cfg_scale x 1000.
            return_dict: bool = True,
            generator = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        """
        x: [b, c, f, h, w] [2, 16, 1, 32, 32]
        t: [b]
        text_states: [b, l, h]
        text_mask: [b, l]
        freqs_cos: [s, d]
        freqs_sin: [s, d]

        NOTE:
            b - batch_size
            c - channel
            f - frame
            h - height
            w - weight
            s - img_len
            h - hidden_size
            l - txt_len
            a - num_heads
            d - head_dim
        """
        out = {}
        # [b, c, f, h, w] or [b, c, h, w] for 2D images (Flux with vanilla mode)
        img = x
        # Handle both 4D (image) and 5D (video) inputs
        # For Flux (use_vanilla=True), input may be 4D from 2D VAE, need to add time dimension
        is_image_input = len(x.shape) == 4
        if is_image_input:
            # [b, c, h, w] -> [b, c, 1, h, w]
            img = img.unsqueeze(2)
        # [b, l, h]
        txt = text_states
        _, _, ot, oh, ow = img.shape
        tt, th, tw = ot // self.patch_size[0], oh // self.patch_size[1], ow // self.patch_size[2]

        # Prepare modulation vectors.
        # time modulation
        if self.use_vanilla:
            t = timestep_embedding(t, 256).to(img.dtype)
            vec = self.time_in(t)
        else:
            # [b, h]
            vec = self.time_in(t)

        # text modulation
        if self.use_vanilla:
            if self.text_pool_type is None:
                pass
            else:
                vec = vec + self.vector_in(text_states_2)
        else:
            if self.text_pool_type is not None:
                if self.text_pool_type == "clip":
                    # [b, h]
                    vec = vec + self.vector_in(text_states_2)
                else:
                    raise NotImplementedError(f"Unsupported text_pool_type: {self.text_pool_type}")
            else:
                pass

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            if self.use_vanilla:
                guidance = timestep_embedding(guidance, 256).to(img.dtype)
                vec = vec + self.guidance_in(guidance)
            else:
                # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
                vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        if self.use_vanilla:
            # Use all three dimensions for patching: temporal (pt), height (ph), width (pw)
            img = rearrange(img, "b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw)",
                          pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2])
            img = self.img_in(img)
            txt = self.txt_in(txt, generator)
        else:
            # [b, c, f, h, w] -> [b, s, h]
            img = self.img_in(img)
            txt = self.txt_in(txt, generator)

        # Compute 'self attention mask'.
        # Added by ckczzjzhang
        # [b, 1, s+l, s+l]
        attn_mask = None
        if self.use_attention_mask and self.attn_mode == "torch":
            batch_size = img.shape[0]
            txt_seq_len = txt.shape[1]
            img_seq_len = img.shape[1]
            seq_len = txt_seq_len+img_seq_len
            # [b, s]
            img_mask = torch.ones(batch_size, img_seq_len, dtype=torch.int64, device=img.device)
            # [b, s+l]
            img_txt_mask = torch.cat([img_mask, text_mask], dim=1)
            # [b, s+l] -> [b, 1, 1, s+l] -> [b, 1, s+l, s+l]
            attn_mask_1 = img_txt_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            # [b, 1, s+l, s+l] -> [b, 1, s+l, s+l]
            attn_mask_2 = attn_mask_1.transpose(2, 3)
            # [b, 1, s+l, s+l], 1 for broadcasting of num_heads
            attn_mask = (attn_mask_1 & attn_mask_2).bool()
            # avoids self-attention weight being NaN for text padding tokens
            attn_mask[:, :, :, 0]= True

        cu_seqlens_q, cu_seqlens_kv = None, None
        cu_seqlens_q_padded, cu_seqlens_kv_padded = None, None
        max_seqlen_q, max_seqlen_kv = None, None
        if self.attn_mode == "flash":
            img_len = img.shape[1]
            if not self.use_attention_mask:
                text_len = txt.shape[1]
                batch_size = img.shape[0]
                text_mask = torch.ones((batch_size, text_len)).to(torch.int32).to(torch.device("cuda"))
            else:
                text_len = text_mask.shape[1]

            if self.use_flash_attn:
                cu_seqlens_q = get_cu_seqlens(text_mask, img_len)
                cu_seqlens_kv = cu_seqlens_q

            if self.use_fused_attn:
                batch_size = img.shape[0]
                img_mask = torch.ones((batch_size, img_len)).to(torch.int32).to(torch.device("cuda"))
                attn_mask = torch.cat((img_mask, text_mask), dim=1)
                attn_mask = (1 - attn_mask).to(torch.bool)
                cu_seqlens_q = get_cu_seqlens_fused(attn_mask)
                cu_seqlens_kv = cu_seqlens_q

            cu_seqlens_q_padded = cu_seqlens_q
            cu_seqlens_kv_padded = cu_seqlens_kv

            max_seqlen_q = img_len + text_len
            max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        img_len = img.shape[1]

        # --------------------- Pass through DiT blocks ------------------------
        for block in self.double_blocks:
            double_block_args = [
                img,
                txt,
                vec,
                attn_mask,
                self.self_attn_mask_type,
                cu_seqlens_q,
                cu_seqlens_kv,
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis
            ]
            img, txt = block(*double_block_args)

        # Compatible with MMDiT.
        if len(self.single_blocks) > 0:
            x = torch.cat((img, txt), dim=1)
            for block in self.single_blocks:
                single_block_args = [
                    x,
                    vec,
                    img_len,
                    attn_mask,
                    self.self_attn_mask_type,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis
                ]
                x = block(*single_block_args)
            img = x[:, :img.shape[1], ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        img = self.unpatchify(img, tt, th, tw)
        img = img.float()
        if return_dict:
            out['x'] = img
            return out
        return img

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum('nthwcopq->nctohpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs


    def params_count(self):
        counts = {
            "double": sum([
                sum(p.numel() for p in block.img_attn_qkv.parameters()) +
                sum(p.numel() for p in block.img_attn_proj.parameters()) +
                sum(p.numel() for p in block.img_mlp.parameters()) +
                sum(p.numel() for p in block.txt_attn_qkv.parameters()) +
                sum(p.numel() for p in block.txt_attn_proj.parameters()) +
                sum(p.numel() for p in block.txt_mlp.parameters())
                for block in self.double_blocks
            ]),
            "single": sum([
                sum(p.numel() for p in block.linear1.parameters()) +
                sum(p.numel() for p in block.linear2.parameters())
                for block in self.single_blocks
            ]),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts

    def set_input_tensor(self, input_tensor):
        pass

#################################################################################
#                     UniCom Diffusion Transformer Configs                        #
#################################################################################

UNICOM_DIFFUSION_TRANSFORMER_CONFIGS = {
    'Flux-dev': {
        'use_vanilla': True,
        'guidance_embed': True,
        'patch_size': [1, 2, 2],
        'hidden_size': 3072,
        'mlp_ratio': 4,
        'num_heads': 24,
        'depth_double_blocks': 19,
        'depth_single_blocks': 38,
        'rope_dim_list': [16, 56, 56],
        'qkv_bias': True,
    }
}
