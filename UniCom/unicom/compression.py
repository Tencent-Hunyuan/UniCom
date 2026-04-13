import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if in_dim > out_dim:
            self.head_dim = out_dim
            self.num_heads = in_dim // self.head_dim
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(in_dim))
            self.v_bias = nn.Parameter(torch.zeros(in_dim))
            self.register_buffer('zero_k_bias', torch.zeros(in_dim))
        else:
            self.num_heads = num_heads
            self.head_dim = out_dim // num_heads
            self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(out_dim))
            self.v_bias = nn.Parameter(torch.zeros(out_dim))
            self.register_buffer('zero_k_bias', torch.zeros(out_dim))

        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(input=x, weight=self.qkv.weight,
                        bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)

        if self.in_dim > self.out_dim:
            x = torch.mean(x, dim=1)
            if self.in_dim // self.num_heads != self.out_dim:
                x = F.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        return x


class GeGluMLP(nn.Module):
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
    def __init__(self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm,
                 mlp_ratio=2, act_layer=nn.SiLU):
        super().__init__()
        self.norm1 = norm_layer(in_dim)
        self.attn = PlainAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = norm_layer(in_dim)
        self.norm2 = norm_layer(out_dim)
        self.mlp = GeGluMLP(
            in_features=out_dim,
            hidden_features=int(out_dim * mlp_ratio),
            act_layer=act_layer,
        )

    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionCompressionNet(nn.Module):
    """Attention-based compression: 1152 -> 64 -> 1152"""
    def __init__(self, original_dim=1152, compress_dim=64, checkpoint_path=None):
        super().__init__()
        self.original_dim = original_dim
        self.compress_dim = compress_dim

        num_heads = original_dim // compress_dim
        self.compress_attn = AttnProjection(
            in_dim=original_dim, out_dim=compress_dim, num_heads=num_heads,
        )
        self.expand_attn = AttnProjection(
            in_dim=compress_dim, out_dim=original_dim, num_heads=num_heads,
        )

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path):
        print(f"Loading compression net from {path}")
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt, strict=True)

    def compress(self, x):
        return self.compress_attn(x)

    def decompress(self, x):
        return self.expand_attn(x)

    def forward(self, x):
        return self.decompress(self.compress(x))


__all__ = ["AttentionCompressionNet"]
