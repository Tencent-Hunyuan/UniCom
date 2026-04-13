import torch
import torch.nn as nn
import torch.nn.functional as F


def get_distributed_world_size(group=None):
    return 1


class DotProductAttention(nn.Module):
    def __init__(
        self,
        num_attention_heads,
        kv_channels,
        num_gqa_groups,
        attention_dropout,
        qkv_format,
        tp_size,
        get_rng_state_tracker,
        sequence_parallel,
        tp_group,
        layer_number,
        attention_type,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = kv_channels

    def forward(self, q, k, v, **kwargs):
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return out.permute(2, 0, 1, 3).contiguous().view(out.shape[2], out.shape[0], -1)


class UnsupportedTransformerEngineBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError("TransformerEngine blocks are not supported in the standalone decoder runtime.")
