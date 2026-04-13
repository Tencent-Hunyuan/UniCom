import torch
import torch.nn as nn
from decoder.runtime_context import get_runtime_args


class FP32_SiLU(nn.SiLU):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.silu(input.float(), inplace=False).to(input.dtype)


class FP32_GELU(nn.GELU):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(input.float(), approximate=self.approximate).to(input.dtype)


def get_activation_layer(act_type):
    args = get_runtime_args()
    if act_type == 'gelu' and args.force_fp32_ops:
        return lambda: FP32_GELU()
    elif act_type == 'gelu':
        return lambda: nn.GELU()
    elif act_type == 'gelu_tanh' and args.force_fp32_ops:
        return lambda: FP32_GELU(approximate='tanh')
    elif act_type == 'gelu_tanh':
        # Approximate `tanh` requires torch >= 1.13
        return lambda: nn.GELU(approximate='tanh')
    elif act_type == 'relu':
        return nn.ReLU
    elif act_type == 'silu' and args.force_fp32_ops:
        return FP32_SiLU
    elif act_type == 'silu':
        return nn.SiLU
    else:
        raise ValueError(f"Unknown activation type: {act_type}")
