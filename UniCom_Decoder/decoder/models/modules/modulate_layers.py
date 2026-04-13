from typing import Callable

import torch
import torch.nn as nn


class ModulateDiT(nn.Module):
    def __init__(self, hidden_size: int, factor: int, act_layer: Callable, dtype=None, device=None):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(hidden_size, factor * hidden_size, bias=True, **factory_kwargs)
        # Zero-initialize the modulation
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(x))


def modulate(x, shift=None, scale=None, unsqueeze_dim=1):
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(unsqueeze_dim))
    elif scale is None:
        return x + shift.unsqueeze(unsqueeze_dim)
    else:
        return x * (1 + scale.unsqueeze(unsqueeze_dim)) + shift.unsqueeze(unsqueeze_dim)


def apply_gate(x, gate=None, tanh=False, unsqueeze_dim=1):
    if gate is None:
        return x
    if tanh:
        return x * gate.unsqueeze(unsqueeze_dim).tanh()
    else:
        return x * gate.unsqueeze(unsqueeze_dim)


def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward
