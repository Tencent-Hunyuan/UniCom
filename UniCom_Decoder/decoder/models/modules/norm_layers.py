import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder.runtime_context import get_runtime_args


class RMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine=True, eps: float = 1e-6,
                 device=None, dtype=None):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class FP32_Layernorm(nn.LayerNorm):
    def __init__(self, hidden_size,  eps:float =1e-5, elementwise_affine: bool =True, bias: bool =True, device = None, dtype = None ):
        super().__init__(hidden_size, eps, elementwise_affine, bias, device, dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        origin_dtype = input.dtype
        return F.layer_norm(input.float(), self.normalized_shape,
                            self.weight.float() if self.weight!=None else None,
                            self.bias.float() if self.bias!=None else None,
                            self.eps).to(origin_dtype)

def get_norm_layer(norm_layer):
    """
    Get the normalization layer.

    Args:
        norm_layer (str): The type of normalization layer.

    Returns:
        norm_layer (nn.Module): The normalization layer.
    """
    args = get_runtime_args()
    if norm_layer == "layer" and args.force_fp32_ops:
        return FP32_Layernorm
    elif norm_layer == "layer":
        return nn.LayerNorm
    elif norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")
