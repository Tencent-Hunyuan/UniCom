import math
import random
import re
import time
import warnings
from dataclasses import dataclass
from typing import List, Union, Optional, Dict, Any, Tuple, Callable, TYPE_CHECKING
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.cuda import nvtx

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, StaticCache
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import TextStreamer
from transformers.generation.utils import GenerationMixin, GenerationConfig, ALL_CACHE_NAMES
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
)

from .compression import AttentionCompressionNet
from .configuration_unicom import UniComConfig
from .unicom_pipeline import UniComText2ImagePipeline, FlowMatchDiscreteScheduler
from .image_processor import UniComImageProcessor
from .siglip2 import Siglip2VisionTransformer, LightProjector
from .tokenization_unicom import UniComTokenizerFast, ImageInfo, ImageTensor, CondImage
from .system_prompt import get_system_prompt

from .cache_utils import TaylorCacheContainer, CacheWithFreqsContainer, set_compile_enabled

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func

# Type aliases
BatchRaggedImages = Union[torch.Tensor, List[Union[torch.Tensor, List[torch.Tensor]]]]
BatchRaggedTensor = Union[torch.Tensor, List[torch.Tensor]]
InputImage = Optional[Union[Image.Image, str, bytes]]


def get_device(tensor: BatchRaggedImages):
    if isinstance(tensor, torch.Tensor):
        return tensor.device
    elif isinstance(tensor, list):
        return get_device(tensor[0])
    else:
        raise ValueError(f"Unsupported type for get_device: {type(tensor)}")


_CONFIG_FOR_DOC = "UniComConfig"

UniCom_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`UniComConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# =======================================================
#     Helper Functions
# =======================================================

def default(val, d):
    return val if val is not None else d


def to_device(data, device):
    if device is None:
        return data
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    else:
        return data


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def real_batched_index_select(t, dim, idx):
    """ index_select for batched index and batched t """
    assert t.ndim >= 2 and idx.ndim >= 2, f"{t.ndim=} {idx.ndim=}"
    assert len(t) == len(idx), f"{len(t)=} != {len(idx)=}"
    return torch.stack([torch.index_select(t[i], dim - 1, idx[i]) for i in range(len(t))])


# =======================================================
#     Module Functions
# =======================================================

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


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2, device="cpu"):
    """
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
            should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
            n-tuples.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid (np.ndarray): [dim, ...]
    """
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
        # assert num are all integers
        num_int = [int(x) for x in num]
        assert (torch.tensor(num) == torch.tensor(num_int)).all(), f"num should be int, but got {num}"
        num = num_int
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)       # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)      # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)       # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32, device=device)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")   # dim x [H, W]
    grid = torch.stack(grid, dim=0)     # [dim, H, W]

    return grid


def build_2d_rope(
        seq_len: int, n_elem: int, image_infos: Optional[List[Tuple[slice, Tuple[int, int]]]] = None,
        device: Optional[torch.device] = None, base: int = 10000, base_rescale_factor: float = 1.0,
        return_all_pos: bool = False,
):
    """
    Reference: https://kexue.fm/archives/10352

    Start from 1, we have
        beta_y = L + (wh - h)/2
        beta_x = L + (wh - w)/2

    Returns
    -------
    cos: torch.Tensor with shape of [seq_len, n_elem]
    sin: torch.Tensor with shape of [seq_len, n_elem]
    """
    assert n_elem % 4 == 0, f"n_elem must be divisible by 4, but got {n_elem}."

    # theta
    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (n_elem / (n_elem - 2))
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    theta = theta.reshape(1, n_elem // 4, 2)    # [1, half_d, 2]

    # position indices
    if image_infos is None:
        image_infos = []

    image_infos_list = [image_infos]
    sample_seq_lens = [seq_len]

    # Prepare position indices for each sample
    x_sections = []
    y_sections = []
    for sample_id, sample_image_infos in enumerate(image_infos_list):
        last_pos = 0
        for sec_slice, (h, w) in sample_image_infos:
            L = sec_slice.start   # start from 0, so image_slice.start is just L
            # previous text
            if last_pos < L:
                y_sections.append(torch.arange(last_pos, L, device=device))
                x_sections.append(torch.arange(last_pos, L, device=device))
            elif h is None:
                # Interleave data has overlapped positions for <boi> <size> <ratio> <timestep> <eoi> tokens.
                y_sections.append(torch.arange(sec_slice.start, sec_slice.stop, device=device))
                x_sections.append(torch.arange(sec_slice.start, sec_slice.stop, device=device))
                continue
            else:
                # Interleave data has overlapped positions for noised image and the successive clean image,
                # leading to last_pos (= last text end L + noise w * h) > L (last text end L).
                pass
            # current image
            beta_y = L + (w * h - h) / 2
            beta_x = L + (w * h - w) / 2
            grid = get_meshgrid_nd((beta_y, beta_x), (beta_y + h, beta_x + w), device=device)  # [2, h, w]
            grid = grid.reshape(2, -1)  # (y, x)
            y_sections.append(grid[0])
            x_sections.append(grid[1])
            # step
            last_pos = L + w * h
        # final text
        y_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id], device=device))
        x_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id], device=device))

    x_pos = torch.cat(x_sections).long()
    y_pos = torch.cat(y_sections).long()
    # If there are overlap positions, we need to remove them.
    x_pos = x_pos[:seq_len]
    y_pos = y_pos[:seq_len]
    all_pos = torch.stack((y_pos, x_pos), dim=1).unsqueeze(1).to(device)    # [seq_len, 1, 2]

    # calc rope
    idx_theta = (all_pos * theta).reshape(all_pos.shape[0], n_elem // 2).repeat(1, 2)

    cos = torch.cos(idx_theta)
    sin = torch.sin(idx_theta)

    if return_all_pos:
        return cos, sin, all_pos

    return cos, sin


def build_batch_2d_rope(
        seq_len: int, n_elem: int, image_infos: Optional[List[List[Tuple[slice, Tuple[int, int]]]]] = None,
        device: Optional[torch.device] = None, base: int = 10000, base_rescale_factor: float = 1.0,
        return_all_pos: bool = False,
):
    cos_list, sin_list, all_pos_list = [], [], []
    if image_infos is None:
        image_infos = [None]
    for i, image_info in enumerate(image_infos):
        res = build_2d_rope(
            seq_len, n_elem, image_infos=image_info, device=device,
            base=base, base_rescale_factor=base_rescale_factor,
            return_all_pos=return_all_pos,
        )
        if return_all_pos:
            cos, sin, all_pos = res
        else:
            cos, sin = res
            all_pos = None
        cos_list.append(cos)
        sin_list.append(sin)
        all_pos_list.append(all_pos)

    stacked_cos = torch.stack(cos_list, dim=0)
    stacked_sin = torch.stack(sin_list, dim=0)

    if return_all_pos:
        return stacked_cos, stacked_sin, all_pos_list

    return stacked_cos, stacked_sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass shifted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =======================================================
#     Modules for Image Generation
# =======================================================

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self,
                 hidden_size,
                 act_layer=nn.GELU,
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


class LinearPatchEmbed(nn.Module):
    """Linear projection for SigLIP latent tokens: (B, N, in_chans) -> (B, N, embed_dim)"""
    def __init__(self, in_chans, embed_dim, act_layer=nn.SiLU, bias=True,
                 device=None, dtype=None):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias, **factory_kwargs)
        self.norm_final = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6,
                                       **factory_kwargs)
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True, **factory_kwargs)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, t):
        x = self.proj(x)
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        dtype = x.dtype
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x.to(dtype), 0, 0


class LinearFinalLayer(nn.Module):
    """Linear projection back to SigLIP latent space: (B, N, hidden) -> (B, N, out_channels)"""
    def __init__(self, hidden_size, out_channels, act_layer=nn.SiLU,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6,
                                       **factory_kwargs)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True, **factory_kwargs)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, t, token_h=0, token_w=0):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


# =======================================================
#     Modules for Transformer Backbone
# =======================================================

@dataclass
class CausalMMOutputWithPast(CausalLMOutputWithPast):
    diffusion_prediction: Optional[torch.Tensor] = None


class UniComStaticCache(StaticCache):
    """
    A custom static cache for multi-modal models that supports dynamic extension of the cache
    and inplace updates of the cache.

    This cache supports batch cache_position updates.
    """
    def __init__(self, *args, **kwargs):
        self.dynamic = kwargs.pop("dynamic", False)
        super().__init__(*args, **kwargs)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        if self.layers[layer_idx].keys is None:
            self.layers[layer_idx].lazy_initialization(key_states)
        k_out = self.layers[layer_idx].keys
        v_out = self.layers[layer_idx].values

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
            # operation, that avoids copies and uses less memory.
            if cache_position.dim() == 1:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)

                if self.dynamic:
                    end = cache_position[-1].item() + 1
                    k_out = k_out[:, :, :end]
                    v_out = v_out[:, :, :end]
            else:
                assert cache_position.dim() == 2, f"multiple batch dims not yet {cache_position.shape=}"
                batch_size, idx_size = cache_position.shape
                assert batch_size == k_out.size(0)
                assert batch_size == v_out.size(0)
                assert batch_size == key_states.size(0)
                assert batch_size == value_states.size(0)
                for i in range(batch_size):
                    unbatched_dim = 1
                    k_out[i].index_copy_(unbatched_dim, cache_position[i], key_states[i])
                    v_out[i].index_copy_(unbatched_dim, cache_position[i], value_states[i])

                if self.dynamic:
                    assert len(cache_position) == 1
                    end = cache_position[0, -1].item() + 1
                    k_out = k_out[:, :, :end]
                    v_out = v_out[:, :, :end]

        return k_out, v_out


class CachedRoPE(object):
    """ A 2D RoPE is determined by rope_image_info and seq_len. """

    def __init__(self, config):
        self.config = config
        self.cos_cache = None
        self.sin_cache = None
        self.seq_len = None
        self.rope_image_info = None

    def __call__(self, seq_len, device, rope_image_info=None, position_ids=None):
        """ Get cached RoPE for given seq_len and rope_image_info.
        If cache miss, compute and cache it.

        Args:
            seq_len (int): The sequence length.
            device (torch.device): The device to store the RoPE.
            rope_image_info (list): The rope image info. list of lists of (slice, (height, width)) tuples.
            position_ids (torch.Tensor): The input positions.

        Returns:
            The RoPE cos and sin tensors.
        """
        if (self.seq_len != seq_len) or (rope_image_info is not None and self.rope_image_info != rope_image_info):
            # Cache miss, compute RoPE
            if self.config.rope_type in ["2d", "default"]:
                self.cos_cache, self.sin_cache = build_batch_2d_rope(
                    image_infos=rope_image_info,
                    seq_len=seq_len,
                    n_elem=self.config.attention_head_dim,
                    device=device,
                    base=self.config.rope_theta,
                )
            else:
                raise NotImplementedError(f"rope_type `{self.config.rope_type}` not supported")
        else:
            # hit cache
            pass

        if position_ids is None:
            # Typically for training
            cos, sin = self.cos_cache, self.sin_cache
        else:
            # Typically for inference
            assert position_ids.dim() == 2, f"{position_ids.shape=}"
            bsz = position_ids.size(0)
            head_size = self.cos_cache.size(-1)
            # Expand batch dim: 1D RoPE cache has batch=1, but CFG uses batch=2
            cos_cache = self.cos_cache.expand(bsz, -1, -1)
            sin_cache = self.sin_cache.expand(bsz, -1, -1)
            cos = torch.gather(cos_cache, dim=1, index=position_ids.unsqueeze(-1).expand(-1, -1, head_size))
            sin = torch.gather(sin_cache, dim=1, index=position_ids.unsqueeze(-1).expand(-1, -1, head_size))

        return cos, sin


class UniComRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, cast_weight_fp32=False):
        """
        UniComRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.cast_weight_fp32 = cast_weight_fp32

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.cast_weight_fp32:
            return (self.weight.float() * hidden_states).to(input_dtype)
        else:
            return self.weight * hidden_states.to(input_dtype)


class UniComMLP(nn.Module):
    def __init__(self, config: UniComConfig, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act

        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        if self.hidden_act == "silu":
            self.intermediate_size *= 2  # SwiGLU
            self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size // 2, self.hidden_size, bias=config.mlp_bias)
        elif self.hidden_act == "gelu":
            self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        else:
            assert False, "other hidden_act are not supported"

    def forward(self, x):
        if self.hidden_act == "silu":
            gate_and_up_proj = self.gate_and_up_proj(x)
            x1, x2 = gate_and_up_proj.chunk(2, dim=-1)
            down_proj = self.down_proj(self.act_fn(x1) * x2)
            return down_proj
        elif self.hidden_act == "gelu":
            intermediate = self.gate_and_up_proj(x)
            intermediate = self.act_fn(intermediate)
            output = self.down_proj(intermediate)
            return output
        else:
            assert False, "other hidden_act are not supported"


class UniComSDPAAttention(nn.Module):
    """PyTorch SDPA attention implementation using torch.nn.functional.scaled_dot_product_attention"""

    def __init__(self, config: UniComConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = 'self'

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # self.head_dim = self.hidden_size // self.num_heads
        self.head_dim: int = config.attention_head_dim
        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads else self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.use_qk_norm = config.use_qk_norm
        self.use_rotary_pos_emb = config.use_rotary_pos_emb
        self.hidden_size_q = self.head_dim * self.num_heads
        self.hidden_size_kv = self.head_dim * self.num_key_value_heads

        # define layers
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size_q + 2 * self.hidden_size_kv,
            bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.hidden_size_q, self.hidden_size, bias=config.attention_o_proj_bias)

        if self.use_qk_norm:
            self.query_layernorm = UniComRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = UniComRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if self.use_rotary_pos_emb:
            self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = None
            return
        scaling_type = self.config.rope_scaling["type"]
        if scaling_type == "custom":
            self.rotary_emb = None
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: Optional[bool] = False,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if output_attentions:
            raise NotImplementedError(
                'UniComModel is using UniComSDPAAttention,'
                'but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`.'
            )

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.reshape(bsz, q_len, self.num_key_value_heads, self.num_key_value_groups + 2,
                                        self.head_dim)
        query_states, key_states, value_states = torch.split(qkv_states, [self.num_key_value_groups, 1, 1], dim=3)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary_pos_emb:
            cos, sin = custom_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            query_states = query_states.to(key_states.dtype)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with
        # custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


UniCom_ATTENTION_CLASSES = {
    "eager": UniComSDPAAttention,
    "sdpa": UniComSDPAAttention,
}


class UniComDecoderLayer(nn.Module):
    def __init__(self, config: UniComConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        attn_impl = config._attn_implementation     # noqa
        if attn_impl in UniCom_ATTENTION_CLASSES:
            self.self_attn = UniCom_ATTENTION_CLASSES[attn_impl](config=config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unsupported attention implementation: {attn_impl}")

        self.mlp = UniComMLP(config, layer_idx=layer_idx)
        if config.norm_type == 'hf_rms' or config.norm_type == 'rms':
            self.input_layernorm = UniComRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = UniComRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        elif config.norm_type == 'fused' or config.norm_type == 'torch_nn':
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            assert False, "other norm_type are not supported"

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor | Any]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            custom_pos_emb (`Tuple[torch.FloatTensor]`, *optional*): custom position embedding for rotary
                position embedding
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use "
                "`attention_mask` instead.`"
            )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            custom_pos_emb=custom_pos_emb,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@add_start_docstrings(
    "The bare UniCom Model outputting raw hidden-states without any specific head on top.",
    UniCom_START_DOCSTRING,
)
class UniComPreTrainedModel(PreTrainedModel):
    config_class = UniComConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["UniComDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


UniCom_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare UniCom Model outputting raw hidden-states without any specific head on top.",
    UniCom_START_DOCSTRING,
)
class UniComModel(UniComPreTrainedModel):
    def __init__(self, config: UniComConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.add_classification_head = config.add_classification_head
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [UniComDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if not config.add_classification_head:
            self.ln_f = UniComRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

        self.shared_tensor = None

    @add_start_docstrings_to_model_forward(UniCom_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            mode: str = "gen_text",
            first_step: Optional[bool] = None,
            post_token_len: int = None,
            num_image_tokens: int = None,
            gen_timestep_scatter_index: Optional[torch.Tensor] = None,
            num_special_tokens: int = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                custom_pos_emb=custom_pos_emb,
                mode=mode,
                first_step=first_step,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if not self.add_classification_head:
            # Do ln_f outside of the model for compatibility with image generation.
            pass
            # hidden_states = self.ln_f(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class UniComForCausalMM(UniComPreTrainedModel, GenerationMixin):
    def __init__(self, config: UniComConfig, skip_load_module: set[str] = {}):
        """UniComForCausalMM

        Args:
            config (UniComConfig): model config to initialize the model
            skip_load_module (set[str], optional):
                modules to skip loading, used for vllm inference. Defaults to {}.
        """
        super().__init__(config)
        self.config = config
        set_compile_enabled(getattr(config, "use_compile", False))
        self._tokenizer: Optional[UniComTokenizerFast] = None

        self.image_processor = UniComImageProcessor(config)

        if 'all' in skip_load_module:
            skip_load_module = {
                'compression',
                'vit',
                'timestep_emb',
                'patch_embed',
                'time_embed',
                'final_layer',
                'transformers',
            }

        if 'compression' not in skip_load_module:
            self.compression = AttentionCompressionNet(
                original_dim=config.siglip_original_dim,
                compress_dim=config.siglip_latent_dim,
                checkpoint_path=config.compression_checkpoint_path,
            )
            self.compression.eval()
            for param in self.compression.parameters():
                param.requires_grad = False

        self._pipeline = None

        if 'vit' not in skip_load_module:
            self.vision_model = Siglip2VisionTransformer(config.vit)
            self.vision_aligner = LightProjector(config.vit_aligner)

        if 'timestep_emb' not in skip_load_module:
            self.timestep_emb = TimestepEmbedder(hidden_size=config.hidden_size)

        if self.config.cfg_distilled:
            self.guidance_emb = TimestepEmbedder(hidden_size=config.hidden_size)
        if self.config.use_meanflow:
            self.timestep_r_emb = TimestepEmbedder(hidden_size=config.hidden_size)

        latent_dim = config.siglip_latent_dim
        if 'patch_embed' not in skip_load_module:
            self.patch_embed = LinearPatchEmbed(
                in_chans=latent_dim,
                embed_dim=config.hidden_size,
                act_layer=nn.SiLU,
            )
        if 'time_embed' not in skip_load_module:
            self.time_embed = TimestepEmbedder(hidden_size=config.hidden_size)
        if 'final_layer' not in skip_load_module:
            self.final_layer = LinearFinalLayer(
                hidden_size=config.hidden_size,
                out_channels=latent_dim,
                act_layer=nn.SiLU,
            )
        # time_embed_2 only exists in MoE models (use_final_time_embed=True).
        # For SigLIP2 models, the final layer reuses self.time_embed.

        if 'transformers' not in skip_load_module:
            # transformer backbone
            self.model = UniComModel(config)
            # linear head
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size

        # Taylor Cache
        self.use_taylor_cache = False

        self.num_image_tokens = None
        self.num_special_tokens = None
        # Initialize cached rope, supporting automatic cache update
        self.cached_rope = CachedRoPE(config)

        # Initialize weights and apply final processing
        self.post_init()

        if not getattr(config, "use_compile", False):
            # Keep inference on the eager path unless compile is explicitly enabled.
            self.forward = torch._dynamo.disable(self.forward)
            if hasattr(self, "model"):
                self.model.forward = torch._dynamo.disable(self.model.forward)

    @classmethod
    def from_config(cls, config: UniComConfig, skip_load_module: set[str] = {}):
        return cls(config, skip_load_module=skip_load_module)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise ValueError("Attribute `tokenizer` has not been initialized yet. Please set it first.")
        return self._tokenizer

    def load_tokenizer(self, tokenizer_path):
        """Load tokenizer and ensure all training special tokens are present.

        If the tokenizer directory already contains all special tokens (e.g., HF
        model directory produced by convert_checkpoint.py), they are loaded directly.
        Otherwise (e.g., base tokenizer from training), the tokens are added here.
        """
        self._tokenizer = UniComTokenizerFast.from_pretrained(
            tokenizer_path, model_version=self.config.model_version,
        )

        # Qwen2 has no bos_token; training uses <|im_start|> as fallback
        if self._tokenizer.bos_token is None:
            self._tokenizer.bos_token = "<|im_start|>"

        expected_vocab = self.config.vocab_size
        if len(self._tokenizer) < expected_vocab:
            special_tokens = {
                "additional_special_tokens": [
                    "<boi>", "<eoi>", "<boa>", "<eoa>", "<bov>", "<eov>",
                    "<img>", "<audio>", "<video>",
                    "<pad>", "<cfg>", "<iw>", "<ih>",
                    "<bot>", "<eot>", "<text>", "<mask>", "<timestep>",
                    "<recaption>", "</recaption>",
                    "<bof>", "<eof>", "<face>",
                    "<think>", "</think>", "<answer>", "</answer>",
                    "<und_boi>", "<und_eoi>",
                    "<src_boi>", "<src_eoi>", "<gen_boi>", "<gen_eoi>", "<gen_img>",
                ] + [f"<img_size_{sz}>" for sz in [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
                ] + [f"<img_ratio_{i}>" for i in range(33)
                ] + ["<joint_img_sep>",
                ] + ["<ref>", "</ref>", "<quad>", "</quad>",
                ] + [f"<pos_x_{x}>" for x in range(1000)
                ] + [f"<pos_y_{y}>" for y in range(1000)
                ] + ["<guidance>",
                ] + [f"<relation_{i}>" for i in range(10)
                ] + [f"</relation_{i}>" for i in range(10)
                ]
            }
            n_added = self._tokenizer.add_special_tokens(special_tokens)
            if n_added > 0:
                self._tokenizer.setup_special_tokens()
                print(f"  Added {n_added} special tokens -> vocab {len(self._tokenizer)}")

    @property
    def pipeline(self):
        if self._pipeline is None:
            self.scheduler = FlowMatchDiscreteScheduler(
                shift=self.generation_config.flow_shift, reverse=True, solver="euler",
            )
            self._pipeline = UniComText2ImagePipeline(
                model=self, scheduler=self.scheduler,
            )
        return self._pipeline

    def instantiate_image_tokens(
            self,
            hidden_states: torch.Tensor,
            timesteps: BatchRaggedTensor,
            images: BatchRaggedImages,
            image_mask: torch.Tensor,
            guidance: torch.Tensor = None,
            timesteps_r: torch.Tensor = None,
    ):
        """Instantiate image embeddings into the input embedding sequence.
        SigLIP compressed latents: (B, N, C) where C = siglip_latent_dim.
        """
        if hidden_states is None:
            t_emb = self.time_embed(timesteps)
            image_emb = self.patch_embed(images, t_emb)[0]
            timestep_emb = self.timestep_emb(timesteps).reshape(images.size(0), -1, self.config.hidden_size)
            cat_list = [timestep_emb, image_emb]

            if guidance is not None:
                guidance_src = self.guidance_emb(guidance.reshape(-1))
                guidance_emb = guidance_src.reshape(images.size(0), -1, self.config.hidden_size)
            if timesteps_r is not None:
                timesteps_r_src = self.timestep_r_emb(timesteps_r.reshape(-1))
                timesteps_r_emb = timesteps_r_src.reshape(images.size(0), -1, self.config.hidden_size)

            if guidance is not None and timesteps_r is not None:
                cat_list = [timestep_emb, guidance_emb, timesteps_r_emb, image_emb]
            elif guidance is not None:
                cat_list = [timestep_emb, guidance_emb, image_emb]
            elif timesteps_r is not None:
                cat_list = [timestep_emb, timesteps_r_emb, image_emb]
            hidden_states = torch.cat(cat_list, dim=1)
            return hidden_states

        bsz, seqlen, n_embd = hidden_states.shape
        device = hidden_states.device
        assert isinstance(images, (torch.Tensor, list)), f"images should be BatchRaggedImages, got {type(images)}"

        if isinstance(images, torch.Tensor):
            assert isinstance(timesteps, torch.Tensor), f"timesteps should be 1-D tensor, got {type(timesteps)}"
            index = torch.arange(seqlen, device=device).unsqueeze(0).repeat(bsz, 1)
            t_emb = self.time_embed(timesteps.to(device))
            image_seq, token_h, token_w = self.patch_embed(images.to(device), t_emb)
            image_scatter_index = index.masked_select(image_mask.to(device).bool()).reshape(bsz, -1)
            hidden_states.scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=image_seq,
            )

        else:   # list
            index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
            for i, (image_i, t_i) in enumerate(zip(images, timesteps)):
                t_i_emb = self.time_embed(t_i)

                if isinstance(image_i, torch.Tensor):
                    image_i_seq, _, _ = self.patch_embed(image_i, t_i_emb)

                elif isinstance(image_i, list):
                    image_i_seq_list = []
                    for j in range(len(image_i)):
                        image_ij = image_i[j]
                        if image_ij.ndim == 2:
                            image_ij = image_ij.unsqueeze(0)
                        image_i_seq_j = self.patch_embed(image_ij, t_i_emb[j:j + 1])[0]
                        image_i_seq_list.append(image_i_seq_j)
                    image_i_seq = torch.cat(image_i_seq_list, dim=1)

                else:
                    raise TypeError(f"image_i should be a torch.Tensor or a list, got {type(image_i)}")

                image_i_index = index[i:i + 1].masked_select(image_mask[i:i + 1].bool()).reshape(1, -1)
                hidden_states[i:i + 1].scatter_(
                    dim=1,
                    index=image_i_index.unsqueeze(-1).repeat(1, 1, n_embd),
                    src=image_i_seq.reshape(1, -1, n_embd),  # (1, img_seqlen, n_embd)
                )

        return hidden_states

    def instantiate_src_image_tokens(
            self,
            hidden_states: torch.Tensor,
            src_x,
            src_t,
            src_image_mask: torch.Tensor,
    ):
        """Embed source image compressed latents (64-dim) into the sequence at src_image_mask positions.

        Follows the original transfusion._instantiate_ragged_image_tokens:
        src_x/src_t are lists (per batch item), each element [n_images, tokens, 64] / [n_images].
        Each image gets its own timestep embedding via patch_embed, then all are scattered.
        """
        if src_x is None:
            return hidden_states

        bsz, seqlen, n_embd = hidden_states.shape
        device = hidden_states.device
        src_image_mask = src_image_mask.to(device).bool()
        index = torch.arange(seqlen, device=device).unsqueeze(0).repeat(bsz, 1)

        if isinstance(src_x, list):
            for i, (image_i, t_i) in enumerate(zip(src_x, src_t)):
                image_i = image_i.to(device=device, dtype=self.dtype)
                t_i = t_i.to(device=device, dtype=self.dtype)
                t_i_emb = self.time_embed(t_i)  # [n_images, n_embd]
                if image_i.shape[-1] != n_embd:
                    src_emb, _, _ = self.patch_embed(image_i, t_i_emb)  # [n_images, tokens, n_embd]
                else:
                    src_emb = image_i
                scatter_idx = index[i:i+1].masked_select(src_image_mask[i:i+1]).reshape(1, -1)
                hidden_states[i:i+1].scatter_(
                    dim=1,
                    index=scatter_idx.unsqueeze(-1).repeat(1, 1, n_embd),
                    src=src_emb.reshape(1, -1, n_embd),
                )
        else:
            src_x = src_x.to(device=device, dtype=self.dtype)
            src_t = src_t.to(device=device, dtype=self.dtype)
            t_emb = self.time_embed(src_t)
            src_emb, _, _ = self.patch_embed(src_x, t_emb)
            scatter_idx = index.masked_select(src_image_mask).reshape(bsz, -1)
            hidden_states.scatter_(
                dim=1,
                index=scatter_idx.unsqueeze(-1).repeat(1, 1, n_embd),
                src=src_emb,
            )
        return hidden_states

    def _forward_vision_encoder(self, images, **image_kwargs):
        image_embeds = self.vision_model(images, **image_kwargs).last_hidden_state
        image_embeds = self.vision_aligner(image_embeds)

        return image_embeds

    def instantiate_vit_image_tokens(
            self,
            hidden_states: torch.Tensor,
            images: torch.Tensor | list[torch.Tensor],
            image_masks: torch.Tensor,
            image_kwargs: dict[str, torch.Tensor],
    ):
        """
        Encode images using vision encoder(vit), and then instantiate the image embeddings into
        the input embedding sequence.

        Args:
            hidden_states (torch.Tensor): input sequence, (bsz, seqlen, n_embd)
            images (torch.Tensor | list[torch.Tensor]): images can be a 3-D or 4-D tensor, or a list of tensors.
            image_masks (torch.Tensor): mask for the images, (bsz, seqlen)
            image_kwargs (dict[str, torch.Tensor]): additional keyword arguments for the image encoder

        Returns:
            Instantiated input sequence
        """
        bsz, seqlen, n_embd = hidden_states.shape
        index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)

        if isinstance(images, torch.Tensor):
            assert images.ndim in [3, 4, 5], f"images should be a 3-D, 4-D, or 5-D tensor, got {images.ndim}-D tensor."
            if images.ndim in [4, 5]:
                bsz, n = images.shape[:2]
                images = images.view(bsz * n, *images.shape[2:])
                image_kwargs = image_kwargs if image_kwargs is not None else {}
                for k, v in image_kwargs.items():
                    image_kwargs[k] = v.reshape(bsz * n, *v.shape[2:])
            else:
                n = 1
            image_embeds = self._forward_vision_encoder(images, **image_kwargs)
            image_seqlen = image_embeds.size(1)

            image_scatter_index = index.masked_select(image_masks.bool()).reshape(bsz, -1)
            hidden_states.scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=image_embeds.reshape(bsz, n * image_seqlen, n_embd),
            )

        elif isinstance(images, list):
            for i, (image, image_mask) in enumerate(zip(images, image_masks)):
                cur_kwargs = {k: v[i] for k, v in image_kwargs.items()} if image_kwargs is not None else {}
                image_embed = self._forward_vision_encoder(image, **cur_kwargs)
                n, image_seqlen, n_embd = image_embed.shape
                image_embed = image_embed.reshape(n * image_seqlen, n_embd)

                image_scatter_index = index[i:i+1].masked_select(image_mask.bool()).reshape(1, -1)
                hidden_states[i:i+1].scatter_(
                    dim=1,
                    index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                    src=image_embed.reshape(1, -1, n_embd),
                )
        else:
            raise ValueError(f"und_images should be Tensor or List, but got {type(images)}")

        return hidden_states

    def instantiate_continuous_tokens(
            self,
            hidden_states: torch.Tensor,
            timesteps: Optional[BatchRaggedTensor] = None,
            timesteps_index: Optional[BatchRaggedTensor] = None,
    ):
        bsz, seqlen, n_embd = hidden_states.shape

        if isinstance(timesteps, list):
            for i, timestep in enumerate(timesteps):
                timestep_src = self.timestep_emb(timestep)  # (n, n_embd)
                hidden_states[i:i+1].scatter_(
                    dim=1,
                    index=timesteps_index[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, n_embd),
                    src=timestep_src.reshape(1, -1, n_embd),
                )
        else:
            timesteps_src = self.timestep_emb(timesteps.reshape(-1))    # (bsz * n, n_embd)
            hidden_states.scatter_(
                dim=1,
                index=timesteps_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=timesteps_src.reshape(bsz, -1, n_embd),
            )

        return hidden_states

    def instantiate_guidance_tokens(
            self,
            hidden_states: torch.Tensor,
            guidance: Optional[BatchRaggedTensor] = None,
            guidance_index: Optional[BatchRaggedTensor] = None,
    ):
        bsz, seqlen, n_embd = hidden_states.shape

        guidance_src = self.guidance_emb(guidance.reshape(-1))    # (bsz * n, n_embd)
        hidden_states.scatter_(
            dim=1,
            index=guidance_index.unsqueeze(-1).repeat(1, 1, n_embd),
            src=guidance_src.reshape(bsz, -1, n_embd),
        )

        return hidden_states


    def instantiate_timestep_r_tokens(
            self,
            hidden_states: torch.Tensor,
            timesteps_r: Optional[BatchRaggedTensor] = None,
            timesteps_r_index: Optional[BatchRaggedTensor] = None,
    ):
        bsz, seqlen, n_embd = hidden_states.shape

        if isinstance(timesteps_r, list):
            for i, timestep_r in enumerate(timesteps_r):
                timestep_r_src = self.timestep_r_emb(timestep_r)  # (n, n_embd)
                hidden_states[i:i+1].scatter_(
                    dim=1,
                    index=timesteps_r_index[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, n_embd),
                    src=timestep_r_src.reshape(1, -1, n_embd),
                )
        else:
            timesteps_r_src = self.timestep_r_emb(timesteps_r.reshape(-1))    # (bsz * n, n_embd)
            hidden_states.scatter_(
                dim=1,
                index=timesteps_r_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=timesteps_r_src.reshape(bsz, -1, n_embd),
            )

        return hidden_states

    def get_image_tokens_hw(self, images: BatchRaggedImages):
        """Get token_h, token_w for SigLIP 3D latents (B, N, C).
        N = token_h * token_w, assuming square patches.
        """
        assert isinstance(images, (torch.Tensor, list)), f"images should be BatchRaggedImages, got {type(images)}"
        if isinstance(images, torch.Tensor):
            n_tokens = images.shape[1]
            side = int(n_tokens ** 0.5)
            token_h, token_w = side, side
            if side * side != n_tokens:
                token_h, token_w = n_tokens, 1
        else:
            token_h, token_w = [], []
            for image_i in images:
                if isinstance(image_i, torch.Tensor):
                    n = image_i.shape[-2] if image_i.ndim == 3 else image_i.shape[1]
                    side = int(n ** 0.5)
                    token_h.append(side)
                    token_w.append(side)
                else:
                    token_h.append([int(img.shape[-2] ** 0.5) for img in image_i])
                    token_w.append([int(img.shape[-2] ** 0.5) for img in image_i])
        return token_h, token_w

    def ragged_final_layer(self, hidden_states, image_mask, timesteps, token_h, token_w, first_step=None):
        n_embd = hidden_states.size(-1)
        num_tokens = token_h * token_w if isinstance(token_h, int) else None

        if isinstance(timesteps, torch.Tensor):
            t_emb = self.time_embed(timesteps)
            if first_step is False:
                image_output = hidden_states[:, self.num_special_tokens:, :]
            else:
                image_output = hidden_states.masked_select(
                    image_mask.unsqueeze(-1).bool()).reshape(-1, num_tokens, n_embd)
            pred = self.final_layer(image_output, t_emb, token_h, token_w)
        else:
            sections = image_mask.sum(1).tolist()
            image_output = hidden_states.masked_select(
                image_mask.unsqueeze(-1).bool()).reshape(-1, n_embd).split(sections)
            pred = []
            for image_output_i, t_i, token_h_i, token_w_i in zip(image_output, timesteps, token_h, token_w):
                t_emb_i = self.time_embed(t_i)
                if isinstance(token_h_i, int):
                    image_output_i = image_output_i.reshape(-1, token_h_i * token_w_i, n_embd)
                    pred_i = self.final_layer(image_output_i, t_emb_i, token_h_i, token_w_i)
                    pred.append(pred_i)
                else:
                    subsections = [token_h_ij * token_w_ij for token_h_ij, token_w_ij in zip(token_h_i, token_w_i)]
                    image_output_i = image_output_i.split(subsections)
                    pred_i = []
                    for j, image_output_ij in enumerate(image_output_i):
                        pred_ij = self.final_layer(image_output_ij[None], t_emb_i[j:j+1], token_h_i[j], token_w_i[j])
                        pred_i.append(pred_ij)
                    pred.append(pred_i)
        return pred

    @staticmethod
    def _check_inputs(cond, target, check_list):
        if cond:
            for name, item in check_list:
                assert item is not None, f"`{name}` should be provided when `{target}`."

    @add_start_docstrings_to_model_forward(UniCom_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,  # bsz x seqlen
            attention_mask: Optional[torch.Tensor] = None,  # bsz x 1 x seqlen x seqlen
            rope_image_info: Optional[list[list[tuple[slice, tuple[int, int]]]]] = None,
            return_dict: bool = True,
            # for gen images
            images: Optional[BatchRaggedImages] = None,  # bsz x c x h x w, or bsz x (n_i x (c x h_ij x w_ij))
            image_mask: Optional[torch.Tensor] = None,  # bsz x seqlen
            timesteps: Optional[BatchRaggedTensor] = None,  # bsz, or bsz x (n_i)
            timesteps_index: Optional[BatchRaggedTensor] = None,  # bsz x k, or bsz x (k_i)
            timesteps_r: Optional[BatchRaggedTensor] = None,  # bsz, or bsz x (n_i)
            timesteps_r_index: Optional[BatchRaggedTensor] = None,  # bsz x k, or bsz x (k_i)
            guidance: Optional[BatchRaggedTensor] = None,  # bsz, or bsz x (n_i)
            guidance_index: Optional[BatchRaggedTensor] = None,  # bsz x k, or bsz x (k_i)
            # for source images (editing: compressed SigLIP latents)
            src_x: Optional[torch.Tensor] = None,
            src_t: Optional[torch.Tensor] = None,
            src_image_mask: Optional[torch.Tensor] = None,
            src_timestep_scatter_index: Optional[torch.Tensor] = None,
            # only for inference
            position_ids: Optional[torch.Tensor] = None,  # bsz x seq_len-1, used for KVCache
            past_key_values: Optional[UniComStaticCache] = None,
            mode: Optional[str] = None,
            first_step: Optional[bool] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_dic = None,
            gen_timestep_scatter_index: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalMMOutputWithPast]:


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Sanity Check of Inputs
        self._check_inputs(mode == "gen_image", "in `gen_image` mode", [
            ("images", images), ("timesteps", timesteps),
        ])
        self._check_inputs(mode == "gen_image" and first_step, "in `gen_image` mode at the first step", [
            ("image_mask", image_mask), ("timesteps_index", timesteps_index),
        ])
        if input_ids is None and images is None:
            raise ValueError("Either input_ids or images should be provided.")
        if input_ids is not None:
            device = input_ids.device
        else:
            device = get_device(images)
        if self.training:
            seqlen = input_ids.size(1)
        else:
            # For inference, we always set seqlen to maximum length to simplify the rope cache handling
            seqlen = self.config.max_position_embeddings
        assert self.config.max_position_embeddings >= seqlen, (
            f"Cannot forward sequence of length {seqlen}, "
            f"max position embeddings is only {self.config.max_position_embeddings}, "
            f"try set --max-position-embeddings to a larger value."
        )

        # Calculate multimodal 2d rope
        cos, sin = self.cached_rope(
            seqlen, device, rope_image_info=rope_image_info, position_ids=position_ids,
        )
        # === Map token ids to embeddings ===
        if input_ids is not None:
            hidden_states = self.model.wte(input_ids)  # (bsz, seqlen, n_embd)
        else:
            hidden_states = None  # only for non-first step inference of the image generation

        # === Input layers ===
        if images is not None:
            if self.config.cfg_distilled and input_ids is None:
                hidden_states = self.instantiate_image_tokens(hidden_states, timesteps, images, image_mask, guidance, timesteps_r)
            else:
                hidden_states = self.instantiate_image_tokens(hidden_states, timesteps, images, image_mask)

        if src_x is not None:
            hidden_states = self.instantiate_src_image_tokens(hidden_states, src_x, src_t, src_image_mask)
        if src_timestep_scatter_index is not None and src_t is not None:
            hidden_states = self.instantiate_continuous_tokens(hidden_states, src_t, src_timestep_scatter_index)

        if timesteps_index is not None:
            hidden_states = self.instantiate_continuous_tokens(hidden_states, timesteps, timesteps_index)

        # guidance token
        if guidance_index is not None:
            hidden_states = self.instantiate_guidance_tokens(hidden_states, guidance, guidance_index)

        # timestep r token
        if timesteps_r_index is not None:
            hidden_states = self.instantiate_timestep_r_tokens(hidden_states, timesteps_r, timesteps_r_index)

        if mode == "gen_text":
            first_step = True
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if not self.use_taylor_cache:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=hidden_states,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                custom_pos_emb=(cos, sin),
                mode=mode,
                first_step=first_step,
                post_token_len = self.post_token_len,
                num_image_tokens = self.num_image_tokens,
                gen_timestep_scatter_index = gen_timestep_scatter_index,
                num_special_tokens = self.num_special_tokens,
            )
            hidden_states = outputs[0]
        else:
            if not hasattr(self.model, "taylor_cache"):
                self.model.taylor_cache = CacheWithFreqsContainer(cache_dic['max_order'])
            if not hasattr(self.model, "counter"):
                self.model.counter = 0

            full_computation = (cache_dic['current_step'] == 0) \
                or (self.model.counter == cache_dic['cache_interval'] -1) \
                or (cache_dic['enable_first_enhance'] and cache_dic['current_step'] < cache_dic['first_enhance_steps']) \
                or (cache_dic['enable_tailing_enhance'] and cache_dic['current_step'] >= cache_dic['num_steps'] - cache_dic['tailing_enhance_steps'])
            if not hasattr(self.model, "last_full_computation_step"):
                self.model.last_full_computation_step = 0
            if full_computation:
                self.model.counter = 0
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=hidden_states,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    custom_pos_emb=(cos, sin),
                    mode=mode,
                    first_step=first_step,
                    post_token_len = self.post_token_len,
                    num_image_tokens = self.num_image_tokens,
                    gen_timestep_scatter_index = gen_timestep_scatter_index,
                    num_special_tokens = self.num_special_tokens,
                )
                hidden_states = outputs[0]

                if cache_dic['enable_first_enhance'] and (cache_dic['current_step'] < (cache_dic['first_enhance_steps']-1)):
                    pass
                else:
                    self.model.taylor_cache.derivatives_computation(hidden_states, distance = cache_dic['current_step'] - self.model.last_full_computation_step, low_freqs_order=cache_dic['low_freqs_order'], high_freqs_order=cache_dic['high_freqs_order'])

                self.model.last_full_computation_step = cache_dic['current_step']
                self.model.taylor_cache.last_past_key_values = outputs.past_key_values
            else:
                self.model.counter += 1
                hidden_states = self.model.taylor_cache.taylor_formula(distance = self.model.counter)
                outputs = BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=self.model.taylor_cache.last_past_key_values,
                    hidden_states=None,
                    attentions=None,
                )
            if cache_dic['current_step'] == cache_dic['num_steps'] - 1:
                self.model.taylor_cache.clear_derivatives()


        # === Output layers ===
        # Apply ln_f to all tokens first (matching original transfusion behavior)
        hidden_states = self.model.ln_f(hidden_states)

        # -- image tokens
        if images is not None:
            token_h, token_w = self.get_image_tokens_hw(images)
            hidden_states = hidden_states.to(device=get_device(images))
            diff_pred = self.ragged_final_layer(
                hidden_states, image_mask, timesteps, token_h, token_w, first_step)
        else:
            diff_pred = None
        # -- text tokens
        if input_ids is None or mode == "gen_image":
            logits = None
        else:
            logits = self.lm_head(hidden_states)  # (bsz, seqlen, vocab_size)
        # -- for inference
        if not return_dict:
            return (logits.float(),) + outputs[1:] + (diff_pred,)
        return CausalMMOutputWithPast(
            logits=logits.float() if logits is not None else None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            diffusion_prediction=diff_pred,
        )

    @staticmethod
    def check_inputs(prompt=None, image=None, message_list=None):
        if prompt is None and message_list is None:
            raise ValueError("Either `prompt` or `message_list` should be provided.")
        if prompt is not None and message_list is not None:
            raise ValueError("`prompt` and `message_list` cannot be provided at the same time.")
        if message_list is not None:
            if not isinstance(message_list, list):
                raise ValueError(f"`message_list` should be a list of messages, but got {type(message_list)}.")
            assert len(message_list) > 0, "`message_list` should be a non-empty list."
            for message in message_list:
                assert isinstance(message, list) or isinstance(message, dict), \
                    f"Each message should be a list of dicts or a dict, but got {type(message)}."
        if image is not None:
            error_msg = \
                "`image` should be a PIL Image, a string path, a base64 string, bytes, or a list of them, but got {}."
            if isinstance(image, list):
                for im in image:
                    assert isinstance(im, (Image.Image, str, bytes)), error_msg.format(type(im))
            else:
                assert isinstance(image, (Image.Image, str, bytes)), error_msg.format(type(image))

    @staticmethod
    def _validate_and_batchify_text(text, name, check_batch_size=None):
        if text is None:
            return text
        assert isinstance(text, str) or isinstance(text, list), \
            f"Input `{name}` should be a string or a list of strings, but got {type(text)}."
        if isinstance(text, str):
            text = [text]
        if check_batch_size is not None:
            if len(text) == 1 and check_batch_size > 1:
                text = text * check_batch_size
            assert len(text) == check_batch_size, \
                f"Input `{name}` should have the same batch size as other inputs({check_batch_size}), got {len(text)}."
        return text

    @staticmethod
    def _validate_and_batchify_image(image, name, check_batch_size=None):
        if image is None:
            return image
        assert isinstance(image, (InputImage, list)), \
            f"Input `{name}` should be a image or a list of images, but got {type(image)}."
        if not isinstance(image, list):
            image = [image]
        batch_image_list = [image] if not isinstance(image[0], list) else image
        for image_list in batch_image_list:
            assert all(isinstance(im, InputImage) for im in image_list), \
                (f"Each item in `{name}` should be a PIL Image, a string path, a base64 string, or bytes, "
                 f"got {[type(im) for im in image_list]}.")
        if check_batch_size is not None:
            if len(batch_image_list) == 1 and check_batch_size > 1:
                batch_image_list = batch_image_list * check_batch_size
            assert len(batch_image_list) == check_batch_size, \
                f"Input `{name}` should have the same batch size as other inputs({check_batch_size})"
        return batch_image_list

    @staticmethod
    def prepare_seed(seed, batch_size):
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [random.randint(0, 10_000_000) for _ in range(batch_size)]
        elif isinstance(seed, int):
            seeds = [seed for _ in range(batch_size)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) for i in range(batch_size)]
            else:
                raise ValueError(f"Length of seed must be equal to the batch_size({batch_size}), got {seed}.")
        else:
            raise ValueError(f"Seed must be an integer, a list of integers, or None, got {seed}.")
        return seeds

    def build_batch_rope_image_info(self, output, sections):
        # Rope 1D. No need to build rope_image_info
        if self.config.rope_type == "default":
            return None

        # Rope 2D
        assert self.config.rope_type == "2d", \
            f"Rope type {self.config.rope_type} not supported by method 'build_batch_rope_image_info'."
        rope_image_info = []
        for image_slices, sections_i in zip(output.all_image_slices, sections):
            rope_2d_image_slices = []
            rope_2d_image_shapes = []
            image_idx = 0

            for section in sections_i:
                if section['type'] in ["gen_image", "cond_src_image"]:
                    assert image_idx < len(image_slices), \
                        f"Image index {image_idx} out of range for image slices with length {len(image_slices)}."
                    rope_2d_image_slices.append(image_slices[image_idx])
                    rope_2d_image_shapes.append((section['token_height'], section['token_width']))
                    image_idx += 1

            rope_image_info.append(list(zip(rope_2d_image_slices, rope_2d_image_shapes)))

        return rope_image_info

    def _encode_cond_image(
            self,
            batch_cond_images: list[list[CondImage]],
            cfg_factor: int = 1,
            generator=None,
    ):
        """Encode conditional images to produce src_x/src_t for editing tasks."""
        if batch_cond_images is None or len(batch_cond_images[0]) == 0:
            return None, None

        return self._encode_src_images(batch_cond_images, cfg_factor)

    def _encode_src_images(
            self,
            batch_cond_images: list[list[CondImage]],
            cfg_factor: int = 1,
    ):
        """Encode conditional images through SigLIP VIT + AttentionCompressionNet to produce src_x (64-dim).

        Each CondImage has .vit_image (pixel values + vision_encoder_kwargs) for running VIT.
        """
        if not hasattr(self, 'compression') or self.compression is None:
            return None, None

        batch_src_x, batch_src_t = [], []
        for cond_images in batch_cond_images:
            src_x_list, src_t_list = [], []
            for cond_image in cond_images:
                pixel_values = cond_image.vit_image

                if not hasattr(pixel_values, 'vision_encoder_kwargs'):
                    continue

                pv = pixel_values.unsqueeze(0).to(device=self.device, dtype=self.dtype)
                vit_kwargs = {}
                if pixel_values.vision_encoder_kwargs:
                    for k, v in pixel_values.vision_encoder_kwargs.items():
                        key = "attention_mask" if k == "pixel_attention_mask" else k
                        if isinstance(v, torch.Tensor):
                            vit_kwargs[key] = v.unsqueeze(0).to(device=self.device)
                        else:
                            vit_kwargs[key] = v

                with torch.no_grad():
                    siglip_features = self.vision_model(pv, **vit_kwargs).last_hidden_state
                    latents = self.compression.compress(siglip_features)

                src_x_list.append(latents)
                src_t_list.append(torch.zeros((latents.shape[0],)))

            if src_x_list:
                batch_src_x.append(torch.cat(src_x_list, dim=0))  # [n_images, 1024, 64]
                batch_src_t.append(torch.cat(src_t_list, dim=0))  # [n_images]

        if not batch_src_x:
            return None, None

        src_x = batch_src_x  # list of [n_i, tokens, 64]
        src_t = batch_src_t  # list of [n_i]

        if cfg_factor > 1:
            src_x = src_x * cfg_factor
            src_t = src_t * cfg_factor

        return src_x, src_t

    @torch.no_grad()
    def prepare_message_list(
            self,
            message_list,
            cond_images: list[CondImage] = None,
            gen_image_info: ImageInfo = None,
    ):
        """ Convert a batch message list of OpenAI style to the internal format. """
        inner_message_list = []
        image_idx = 0
        for message in message_list:
            content = message["content"]
            if isinstance(content, str):
                inner_message_list.append(dict(role=message["role"], type="text", content=content))
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        inner_message_list.append(dict(role=message["role"], type="text", content=item['text']))
                    elif item["type"] == "image":
                        if all(key not in item for key in ["image", "url", "path", "base64"]):
                            continue
                        assert cond_images is not None and image_idx < len(cond_images), \
                            f"Image index {image_idx} out of range for cond images with length {len(cond_images)}."
                        image = cond_images[image_idx]
                        inner_message_list.append(dict(role=message["role"], type=image.section_type, content=image.i))
                        image_idx += 1
                    else:
                        raise NotImplementedError(f"Message content type {item['type']} not supported.")
            else:
                raise ValueError(f"Message content should be str or list, but got {type(content)}.")

        if gen_image_info is not None:
            inner_message_list.append(dict(role="assistant", type="gen_image", content=gen_image_info))

        return inner_message_list

    def preprocess_inputs(
            self,
            prompt: str | list[str] = None,
            image: InputImage | list[InputImage] = None,
            cot_text=None,
            message_list=None,
            cfg_factor=1,
            bot_task='auto',
            system_prompt=None,
            max_new_tokens=None,
            mode="gen_text",
            image_size="auto",
            infer_align_image_size=False,
            device=None,
            **kwargs,
    ):
        # 1. Sanity check
        self.check_inputs(prompt, image, message_list)

        # 2. Format inputs
        batch_message_list = message_list
        batch_prompt = prompt
        batch_cot_text = cot_text
        batch_system_prompt = system_prompt

        #   -- 2.1 message_list
        batch_cond_images = kwargs.get('batch_cond_images', None)
        if batch_message_list is not None:
            if isinstance(batch_message_list[0], dict):
                batch_message_list = [batch_message_list]
            batch_size = len(batch_message_list)

            # Multiple cond images are allowed.
            if batch_cond_images is None:
                batch_cond_images = [
                    self.image_processor.build_cond_images(
                        message_list=message_list_,
                        infer_align_image_size=infer_align_image_size,
                    )
                    for message_list_ in batch_message_list
                ]
            if mode == "gen_image":
                batch_gen_image_info = [
                    self.image_processor.build_gen_image_info(image_size, add_guidance_token=self.config.cfg_distilled, add_timestep_r_token=self.config.use_meanflow) for _ in range(batch_size)
                ]
            else:
                batch_gen_image_info = [None] * batch_size
            # Convert OpenAI message list into inner message list
            batch_message_list = [
                self.prepare_message_list(message_list_, cond_images, gen_image_info)
                for message_list_, cond_images, gen_image_info in zip(
                    batch_message_list, batch_cond_images, batch_gen_image_info
                )
            ]

        #   -- 2.2 Prompt, image, cot text, system prompt
        else:
            batch_prompt = self._validate_and_batchify_text(batch_prompt, 'prompt')
            batch_size = len(batch_prompt)

            batch_cot_text = self._validate_and_batchify_text(batch_cot_text, 'cot_text', batch_size)
            batch_system_prompt = self._validate_and_batchify_text(batch_system_prompt, 'system_prompt', batch_size)

            batch_image_list = self._validate_and_batchify_image(image, 'image', batch_size)
            if batch_cond_images is None:
                batch_cond_images = [
                    self.image_processor.build_cond_images(
                        image_list=image_list,
                        infer_align_image_size=infer_align_image_size
                    )
                    for image_list in batch_image_list
                ] if batch_image_list is not None else None

            if mode == "gen_image":
                batch_gen_image_info = [
                    self.image_processor.build_gen_image_info(image_size, add_guidance_token=self.config.cfg_distilled, add_timestep_r_token=self.config.use_meanflow) for _ in range(batch_size)
                ]
            else:
                batch_gen_image_info = [None] * batch_size

        # Apply batched prompt or batched message_list to build input sequence with associated info.
        # If `drop_think` enabled, always drop <tool_call> parts in the context.
        drop_think = kwargs.get('drop_think', getattr(self.generation_config, 'drop_think', False))
        out = self._tokenizer.apply_chat_template(
            batch_prompt=batch_prompt,
            batch_message_list=batch_message_list,
            mode=mode,
            batch_gen_image_info=batch_gen_image_info,
            batch_cond_images=batch_cond_images,
            batch_system_prompt=batch_system_prompt,
            batch_cot_text=batch_cot_text,
            max_length=kwargs.get('max_length', self.generation_config.max_length),
            bot_task=bot_task,
            image_base_size=(
                None if mode == "gen_text" and bot_task == "auto" else self.image_processor.reso_group.base_size
            ),
            sequence_template=getattr(self.generation_config, 'sequence_template', 'pretrain'),
            cfg_factor=cfg_factor,
            drop_think=drop_think,
        )
        out['batch_size'] = batch_size
        out['batch_cond_images'] = batch_cond_images
        out['batch_gen_image_info'] = batch_gen_image_info

        # 8. Define stop tokens by tasks
        tkw = self._tokenizer
        if bot_task == "auto":
            stop_token_id = dict(
                auto=self._tokenizer.conversation.stop_token_ids,
            )
        else:
            if image_size == "auto":
                extra_auto_stops = [tkw.ratio_token_id(i) for i in range(33)]
            else:
                extra_auto_stops = [tkw.boi_token_id]
            stop_token_id = dict(
                auto=self._tokenizer.conversation.stop_token_ids + extra_auto_stops,
                recaption=[tkw.end_of_recaption_token_id],
                think=[tkw.end_of_think_token_id, tkw.end_of_recaption_token_id],
                img_ratio=extra_auto_stops,
            )
        out['stop_token_id'] = stop_token_id

        return out

    def prepare_model_inputs(
            self,
            prompt: str | list[str] = None,
            image: InputImage | list[InputImage] = None,
            mode="gen_text",
            system_prompt=None,
            cot_text=None,
            image_size="auto",
            message_list=None,
            device=None,
            max_new_tokens=None,
            **kwargs,
    ):
        device = default(device, self.device)

        # 1. apply chat template
        cfg_factor = {"gen_text": 1, "gen_image": 2}
        if self.config.cfg_distilled:
            cfg_factor["gen_image"] = 1

        bot_task = kwargs.pop("bot_task", "auto")

        out = kwargs.pop('tokenizer_output', None)
        if out is None:
            out = self.preprocess_inputs(
                prompt=prompt,
                image=image,
                mode=mode,
                system_prompt=system_prompt,
                cot_text=cot_text,
                image_size=image_size,
                message_list=message_list,
                cfg_factor=cfg_factor[mode],
                bot_task=bot_task,
                **kwargs,
            )
        output, sections = out['output'], out['sections']

        batch_size = out['batch_size']
        batch_cond_images = out['batch_cond_images']
        batch_gen_image_info = out['batch_gen_image_info']
        stop_token_id = out['stop_token_id']
        #if batch_gen_image_info[0] is not None:
        #    print("batch_gen_image_info image_token_length:", batch_gen_image_info[0].image_token_length)
        #   -- 2.3 seed
        seeds = self.prepare_seed(seed=kwargs.get('seed'), batch_size=batch_size)
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        # 4. Encode conditional images (produces src_x/src_t for editing)
        src_x, src_t = self._encode_cond_image(
            batch_cond_images, cfg_factor[mode], generator=generator,
        )

        # 5. Build position embeddings
        rope_image_info = self.build_batch_rope_image_info(output, sections)

        # 6. Build kv cache
        if mode == "gen_image":
            # Image generation will not extend sequence length, using token length as max_cache_len is enough.
            max_cache_len = output.tokens.shape[1]
        else:
            max_cache_len = output.tokens.shape[1] + default(max_new_tokens, self.generation_config.max_length)
        cache = UniComStaticCache(
            config=self.config,
            max_batch_size=batch_size * cfg_factor[mode],
            max_cache_len=max_cache_len,
            dtype=self.dtype,
            dynamic=mode == "gen_text",
        )

        # 7. Build position ids
        batch_position_ids = torch.arange(
            0, output.tokens.shape[1], dtype=torch.long, device=device)[None].expand(
            batch_size * cfg_factor[mode], -1)  # use expand to share indices to save memory

        # 8. Define stop tokens by tasks
        tkw = self._tokenizer
        if mode == "gen_image":
            eos_token_id = None  # don't need to define eos_token_id for image generation
        else:
            if bot_task == "auto":
                stop_token_id = dict(
                    auto=self._tokenizer.conversation.stop_token_ids,
                )
            else:
                if image_size == "auto":
                    extra_auto_stops = tkw.get_all_ratio_token_ids()
                else:
                    extra_auto_stops = [tkw.boi_token_id]
                stop_token_id = dict(
                    auto=self._tokenizer.conversation.stop_token_ids + extra_auto_stops,
                    recaption=[tkw.end_of_recaption_token_id],
                    think=[tkw.end_of_think_token_id, tkw.end_of_recaption_token_id],
                    img_ratio=extra_auto_stops,
                )
            eos_token_id = stop_token_id[bot_task]

        # 9. Build model input kwargs
        model_input_kwargs = dict(
            input_ids=output.tokens.to(device),
            position_ids=batch_position_ids,
            past_key_values=cache,
            mode=mode,
            rope_image_info=rope_image_info,
            image_mask=to_device(output.gen_image_mask, device),
            timesteps_index=to_device(output.gen_timestep_scatter_index, device),
            guidance_index=to_device(output.guidance_scatter_index, device),
            timesteps_r_index=to_device(output.gen_timestep_r_scatter_index, device),
            # for source image editing
            src_x=to_device(src_x, device),
            src_t=to_device(src_t, device),
            src_image_mask=to_device(output.src_image_mask, device),
            src_timestep_scatter_index=to_device(output.src_timestep_scatter_index, device),
            # for inner usage
            tokenizer_output=output,
            batch_gen_image_info=batch_gen_image_info,
            generator=generator,
            batch_cond_images=batch_cond_images,
            # generation config
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
            gen_timestep_scatter_index=to_device(output.gen_timestep_scatter_index, device),
        )

        return model_input_kwargs

    def _prepare_attention_mask_for_generation(
            self,
            inputs_tensor: torch.Tensor,
            generation_config: GenerationConfig,
            model_kwargs: dict[str, Any],
    ) -> Optional[torch.Tensor]:
        # create `4d` bool attention mask (b, 1, seqlen, seqlen) using this implementation to bypass the 2d requirement
        # in the `transformers.generation_utils.GenerationMixin.generate`.
        # This implementation can handle sequences with text and image modalities, where text tokens use causal
        # attention and image tokens use full attention.
        bsz, seq_len = inputs_tensor.shape
        tokenizer_output = model_kwargs["tokenizer_output"]
        batch_full_attn_slices = [
            self.image_processor.prepare_full_attn_slices(tokenizer_output, i)
            for i in range(bsz)
        ]
        #if len(batch_full_attn_slices[0]) == 0:
        #    return None

        attention_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=inputs_tensor.device).tril(
            diagonal=0).repeat(bsz, 1, 1)
        for i in range(bsz):
            for j, image_slice in enumerate(batch_full_attn_slices[i]):
                attention_mask[i, image_slice, image_slice] = True
        attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
            tokenizer_output=None, batch_gen_image_info=None, batch_cond_images=None,
            infer_align_image_size=False, generator=None, **kwargs
    ):
        position_ids = kwargs.get("position_ids")
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            assert position_ids is not None, "position_ids must be provided in kwargs."
            if input_ids is not None and input_ids.shape[1] != position_ids.shape[1]:    # in decode steps
                input_ids = torch.gather(input_ids, dim=1, index=position_ids)
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                # "use_cache": kwargs.get("use_cache"),
                "rope_image_info": kwargs["rope_image_info"],
                "mode": kwargs["mode"],
                "images": kwargs.get("images"),
                "image_mask": kwargs.get("image_mask"),
                "timesteps": kwargs.get("timesteps"),
                "timesteps_index": kwargs.get("timesteps_index"),
                "timesteps_r": kwargs.get("timesteps_r"),
                "timesteps_r_index": kwargs.get("timesteps_r_index"),
                "guidance": kwargs.get("guidance"),
                "guidance_index": kwargs.get("guidance_index"),
                "src_x": kwargs.get("src_x"),
                "src_t": kwargs.get("src_t"),
                "src_image_mask": kwargs.get("src_image_mask"),
                "src_timestep_scatter_index": kwargs.get("src_timestep_scatter_index"),
                "cache_dic":  kwargs.get("cache_dic"),
                "gen_timestep_scatter_index": kwargs.get("gen_timestep_scatter_index"),
            }
        )

        return model_inputs

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: dict[str, Any],
            is_encoder_decoder: bool = False,
            num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        """ This function is run after each step of model forward. It updates model kwargs for next forward step.
        """
        mode = model_kwargs["mode"]

        updated_model_kwargs = {
            "mode": mode,
            "rope_image_info": model_kwargs["rope_image_info"],
        }

        # update past_key_values keeping its naming used in model code
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                updated_model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        if "tokenizer_output" in model_kwargs:
            # After prefill step
            if mode == "gen_text":
                # When enable batching, we use right padding, which requires a real_pos to index the valid
                # end position of the sequence. If tokenizer_output in model_kwargs, it means we are in the
                # prefill step of generation.
                real_pos = to_device(model_kwargs["tokenizer_output"].real_pos, self.device)
                updated_model_kwargs["position_ids"] = real_pos
            else:
                # inputs_pos
                image_mask = model_kwargs["image_mask"]
                bsz, seq_len = image_mask.shape
                index = torch.arange(seq_len, device=image_mask.device).unsqueeze(0).repeat(bsz, 1)
                position_ids = index.masked_select(image_mask.bool()).reshape(bsz, -1)
                timestep_position_ids = \
                    index[torch.arange(bsz), model_kwargs["timesteps_index"][:, -1]].unsqueeze(-1)
                pos_cat_list = [timestep_position_ids, ]
                if self.config.cfg_distilled:
                    guidance_position_ids = index[torch.arange(bsz), model_kwargs["guidance_index"][:, -1]].unsqueeze(-1)
                    pos_cat_list.append(guidance_position_ids)
                if self.config.use_meanflow:
                    timestep_r_position_ids = index[torch.arange(bsz), model_kwargs["timesteps_r_index"][:, -1]].unsqueeze(-1)
                    pos_cat_list.append(timestep_r_position_ids)
                pos_cat_list.append(position_ids)
                updated_model_kwargs["position_ids"] = torch.cat(pos_cat_list, dim=1)

                # attention mask
                mask_list = []
                for attention_mask_i, position_ids_i in zip(
                        model_kwargs["attention_mask"], updated_model_kwargs["position_ids"]):
                    mask_list.append(torch.index_select(attention_mask_i, dim=1, index=position_ids_i.reshape(-1)))
                attention_mask = torch.stack(mask_list, dim=0)
                updated_model_kwargs["attention_mask"] = attention_mask
                updated_model_kwargs["gen_timestep_scatter_index"] = model_kwargs["gen_timestep_scatter_index"]
        else:
            # After decode steps
            if mode == "gen_text":
                # Now we are in the decode steps.
                updated_model_kwargs["position_ids"] = model_kwargs["position_ids"] + 1
                # Remove attention mask to use full attention of 1 x seqlen in decode steps
            else:
                updated_model_kwargs["position_ids"] = model_kwargs["position_ids"]
                updated_model_kwargs["attention_mask"] = model_kwargs["attention_mask"]
                updated_model_kwargs["gen_timestep_scatter_index"] = model_kwargs["gen_timestep_scatter_index"]
        return updated_model_kwargs

    class _StageTransitionLogitsProcessor(LogitsProcessor):
        def __init__(self, stage_transitions: list[tuple[int, list[int]]], batch_size: int):
            self.transition_map = {stop_id: list(append_ids) for stop_id, append_ids in stage_transitions}
            self.pending_tokens = [[] for _ in range(batch_size)]
            self.completed = [set() for _ in range(batch_size)]

        def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
            batch_size = input_ids.shape[0]
            last_tokens = input_ids[:, -1]
            device = scores.device
            min_score = torch.finfo(scores.dtype).min

            for i in range(batch_size):
                last_token = last_tokens[i].item()

                # Consume pending tokens if the last token matches the head.
                if self.pending_tokens[i] and last_token == self.pending_tokens[i][0]:
                    self.pending_tokens[i].pop(0)

                # If pending tokens remain, force the next token.
                if self.pending_tokens[i]:
                    scores[i].fill_(min_score)
                    scores[i, self.pending_tokens[i][0]] = 0
                    continue

                # Trigger stage transition if needed.
                if last_token in self.transition_map and last_token not in self.completed[i]:
                    self.completed[i].add(last_token)
                    next_tokens = self.transition_map[last_token]
                    if next_tokens:
                        self.pending_tokens[i] = list(next_tokens)
                        scores[i].fill_(min_score)
                        scores[i, self.pending_tokens[i][0]] = 0

                scores[i] = scores[i].to(device)

            return scores

    class _ConditionalSliceVocabLogitsProcessor(LogitsProcessor):
        def __init__(
            self,
            trigger_token_ids: list[int],
            vocab_start: int,
            vocab_end: int,
            other_slices: Optional[list[tuple[int, int]]] = None,
            force_greedy: bool = False,
        ):
            self.trigger_token_ids = set(trigger_token_ids)
            self.vocab_start = vocab_start
            self.vocab_end = vocab_end
            self.other_slices = other_slices or []
            self.force_greedy = force_greedy

        def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
            last_tokens = input_ids[:, -1]
            min_score = torch.finfo(scores.dtype).min
            for i in range(scores.size(0)):
                if last_tokens[i].item() not in self.trigger_token_ids:
                    continue
                original_scores = scores[i].clone()
                scores[i].fill_(min_score)
                scores[i, self.vocab_start:self.vocab_end] = original_scores[self.vocab_start:self.vocab_end]
                for start, end in self.other_slices:
                    scores[i, start:end] = original_scores[start:end]
                if self.force_greedy:
                    max_token_id = scores[i].argmax().item()
                    scores[i].fill_(min_score)
                    scores[i, max_token_id] = 0
            return scores

    def _get_ratio_index_from_token(self, ratio_token_id: int, tokenizer) -> int:
        if hasattr(tokenizer, "get_all_ratio_token_ids"):
            ratio_token_ids = tokenizer.get_all_ratio_token_ids()
            try:
                ratio_index = ratio_token_ids.index(ratio_token_id)
            except ValueError as exc:
                raise ValueError(f"Unknown ratio token id {ratio_token_id}") from exc
        else:
            ratio_index = ratio_token_id - tokenizer.ratio_token_id(0)
        if ratio_index < 0 or ratio_index >= len(self.image_processor.reso_group):
            raise ValueError(f"ratio_index {ratio_index} out of range for reso_group")
        return ratio_index

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            use_model_defaults: Optional[bool] = None,
            generator: Optional[list[torch.Generator]] = None,
            decode_text: bool = False,
            verbose: int = 0,
            stage_transitions: Optional[list[tuple[int, list[int]]]] = None,
            final_stop_tokens: Optional[list[int]] = None,
            **kwargs,
    ):
        gen_config = default(generation_config, self.generation_config)
        mode = kwargs.get("mode", "gen_text")
        output = kwargs["tokenizer_output"]
        indices = torch.where(output.tokens[0] == self._tokenizer.encode("<img>")[0])[0]
        if indices.shape[0] > 0:
            last_idx = indices[-1]
            self.post_token_len = int(output.tokens[0].shape[0] - 1 - last_idx)
        else:
            self.post_token_len = None
        # Log info
        if verbose >= 1:
            context = self._tokenizer.decode(output.tokens[0], skip_special_tokens=False)
            # Replace <img><img>...<img> with [<img>]{number}
            img_token = self._tokenizer.get_img_token()
            context = re.sub(f"({img_token})+", lambda m: f"[{img_token}]{{{len(m.group(0)) // 5}}}", context)
            info_list = [
                ("token shape", output.tokens.shape),
                ("context[0]", context),
            ]
            if mode == "gen_image":
                if generator is not None:
                    info_list.extend([
                        ("seed", [g.initial_seed() for g in generator]),
                    ])
                info_list.extend([
                    ("image_size",
                     [f"{info.image_height}x{info.image_width}" for info in kwargs["batch_gen_image_info"]]),
                    ("infer_steps", gen_config.diff_infer_steps),
                    ("guidance_scale", gen_config.diff_guidance_scale),
                    ("flow_shift", gen_config.flow_shift),
                ])
            else:
                info_list.extend([
                    ("do_sample", kwargs.get("do_sample", gen_config.do_sample)),
                    ("max_new_tokens", kwargs.get("max_new_tokens", gen_config.max_new_tokens)),
                    ("top_k", kwargs.get("top_k", gen_config.top_k)),
                    ("top_p", kwargs.get("top_p", gen_config.top_p)),
                    ("temperature", kwargs.get("temperature", gen_config.temperature)),
                    ("repetition_penalty", kwargs.get("repetition_penalty", gen_config.repetition_penalty)),
                ])
            max_key_len = max(len(k) for k, _ in info_list)
            info_str = "=" * 50 + \
                       "\nModel input info:\n" + \
                       "\n".join([f"    {k.rjust(max_key_len)}: {v}" for k, v in info_list]) + \
                       "\n--------------------------------------------------"
            print(info_str, flush=True)
            start_time = time.time()

        if mode == "gen_text":
            if verbose >= 2 and streamer is None:
                streamer = TextStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=False)   # noqa

            with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
                if stage_transitions is not None:
                    if final_stop_tokens is None:
                        raise ValueError("`final_stop_tokens` must be provided when `stage_transitions` is set.")
                    if logits_processor is None:
                        logits_processor = LogitsProcessorList()
                    elif not isinstance(logits_processor, LogitsProcessorList):
                        logits_processor = LogitsProcessorList(logits_processor)
                    input_ids = kwargs.get("input_ids")
                    if input_ids is None:
                        raise ValueError("`input_ids` must be provided for multi-stage generation.")
                    logits_processor.append(
                        self._StageTransitionLogitsProcessor(stage_transitions, input_ids.shape[0])
                    )
                    kwargs["eos_token_id"] = final_stop_tokens

                samples = super().generate(
                    inputs=inputs,
                    generation_config=gen_config,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    synced_gpus=synced_gpus,
                    assistant_model=assistant_model,
                    streamer=streamer,
                    negative_prompt_ids=negative_prompt_ids,
                    negative_prompt_attention_mask=negative_prompt_attention_mask,
                    use_model_defaults=use_model_defaults,
                    **kwargs,
                )
                if decode_text:
                    samples = self.decode_text(samples, input_length=kwargs["input_ids"].shape[1])

        elif mode == "gen_image":
            batch_gen_image_info: list[ImageInfo] = kwargs.get("batch_gen_image_info")
            if batch_gen_image_info is None:
                raise ValueError("`batch_gen_image_info` should be provided when `mode` is `gen_image`.")
            self.num_image_tokens = (batch_gen_image_info[0].image_token_length)
            #                       + (1 if batch_gen_image_info[0].add_timestep_token else 0)
            #                       + (1 if batch_gen_image_info[0].add_guidance_token else 0) )
            self.num_special_tokens = ((1 if batch_gen_image_info[0].add_timestep_token else 0) +
                                       (1 if batch_gen_image_info[0].add_guidance_token else 0) +
                                       (1 if batch_gen_image_info[0].add_timestep_r_token else 0) )
            results = self.pipeline(
                batch_size=len(batch_gen_image_info),
                image_size=[batch_gen_image_info[0].image_height, batch_gen_image_info[0].image_width],
                num_inference_steps=gen_config.diff_infer_steps,
                guidance_scale=gen_config.diff_guidance_scale,
                generator=generator,
                meanflow=self.config.use_meanflow,
                model_kwargs=kwargs,
                cfg_distilled = self.config.cfg_distilled,
            )
            samples = results[0]

        else:
            raise ValueError(f"Unknown mode {mode}, only `gen_text` and `gen_image` are supported.")

        if verbose >= 1:
            end_time = time.time()
            print(f"Generation completed in {end_time - start_time:.2f} seconds.", flush=True)  # noqa

        return samples

    def decode_text(self, output: torch.Tensor, input_length: int = None):
        if output.ndim == 2:
            return [self.decode_text(output_i, input_length) for output_i in output]
        elif output.ndim == 1:
            if input_length is not None:
                output = output[input_length:]
            text = self._tokenizer.decode(output)
            return text
        else:
            raise ValueError(f"output should be 1D or 2D tensor, but got {output.ndim}D tensor.")

    @torch.no_grad()
    def generate_image(
            self,
            prompt=None,
            image=None,
            message_list=None,
            seed=None,
            image_size="auto",
            use_system_prompt=None,
            system_prompt=None,
            bot_task=None,
            infer_align_image_size=False,
            diff_infer_steps=None,
            diff_guidance_scale=None,
            flow_shift=None,
            verbose=None,
            use_taylor_cache=False,
            taylor_cache_interval=None,
            taylor_cache_order=None,
            taylor_cache_enable_first_enhance=None,
            taylor_cache_first_enhance_steps=None,
            taylor_cache_enable_tailing_enhance=None,
            taylor_cache_tailing_enhance_steps=None,
            taylor_cache_low_freqs_order=None,
            taylor_cache_high_freqs_order=None,
            **kwargs,
    ):
        max_new_tokens = kwargs.pop("max_new_tokens", 2048)
        cot_text = kwargs.pop("cot_text", None)

        if diff_infer_steps is not None:
            self.generation_config.diff_infer_steps = diff_infer_steps
        if diff_guidance_scale is not None:
            self.generation_config.diff_guidance_scale = diff_guidance_scale
        if flow_shift is not None:
            self.generation_config.flow_shift = flow_shift
        if verbose is not None:
            kwargs["verbose"] = verbose

        use_system_prompt = default(use_system_prompt, self.generation_config.use_system_prompt)
        bot_task = default(bot_task, self.generation_config.bot_task)
        self.generation_config.use_system_prompt = use_system_prompt
        self.generation_config.bot_task = bot_task
        system_prompt = get_system_prompt(use_system_prompt, bot_task, system_prompt)
        system_prompt = system_prompt.strip() if system_prompt is not None else ""

        self.taylor_cache_interval = taylor_cache_interval
        self.taylor_cache_order = taylor_cache_order
        self.taylor_cache_enable_first_enhance = taylor_cache_enable_first_enhance
        self.taylor_cache_first_enhance_steps = taylor_cache_first_enhance_steps
        self.taylor_cache_enable_tailing_enhance = taylor_cache_enable_tailing_enhance
        self.taylor_cache_tailing_enhance_steps = taylor_cache_tailing_enhance_steps
        self.taylor_cache_low_freqs_order = taylor_cache_low_freqs_order
        self.taylor_cache_high_freqs_order = taylor_cache_high_freqs_order
        self.use_taylor_cache = False

        batch_cond_images_cache = None
        tkw = self._tokenizer
        need_ratio = image_size == "auto" or bot_task == "img_ratio"
        if bot_task in ["think", "recaption", "think_recaption"]:
            first_bot_task = bot_task.split("_")[0]
            stage_transitions = []

            if first_bot_task == "think" and "recaption" in bot_task:
                stage_transitions.append(
                    (tkw.end_of_think_token_id, [tkw.convert_tokens_to_ids(tkw.recaption_token)])
                )

            if need_ratio:
                answer_prefix_tokens = []
                if getattr(self.generation_config, "sequence_template", "pretrain") == "instruct":
                    answer_prefix_tokens = [tkw.convert_tokens_to_ids(tkw.answer_token)]
                image_base_size = self.image_processor.reso_group.base_size
                if "recaption" in bot_task:
                    transition_id = tkw.end_of_recaption_token_id
                else:
                    transition_id = tkw.end_of_think_token_id
                stage_transitions.append(
                    (transition_id, answer_prefix_tokens + [tkw.boi_token_id, tkw.size_token_id(image_base_size)])
                )
                final_stop_tokens = list(range(tkw.start_ratio_token_id, tkw.end_ratio_token_id + 1))
                for start, end in getattr(tkw, "ratio_token_other_slices", []):
                    final_stop_tokens.extend(range(start, end))
            else:
                if "recaption" in bot_task:
                    final_stop_tokens = [tkw.end_of_recaption_token_id]
                else:
                    final_stop_tokens = [tkw.end_of_think_token_id, tkw.end_of_recaption_token_id]

            model_inputs = self.prepare_model_inputs(
                prompt=prompt, image=image, message_list=message_list, system_prompt=system_prompt,
                seed=seed, max_new_tokens=max_new_tokens, mode="gen_text", bot_task=first_bot_task,
                batch_cond_images=batch_cond_images_cache, infer_align_image_size=infer_align_image_size,
            )
            batch_cond_images_cache = model_inputs['batch_cond_images']
            logits_processor = None
            if need_ratio:
                image_base_size = self.image_processor.reso_group.base_size
                logits_processor = LogitsProcessorList([
                    self._ConditionalSliceVocabLogitsProcessor(
                        trigger_token_ids=[tkw.size_token_id(image_base_size)],
                        vocab_start=tkw.start_ratio_token_id,
                        vocab_end=tkw.end_ratio_token_id + 1,
                        other_slices=getattr(tkw, "ratio_token_other_slices", []),
                        force_greedy=True,
                    )
                ])

            input_length = model_inputs["input_ids"].shape[1]
            if stage_transitions:
                outputs = self.generate(
                    **model_inputs,
                    decode_text=False,
                    stage_transitions=stage_transitions,
                    final_stop_tokens=final_stop_tokens,
                    logits_processor=logits_processor,
                    **kwargs,
                )
            else:
                outputs = self.generate(**model_inputs, decode_text=False, logits_processor=logits_processor, **kwargs)

            generated_tokens = outputs[:, input_length:]
            if "recaption" in bot_task:
                end_token_id = tkw.end_of_recaption_token_id
            else:
                end_token_id = tkw.end_of_think_token_id
            end_positions = (generated_tokens[0] == end_token_id).nonzero(as_tuple=False)
            if end_positions.numel() > 0:
                end_pos = end_positions[0].item()
                cot_tokens = generated_tokens[0, :end_pos + 1]
            else:
                cot_tokens = generated_tokens[0]
            cot_text_gen = self._tokenizer.decode(cot_tokens)

            if first_bot_task == "think":
                cot_text = [tkw.think_token + cot_text_gen]
            else:
                cot_text = [tkw.recaption_token + cot_text_gen]

            if self.generation_config.drop_think and tkw.think_token in cot_text[0]:
                if tkw.recaption_token in cot_text[0]:
                    recaption_part = cot_text[0].split(tkw.recaption_token)[1]
                    if tkw.end_of_recaption_token in recaption_part:
                        recaption_part = recaption_part.split(tkw.end_of_recaption_token)[0]
                    cot_text = [tkw.recaption_token + recaption_part + tkw.end_of_recaption_token]

                    if system_prompt:
                        system_prompt = get_system_prompt("en_recaption", bot_task)

            if need_ratio:
                ratio_token_id = outputs[0, -1].item()  # get the original ratio index from the generated tokens
                ratio_index = self._get_ratio_index_from_token(ratio_token_id, tkw)
                reso = self.image_processor.reso_group[ratio_index]
                image_size = reso.height, reso.width

        elif need_ratio:
            self.image_processor.build_img_ratio_slice_logits_proc(self.tokenizer)
            model_inputs = self.prepare_model_inputs(
                prompt=prompt, image=image, cot_text=cot_text, message_list=message_list, max_new_tokens=1,
                system_prompt=system_prompt, seed=seed, mode="gen_text", bot_task="img_ratio",
                batch_cond_images=batch_cond_images_cache, infer_align_image_size=infer_align_image_size,
            )
            batch_cond_images_cache = model_inputs['batch_cond_images']
            outputs = self.generate(**model_inputs, do_sample=False, logits_processor=self.image_processor.img_ratio_slice_logits_processor, **kwargs)
            ratio_index = outputs[0, -1].item()
            reso = self.image_processor.reso_group[ratio_index]
            image_size = reso.height, reso.width

        # Generate image
        self.use_taylor_cache = use_taylor_cache
        model_inputs = self.prepare_model_inputs(
            prompt=prompt, image=image, cot_text=cot_text, message_list=message_list, system_prompt=system_prompt,
            seed=seed, image_size=image_size, mode="gen_image", batch_cond_images=batch_cond_images_cache,
            infer_align_image_size=infer_align_image_size,
        )
        batch_cond_images_cache = model_inputs['batch_cond_images']
        outputs = self.generate(**model_inputs, **kwargs)
        resolved_image_size = (image_size[0], image_size[1]) if isinstance(image_size, (tuple, list)) else image_size
        return cot_text, outputs, resolved_image_size


__all__ = [
    "UniComForCausalMM",
    "UniComModel",
    "UniComPreTrainedModel",
    "TimestepEmbedder",
    "LinearPatchEmbed",
    "LinearFinalLayer",
    "CachedRoPE",
    "apply_rotary_pos_emb",
    "build_batch_2d_rope",
]
