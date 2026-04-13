import importlib.metadata
import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func
    from flash_attn.bert_padding import index_first_axis
except ImportError:
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func = None, None, None
    index_first_axis = None
from packaging import version
from transformers.utils.import_utils import _is_package_available
