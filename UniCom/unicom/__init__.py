# UniCom: Unified Multimodal Model with SigLIP2 compressed features
"""
UniCom Package

A unified multimodal model using SigLIP2 compressed features as the image
generation latent space (no VAE). Supports:
- Text-to-Text: Text generation from text prompts
- Text-to-Image: Feature generation from text prompts (outputs .pt/.csv)
- Text & Image to Text: Text generation from text and image inputs
- Text & Image to Image: Feature generation from text and image inputs

"""

from typing import TYPE_CHECKING

from utils import _LazyModule
from utils.import_utils import define_import_structure

if TYPE_CHECKING:
    from .configuration_unicom import *
    from .modeling_unicom import *
    from .compression import *
    from .image_processor import *
    from .siglip2 import *
    from .tokenization_unicom import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
