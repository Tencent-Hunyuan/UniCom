from pathlib import Path
import logging

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VISION_ENCODER_BASE = PROJECT_ROOT / "model_zoo"
VISION_ENCODER_META_INFO = {
    "siglip2-so400m-patch16-naflex": {
        "path": str(VISION_ENCODER_BASE / "siglip2-so400m-patch16-naflex"),
        "downsample_factor": [16, 16],
    },
}

PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def load_vision_model(
    vision_model_type,
    vision_model_precision=None,
    device=None,
    logger=None,
    require_grad=False,
    eval_mode=True,
):
    if vision_model_type != "siglip2-so400m-patch16-naflex":
        raise NotImplementedError(
            f"Only siglip2-so400m-patch16-naflex is supported in this decoder runtime, got {vision_model_type}"
        )

    if logger is None:
        logger = logging.getLogger(__name__)

    vision_model_path = Path(VISION_ENCODER_META_INFO[vision_model_type]["path"])
    from .siglip2.model import Siglip2VisionTransformer

    vision_model = Siglip2VisionTransformer.from_pretrained(vision_model_path)
    if vision_model_precision is not None:
        logger.warning(f"You are transforming the Vision Encoder to {vision_model_precision}.")
        vision_model = vision_model.to(dtype=PRECISION_TO_TYPE[vision_model_precision])
    if device is not None:
        vision_model = vision_model.to(device=device)
    if not require_grad:
        vision_model.requires_grad_(False)
    if eval_mode:
        vision_model.eval()
    return vision_model
