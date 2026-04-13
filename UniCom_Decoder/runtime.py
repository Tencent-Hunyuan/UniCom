import argparse
import csv
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
import yaml
from diffusers.utils.torch_utils import randn_tensor
from loguru import logger
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from .bootstrap import bootstrap
from .decoder.utils import ResolutionGroup


PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_state_dict(state, load_key: str):
    if isinstance(state, dict) and load_key in state:
        state_dict = state[load_key]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state
    if isinstance(state_dict, dict):
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not str(key).endswith("._extra_state")
        }
    return state_dict


def _infer_arch_args_from_ckpt(ckpt_file: Path, load_key: str):
    try:
        state = _torch_load(ckpt_file, map_location="cpu")
        state_dict = _extract_state_dict(state, load_key)
    except Exception:
        return {}

    if not isinstance(state_dict, dict):
        return {}

    inferred = {}

    final_proj = state_dict.get("txt_in.final_proj.weight")
    if getattr(final_proj, "ndim", 0) == 2:
        inferred["text_states_dim"] = int(final_proj.shape[0])

    vector_weight = state_dict.get("vector_in.in_layer.weight")
    if getattr(vector_weight, "ndim", 0) == 2:
        inferred["text_pool_type"] = "clip"
        inferred["text_states_dim_2"] = int(vector_weight.shape[1])
    elif not any(str(key).startswith("vector_in.") for key in state_dict.keys()):
        inferred["text_pool_type"] = None

    return inferred


def _load_saved_args(args_path: Optional[Path]):
    if args_path is None or not args_path.exists():
        return {}
    with args_path.open() as f:
        return json.load(f)


def _needs_arch_inference(config: dict) -> bool:
    required_keys = ("text_states_dim", "text_pool_type", "text_states_dim_2")
    return any(key not in config for key in required_keys)


def _load_yaml_config(config_path: Optional[Path]) -> dict:
    if config_path is None or not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    base_value = data.pop("base", None)
    if base_value is None:
        return data

    base_path = Path(base_value)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    merged = _load_yaml_config(base_path.resolve())
    merged.update(data)
    return merged


def _is_primary_shard(args_like) -> bool:
    shard_index = getattr(args_like, "shard_index", None)
    if shard_index is None and isinstance(args_like, dict):
        shard_index = args_like.get("shard_index")
    try:
        return int(shard_index or 0) == 0
    except (TypeError, ValueError):
        return True


def _rank0_print(args_like, message: str):
    if _is_primary_shard(args_like):
        print(message, flush=True)


def _rank0_info(args_like, message: str):
    if _is_primary_shard(args_like):
        logger.info(message)


def _resolve_path(value, project_root: Path) -> Optional[str]:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def _normalize_device(value):
    if isinstance(value, list):
        return [_normalize_device(item) for item in value]
    if isinstance(value, int):
        return f"cuda:{value}"
    if isinstance(value, str):
        text = value.strip()
        if "," in text:
            items = [item.strip() for item in text.split(",") if item.strip()]
            return [_normalize_device(item) for item in items]
        if text.isdigit():
            return f"cuda:{text}"
        return text
    raise ValueError(f"Unsupported device value: {value!r}")


def _resolve_ckpt(ckpt: Path) -> Path:
    if ckpt.is_file():
        return ckpt
    if ckpt.is_dir():
        if (ckpt / "checkpoints").exists() and (ckpt / "checkpoints" / "latest").exists():
            latest_name = (ckpt / "checkpoints" / "latest").read_text().strip()
            ckpt = ckpt / "checkpoints" / latest_name
        matches = sorted(ckpt.glob("*_model_states.pt"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot resolve checkpoint from {ckpt}")


def _infer_args_path(ckpt_file: Path) -> Optional[Path]:
    try:
        return ckpt_file.parents[2] / "args.json"
    except IndexError:
        return None


def build_runtime_args(cli_args) -> SimpleNamespace:
    project_root = cli_args.project_root.resolve()
    config_path = cli_args.config.resolve()

    merged = _load_yaml_config(config_path)
    _rank0_print(merged, f"[decoder] Loading config: {config_path}")

    required_fields = ("ckpt", "csv_path", "output_dir")
    missing_fields = [name for name in required_fields if merged.get(name) is None]
    if missing_fields:
        missing_display = ", ".join(missing_fields)
        raise ValueError(f"Missing required config values: {missing_display}")

    ckpt_file = _resolve_ckpt(Path(_resolve_path(merged["ckpt"], project_root)))
    args_path_value = merged.get("args_path")
    args_path = Path(_resolve_path(args_path_value, project_root)) if args_path_value else _infer_args_path(ckpt_file)

    _rank0_print(merged, f"[decoder] Resolved checkpoint: {ckpt_file}")
    if args_path and args_path.exists():
        _rank0_print(merged, f"[decoder] Loading saved args: {args_path}")
    merged.update(_load_saved_args(args_path))

    if _needs_arch_inference(merged):
        _rank0_print(merged, "[decoder] Inferring missing architecture args from checkpoint")
        merged.update(_infer_arch_args_from_ckpt(ckpt_file, merged["load_key"]))
    else:
        _rank0_print(merged, "[decoder] Architecture args already present, skip checkpoint pre-scan")

    merged["ckpt"] = str(ckpt_file)
    merged["args_path"] = str(args_path) if args_path else None
    merged["csv_path"] = Path(_resolve_path(merged["csv_path"], project_root))
    merged["output_dir"] = Path(_resolve_path(merged["output_dir"], project_root))
    merged["device"] = _normalize_device(merged.get("device") or 0)
    merged["mode"] = merged.get("mode", "eval_ar")
    merged["condition_column"] = merged.get("condition_column", "feature_path")
    merged["comparison_output"] = bool(merged.get("comparison_output", False))
    if merged.get("flow_shift") is not None:
        merged["flow_shift_eval"] = merged["flow_shift"]
    merged["siglip2_model_path"] = _resolve_path(merged.get("siglip2_model_path"), project_root)
    merged["vae_path"] = _resolve_path(merged.get("vae_path"), project_root)
    if merged.get("negative_condition_path") is not None:
        merged["negative_condition_path"] = Path(_resolve_path(merged["negative_condition_path"], project_root))

    return SimpleNamespace(**merged)


def _prepare_extra_kwargs(func, kwargs):
    accepts = set(inspect.signature(func).parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepts}


def _setup_logger(worker_name: str):
    worker_name = str(worker_name)
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>"
        + worker_name
        + "</cyan> | <level>{message}</level>",
    )


def _format_batch_header(batch_index: int, total_batches: Optional[int]) -> str:
    if total_batches is None:
        return f"Batch {batch_index}"
    return f"Batch {batch_index}/{total_batches}"


def _format_batch_visual(batch_rows: List[dict], cli_args) -> str:
    sample_row = batch_rows[0]
    w = 24
    lines = [""]

    height = sample_row.get("height")
    width = sample_row.get("width")
    if height not in (None, "") and width not in (None, ""):
        lines.append(f"{'size:':>{w}} ({int(width)}, {int(height)})")
        lines.append(f"{'frame:':>{w}} 1")

    prompts = [row.get("prompt", "") for row in batch_rows if row.get("prompt") not in (None, "")]
    if prompts:
        prompt_preview = prompts if len(prompts) <= 1 else [prompt[:80] for prompt in prompts]
        lines.append(f"{'prompt:':>{w}} {prompt_preview}")

    condition_path = sample_row.get(cli_args.condition_column)
    if condition_path not in (None, ""):
        lines.append(f"{'image_path:':>{w}} {condition_path}")

    seeds = [int(row["seed"]) for row in batch_rows if row.get("seed") not in (None, "")]
    if seeds:
        seed_display = seeds[0] if len(set(seeds)) == 1 else seeds
        lines.append(f"{'seed:':>{w}} {seed_display}")

    lines.append(f"{'neg_prompt:':>{w}} ['']")
    lines.append(f"{'infer_steps:':>{w}} {cli_args.infer_steps}")
    lines.append(f"{'num_images_per_prompt:':>{w}} 1")
    lines.append(f"{'guidance_scale:':>{w}} {cli_args.cfg_scale}")
    return "\n".join(lines)


def _iter_shard_rows(cli_args):
    with cli_args.csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader((line.replace("\0", " ") for line in f))
        for row_idx, row in enumerate(reader):
            if row_idx % cli_args.num_shards != cli_args.shard_index:
                continue
            yield row


def _count_total_batches(rows: List[dict], cli_args) -> int:
    batch_count = 0
    batch_rows = []
    for row in rows:
        if batch_rows:
            same_height = int(batch_rows[0]["height"]) == int(row["height"])
            same_width = int(batch_rows[0]["width"]) == int(row["width"])
            if not (same_height and same_width) or len(batch_rows) >= cli_args.batch_size:
                batch_count += 1
                batch_rows = []
        batch_rows.append(row)
    if batch_rows:
        batch_count += 1
    return batch_count


def _get_output_path(row: dict, cli_args) -> Path:
    if cli_args.mode == "eval_gt":
        return cli_args.output_dir / _build_eval_gt_filename(row)
    return cli_args.output_dir / _build_output_filename(row)


def _collect_pending_shard_rows(cli_args, log_skips: bool = True) -> List[dict]:
    pending_rows = []
    for row in _iter_shard_rows(cli_args):
        save_path = _get_output_path(row, cli_args)
        if getattr(cli_args, "skip_existing", False) and save_path.exists():
            if log_skips:
                _rank0_info(cli_args, f"Skip existing: {save_path}")
            continue
        pending_rows.append(row)
    return pending_rows


def _log_batch(runtime, batch_rows, cli_args, batch_index: Optional[int] = None, total_batches: Optional[int] = None):
    if not runtime.is_primary_shard or not batch_rows:
        return
    if batch_index is not None:
        logger.info(_format_batch_header(batch_index, total_batches))
    logger.info(str([str(row["index"]) for row in batch_rows]))
    logger.info(_format_batch_visual(batch_rows, cli_args))


def _retrieve_timesteps(scheduler, num_inference_steps, device, n_tokens):
    scheduler.set_timesteps(num_inference_steps, device=device, n_tokens=n_tokens)
    return scheduler.timesteps


def _prepare_latents(vae, scheduler, batch_size, num_channels_latents, height, width, frame, dtype, device, generator):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    shape = (
        batch_size,
        num_channels_latents,
        frame,
        int(height) // vae_scale_factor,
        int(width) // vae_scale_factor,
    )
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma
    return latents


def _align_hw(args, height: int, width: int) -> Tuple[int, int]:
    base_size = getattr(args, "bucket_hw_base_size", None)
    bucket_stride = getattr(args, "bucket_hw_bucket_stride", None)
    if not base_size or not bucket_stride:
        return int(height), int(width)
    reso_group = ResolutionGroup(base_size=base_size, step=bucket_stride)
    target_width, target_height = reso_group.get_target_size(int(width), int(height))
    return int(target_height), int(target_width)


def _decode_latents(vae, latents, vae_dtype, enable_autocast):
    if latents.ndim == 5 and latents.shape[2] == 1:
        latents = latents.squeeze(2)

    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
        latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    else:
        latents = latents / vae.config.scaling_factor

    with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=enable_autocast):
        image = vae.decode(latents, return_dict=False)[0]

    if image.ndim == 5 and image.shape[2] == 1:
        image = image.squeeze(2)
    image = (image / 2 + 0.5).clamp(0, 1)
    images = []
    for image_item in image.detach().cpu():
        image_item = image_item.permute(1, 2, 0).float().numpy()
        image_item = (image_item * 255.0).round().clip(0, 255).astype("uint8")
        images.append(Image.fromarray(image_item))
    return images


class MinimalFluxRuntime:
    def __init__(self, project_root: Path, args: SimpleNamespace, device: str):
        self.project_root = project_root
        self.args = args
        self.device = torch.device(device)
        self.worker_name = getattr(args, "worker_name", str(device))
        self.progress_position = int(getattr(args, "progress_position", 0))
        self.is_primary_shard = _is_primary_shard(args)

        self.decoder_root, model_mod, posemb_mod = bootstrap(project_root, args)
        self.get_nd_rotary_pos_embed = posemb_mod.get_nd_rotary_pos_embed

        from decoder.constants import VAE_PATH
        from decoder.diffusion.schedulers import FlowMatchDiscreteScheduler
        from decoder.vae.flux_vae import FluxAutoencoderKL

        logger.info("Building transformer model")
        cfg = model_mod.UNICOM_DIFFUSION_TRANSFORMER_CONFIGS[args.model]
        self.model = model_mod.UniComDiffusionTransformer(
            in_channels=args.latent_channels,
            out_channels=args.latent_channels,
            multitask_mask_training_type=None,
            dtype=PRECISION_TO_TYPE[args.precision],
            device=self.device,
            **cfg,
        ).to(self.device)
        self.model.eval()
        logger.info(f"Loading decoder checkpoint: {args.ckpt}")
        self._load_checkpoint(Path(args.ckpt), args.load_key, args.strict_load)

        vae_path = Path(getattr(args, "vae_path", None) or VAE_PATH[args.vae])
        logger.info(f"Loading VAE: {vae_path}")
        self.vae = FluxAutoencoderKL.from_pretrained(vae_path).to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.s_ratio = 8
        self.t_ratio = 1
        self.scheduler = FlowMatchDiscreteScheduler(
            shift=args.flow_shift_eval,
            reverse=args.flow_reverse,
            solver=args.flow_solver,
            use_flux_shift=args.use_flux_shift,
            flux_base_shift=args.flux_base_shift,
            flux_max_shift=args.flux_max_shift,
            flux_base_token=args.flux_base_token,
            flux_max_token=args.flux_max_token,
            flux_shift_factor=args.flux_shift_factor,
        )

        self.image_encoder = None
        self.image_processor = None
        logger.info(
            f"Loaded runtime on device={self.device} model={args.model} precision={args.precision} "
            f"vae={args.vae} text_projection={args.text_projection} aba_mode={args.aba_mode}"
        )

    def _lazy_init_siglip2(self):
        if self.image_encoder is not None and self.image_processor is not None:
            return

        from transformers.models.siglip2.image_processing_siglip2_fast import Siglip2ImageProcessorFast
        from decoder.visual_encoders import load_vision_model

        vision_encoder_name = getattr(self.args, "vision_encoder_name", "siglip2-so400m-patch16-naflex")
        self.image_encoder = load_vision_model(vision_encoder_name, device=self.device)
        processor_path = getattr(self.args, "siglip2_model_path", None) or getattr(self.args, "vision_encoder_path", None)
        if processor_path is None:
            from decoder.visual_encoders import VISION_ENCODER_META_INFO

            processor_path = VISION_ENCODER_META_INFO[vision_encoder_name]["path"]
        self.image_processor = Siglip2ImageProcessorFast.from_pretrained(processor_path)

    def _load_checkpoint(self, ckpt_path: Path, load_key: str, strict_load: bool):
        state = _torch_load(ckpt_path, map_location="cpu")
        state_dict = _extract_state_dict(state, load_key)
        self.model.load_state_dict(state_dict, strict=strict_load)
        logger.info("Decoder checkpoint loaded")

    def _encode_prompt_2(self, prompt: str):
        if getattr(self.args, "text_pool_type", None) is None:
            return None, None
        shape = (1, self.args.text_states_dim_2)
        zeros = torch.zeros(shape, device=self.device, dtype=PRECISION_TO_TYPE[self.args.precision])
        return zeros, zeros

    def _encode_prompt_2_batch(self, batch_size: int):
        if getattr(self.args, "text_pool_type", None) is None:
            return None, None
        shape = (batch_size, self.args.text_states_dim_2)
        zeros = torch.zeros(shape, device=self.device, dtype=PRECISION_TO_TYPE[self.args.precision])
        return zeros, zeros

    def _prepare_condition_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        image = image.convert("RGB")
        if image.size == (width, height):
            return image
        return ImageOps.fit(image, (width, height), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))

    def _encode_siglip2_image(
        self,
        image: Image.Image,
        width: int,
        height: int,
        target_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._lazy_init_siglip2()
        image = self._prepare_condition_image(image, width, height)
        select_layer = int(getattr(self.args, "siglip2_select_layer", -1))
        inputs = self.image_processor(
            images=image,
            max_num_patches=getattr(self.args, "siglip2_max_num_patches", 1024),
            return_tensors="pt",
        ).to(device=self.device)
        outputs = self.image_encoder(
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["pixel_attention_mask"],
            spatial_shapes=inputs["spatial_shapes"],
            output_hidden_states=select_layer < -1,
        )
        prompt_embeds = self.image_encoder.feature_select(outputs, select_layer).to(
            device=self.device, dtype=target_dtype
        )
        prompt_mask = torch.ones(
            (prompt_embeds.shape[0], prompt_embeds.shape[1]),
            device=self.device,
            dtype=torch.long,
        )
        return prompt_embeds, prompt_mask

    def _load_pt_condition(self, condition_path: Path, target_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        condition = _torch_load(condition_path, map_location="cpu")
        if condition.ndim == 2:
            condition = condition.unsqueeze(0)
        prompt_embeds = condition.to(device=self.device, dtype=target_dtype)
        prompt_mask = torch.ones(
            (prompt_embeds.shape[0], prompt_embeds.shape[1]),
            device=self.device,
            dtype=torch.long,
        )
        return prompt_embeds, prompt_mask

    def _get_recon_target_size(self, image_path: Path) -> Tuple[int, int]:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
        target_height, target_width = _align_hw(self.args, height, width)
        return target_height, target_width

    def _resolve_condition_mode(self, condition_mode: str, condition_path: Path) -> str:
        if condition_mode != "auto":
            return condition_mode
        return "pt" if condition_path.suffix.lower() == ".pt" else "siglip2"

    def _load_positive_condition(
        self,
        condition_mode: str,
        condition_path: Path,
        width: int,
        height: int,
        target_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        resolved_mode = self._resolve_condition_mode(condition_mode, condition_path)
        if resolved_mode == "pt":
            return self._load_pt_condition(condition_path, target_dtype)
        if resolved_mode == "siglip2":
            with Image.open(condition_path) as image:
                return self._encode_siglip2_image(image, width, height, target_dtype)
        raise ValueError(f"Unsupported condition_mode: {resolved_mode}")

    def _load_positive_condition_batch(
        self,
        condition_mode: str,
        condition_paths: List[Path],
        width: int,
        height: int,
        target_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embed_list = []
        mask_list = []
        for condition_path in condition_paths:
            prompt_embeds, prompt_mask = self._load_positive_condition(
                condition_mode=condition_mode,
                condition_path=condition_path,
                width=width,
                height=height,
                target_dtype=target_dtype,
            )
            embed_list.append(prompt_embeds)
            mask_list.append(prompt_mask)
        return torch.cat(embed_list, dim=0), torch.cat(mask_list, dim=0)

    def _load_negative_condition(
        self,
        negative_condition_mode: str,
        negative_condition_path: Optional[Path],
        prompt_embeds: torch.Tensor,
        width: int,
        height: int,
        target_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if negative_condition_mode == "zeros":
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_prompt_mask = torch.ones(
                (negative_prompt_embeds.shape[0], negative_prompt_embeds.shape[1]),
                device=self.device,
                dtype=torch.long,
            )
            return negative_prompt_embeds, negative_prompt_mask

        if negative_condition_mode == "pt":
            if negative_condition_path is None:
                raise ValueError("--negative-condition-path is required when --negative-condition-mode=pt")
            return self._load_pt_condition(negative_condition_path, target_dtype)

        if negative_condition_mode == "siglip2":
            if negative_condition_path is not None:
                with Image.open(negative_condition_path) as image:
                    return self._encode_siglip2_image(image, width, height, target_dtype)
            black_image = Image.new("RGB", (width, height), (0, 0, 0))
            return self._encode_siglip2_image(black_image, width, height, target_dtype)

        raise ValueError(f"Unsupported negative_condition_mode: {negative_condition_mode}")

    def _load_negative_condition_batch(
        self,
        negative_condition_mode: str,
        negative_condition_path: Optional[Path],
        prompt_embeds: torch.Tensor,
        width: int,
        height: int,
        target_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if negative_condition_mode == "zeros":
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_prompt_mask = torch.ones(
                (negative_prompt_embeds.shape[0], negative_prompt_embeds.shape[1]),
                device=self.device,
                dtype=torch.long,
            )
            return negative_prompt_embeds, negative_prompt_mask

        if negative_condition_mode == "pt":
            if negative_condition_path is None:
                raise ValueError("--negative-condition-path is required when --negative-condition-mode=pt")
            negative_prompt_embeds, negative_prompt_mask = self._load_pt_condition(negative_condition_path, target_dtype)
        elif negative_condition_mode == "siglip2":
            if negative_condition_path is not None:
                with Image.open(negative_condition_path) as image:
                    negative_prompt_embeds, negative_prompt_mask = self._encode_siglip2_image(
                        image, width, height, target_dtype
                    )
            else:
                black_image = Image.new("RGB", (width, height), (0, 0, 0))
                negative_prompt_embeds, negative_prompt_mask = self._encode_siglip2_image(
                    black_image, width, height, target_dtype
                )
        else:
            raise ValueError(f"Unsupported negative_condition_mode: {negative_condition_mode}")

        if negative_prompt_embeds.shape[0] == 1 and prompt_embeds.shape[0] > 1:
            negative_prompt_embeds = negative_prompt_embeds.expand(prompt_embeds.shape[0], -1, -1).contiguous()
            negative_prompt_mask = negative_prompt_mask.expand(prompt_embeds.shape[0], -1).contiguous()
        return negative_prompt_embeds, negative_prompt_mask

    def _build_rope(self, height: int, width: int, frame: int, ntk_factor: float):
        rope_sizes = [
            max(frame // self.t_ratio * self.model.patch_size[0], 1),
            height // (self.s_ratio * self.model.patch_size[1]),
            width // (self.s_ratio * self.model.patch_size[2]),
        ]
        head_dim = self.model.hidden_size // self.model.num_heads
        rope_dim_list = self.model.rope_dim_list or [head_dim // 3 for _ in range(3)]
        freqs_cos, freqs_sin = self.get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.args.rope_theta,
            use_real=True,
            theta_rescale_factor=ntk_factor,
            interpolation_factor=self.args.rope_interpolation_factor,
        )
        return freqs_cos, freqs_sin, freqs_cos.shape[0]

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        condition_path: Path,
        height: int,
        width: int,
        seed: int,
        infer_steps: int,
        cfg_scale: float,
        flux_cfg_scale: float,
        ntk_factor: float = 1.0,
        condition_mode: str = "auto",
        negative_condition_path: Optional[Path] = None,
        negative_condition_mode: str = "zeros",
    ) -> Image.Image:
        return self.generate_batch(
            prompts=[prompt],
            condition_paths=[condition_path],
            height=height,
            width=width,
            seeds=[seed],
            infer_steps=infer_steps,
            cfg_scale=cfg_scale,
            flux_cfg_scale=flux_cfg_scale,
            ntk_factor=ntk_factor,
            condition_mode=condition_mode,
            negative_condition_path=negative_condition_path,
            negative_condition_mode=negative_condition_mode,
        )[0]

    @torch.no_grad()
    def reconstruct_image(
        self,
        image_path: Path,
        seed: int,
        infer_steps: int,
        cfg_scale: float,
        flux_cfg_scale: float,
        ntk_factor: float = 1.0,
        condition_mode: str = "siglip2",
        negative_condition_path: Optional[Path] = None,
        negative_condition_mode: str = "zeros",
    ) -> Tuple[Image.Image, Image.Image]:
        target_height, target_width = self._get_recon_target_size(image_path)
        reconstructed = self.generate_batch(
            prompts=[""],
            condition_paths=[image_path],
            height=target_height,
            width=target_width,
            seeds=[seed],
            infer_steps=infer_steps,
            cfg_scale=cfg_scale,
            flux_cfg_scale=flux_cfg_scale,
            ntk_factor=ntk_factor,
            condition_mode=condition_mode,
            negative_condition_path=negative_condition_path,
            negative_condition_mode=negative_condition_mode,
        )[0]
        with Image.open(image_path) as image:
            aligned_input = self._prepare_condition_image(image.convert("RGB"), target_width, target_height)
        return aligned_input, reconstructed

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        condition_paths: List[Path],
        height: int,
        width: int,
        seeds: List[int],
        infer_steps: int,
        cfg_scale: float,
        flux_cfg_scale: float,
        ntk_factor: float = 1.0,
        condition_mode: str = "auto",
        negative_condition_path: Optional[Path] = None,
        negative_condition_mode: str = "zeros",
        progress_desc: Optional[str] = None,
    ) -> List[Image.Image]:
        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        autocast_enabled = target_dtype != torch.float32 and self.device.type == "cuda"
        vae_autocast_enabled = vae_dtype != torch.float32 and self.device.type == "cuda"
        height, width = _align_hw(self.args, height, width)

        prompt_embeds, prompt_mask = self._load_positive_condition_batch(
            condition_mode=condition_mode,
            condition_paths=condition_paths,
            width=width,
            height=height,
            target_dtype=target_dtype,
        )
        negative_prompt_embeds, negative_prompt_mask = self._load_negative_condition_batch(
            negative_condition_mode=negative_condition_mode,
            negative_condition_path=negative_condition_path,
            prompt_embeds=prompt_embeds,
            width=width,
            height=height,
            target_dtype=target_dtype,
        )

        batch_size = len(condition_paths)
        prompt_embeds_2, negative_prompt_embeds_2 = self._encode_prompt_2_batch(batch_size)
        if cfg_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_mask = torch.cat([negative_prompt_mask, prompt_mask], dim=0)
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2], dim=0)

        freqs_cos, freqs_sin, n_tokens = self._build_rope(height, width, frame=1, ntk_factor=ntk_factor)
        timesteps = _retrieve_timesteps(self.scheduler, infer_steps, self.device, n_tokens)

        generators = [torch.Generator(device=self.device).manual_seed(int(seed)) for seed in seeds]
        latents = torch.cat(
            [
                _prepare_latents(
                    self.vae,
                    self.scheduler,
                    batch_size=1,
                    num_channels_latents=self.model.config.in_channels,
                    height=height,
                    width=width,
                    frame=1,
                    dtype=target_dtype,
                    device=self.device,
                    generator=generator,
                )
                for generator in generators
            ],
            dim=0,
        )

        extra_step_kwargs = _prepare_extra_kwargs(self.scheduler.step, {})

        step_iterator = tqdm(
            timesteps,
            desc=progress_desc or f"{self.worker_name} {width}x{height} bs={len(condition_paths)}",
            leave=False,
            position=self.progress_position,
            disable=(not getattr(self.args, "show_progress", True)) or (not self.is_primary_shard),
            dynamic_ncols=True,
        )
        for t in step_iterator:
            latent_model_input = torch.cat([latents] * 2) if cfg_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            t_expand = t.repeat(latent_model_input.shape[0])
            guidance_expand = None
            if flux_cfg_scale is not None:
                guidance_expand = (
                    torch.tensor(
                        [flux_cfg_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=self.device,
                    ).to(target_dtype)
                    * 1000.0
                )

            with torch.autocast(
                device_type="cuda",
                dtype=target_dtype,
                enabled=autocast_enabled,
            ):
                noise_pred = self.model(
                    latent_model_input,
                    t_expand,
                    text_states=prompt_embeds,
                    text_mask=prompt_mask,
                    text_states_2=prompt_embeds_2,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    guidance=guidance_expand,
                    return_dict=True,
                    generator=generators,
                )["x"]

            if cfg_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]

        if self.is_primary_shard:
            logger.info(f"Finished batch: batch_size={len(condition_paths)} size={width}x{height}")
        return _decode_latents(self.vae, latents, vae_dtype, vae_autocast_enabled)


def parse_cli():
    parser = argparse.ArgumentParser(description="Minimal single-device Flux detokenizer inference.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def _build_output_filename(row: dict) -> str:
    index = row["index"]
    prompt = row["prompt"][:20].replace("/", "_")
    sample_j = row.get("sample_j")
    if sample_j not in (None, ""):
        return f"{index}_{sample_j}_{prompt}.png"
    return f"{index}_{prompt}.png"


def _build_eval_gt_filename(row: dict) -> str:
    return f"{row['index']}.png"


def _concat_images(left: Image.Image, right: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def _flush_batch(runtime, batch_rows, cli_args, batch_index: Optional[int] = None, total_batches: Optional[int] = None):
    if not batch_rows:
        return
    _log_batch(runtime, batch_rows, cli_args, batch_index=batch_index, total_batches=total_batches)
    images = runtime.generate_batch(
        prompts=[row["prompt"] for row in batch_rows],
        condition_paths=[Path(row[cli_args.condition_column]) for row in batch_rows],
        height=int(batch_rows[0]["height"]),
        width=int(batch_rows[0]["width"]),
        seeds=[int(row["seed"]) for row in batch_rows],
        infer_steps=cli_args.infer_steps,
        cfg_scale=cli_args.cfg_scale,
        flux_cfg_scale=cli_args.flux_cfg_scale,
        ntk_factor=cli_args.ntk_factor,
        condition_mode=cli_args.condition_mode,
        negative_condition_path=getattr(cli_args, "negative_condition_path", None),
        negative_condition_mode=cli_args.negative_condition_mode,
        progress_desc=_format_batch_header(batch_index, total_batches) if batch_index is not None else None,
    )
    for row, image in zip(batch_rows, images):
        save_path = _get_output_path(row, cli_args)
        image.save(save_path)
        if runtime.is_primary_shard:
            logger.info(f"Sample save to: {save_path}")


def _run_eval_gt(runtime, cli_args, pending_rows: List[dict]):
    total_batches = len(pending_rows) if runtime.is_primary_shard else None
    for batch_index, row in enumerate(pending_rows, start=1):
        _log_batch(runtime, [row], cli_args, batch_index=batch_index, total_batches=total_batches)
        image_path = Path(row[cli_args.condition_column])
        input_image, recon_image = runtime.reconstruct_image(
            image_path=image_path,
            seed=int(row["seed"]),
            infer_steps=cli_args.infer_steps,
            cfg_scale=cli_args.cfg_scale,
            flux_cfg_scale=cli_args.flux_cfg_scale,
            ntk_factor=cli_args.ntk_factor,
            condition_mode=cli_args.condition_mode,
            negative_condition_path=getattr(cli_args, "negative_condition_path", None),
            negative_condition_mode=cli_args.negative_condition_mode,
        )
        output_image = _concat_images(input_image, recon_image) if cli_args.comparison_output else recon_image
        save_path = _get_output_path(row, cli_args)
        output_image.save(save_path)
        if runtime.is_primary_shard:
            logger.info(f"Sample save to: {save_path}")


def run_cli():
    cli_args = parse_cli()
    runtime_args = build_runtime_args(cli_args)

    if getattr(runtime_args, "worker_name", None) is None:
        runtime_args.worker_name = f"shard{runtime_args.shard_index}"

    _setup_logger(runtime_args.worker_name)
    if runtime_args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= runtime_args.shard_index < runtime_args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    runtime = MinimalFluxRuntime(
        cli_args.project_root,
        runtime_args,
        runtime_args.device,
    )

    runtime_args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Start inference csv={runtime_args.csv_path} output_dir={runtime_args.output_dir} "
        f"device={runtime_args.device} batch_size={runtime_args.batch_size} "
        f"shard={runtime_args.shard_index}/{runtime_args.num_shards}"
    )
    pending_rows = _collect_pending_shard_rows(runtime_args, log_skips=runtime.is_primary_shard)
    if runtime_args.mode == "eval_gt":
        _run_eval_gt(runtime, runtime_args, pending_rows)
    else:
        total_batches = _count_total_batches(pending_rows, runtime_args) if runtime.is_primary_shard else None
        batch_rows = []
        batch_index = 0
        for row in pending_rows:
            if batch_rows:
                same_height = int(batch_rows[0]["height"]) == int(row["height"])
                same_width = int(batch_rows[0]["width"]) == int(row["width"])
                if not (same_height and same_width) or len(batch_rows) >= runtime_args.batch_size:
                    batch_index += 1
                    _flush_batch(runtime, batch_rows, runtime_args, batch_index=batch_index, total_batches=total_batches)
                    batch_rows = []
            batch_rows.append(row)
        if batch_rows:
            batch_index += 1
            _flush_batch(runtime, batch_rows, runtime_args, batch_index=batch_index, total_batches=total_batches)
    logger.info("Inference complete")
