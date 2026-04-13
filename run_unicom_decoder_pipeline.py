import argparse
import csv
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
UNICOM_ROOT = PROJECT_ROOT / "UniCom"
DECODER_ROOT = PROJECT_ROOT / "UniCom_Decoder"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UniCom latent generation and diffusion decoder in one pipeline."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompt", type=str, help="Single prompt to generate.")
    input_group.add_argument(
        "--csv-path",
        type=Path,
        help="CSV input with columns: index, seed, prompt.",
    )

    parser.add_argument("--model-path", type=Path, required=True, help="UniCom HF model path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Pipeline output root. Latents and images will be stored under this directory.",
    )
    parser.add_argument("--image", type=str, nargs="*", default=None, help="Optional input image(s) for editing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=str, default="1024")
    parser.add_argument("--diff-infer-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--cot-max-new-tokens", type=int, default=1536)
    parser.add_argument("--cot-no-do-sample", action="store_false", dest="cot_do_sample")
    parser.add_argument(
        "--bot-task",
        type=str,
        default="think_recaption",
        choices=["vanilla", "recaption", "think_recaption"],
    )
    parser.add_argument("--use-system-prompt", type=str, default="en_unified")
    parser.add_argument("--drop-think", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument(
        "--unicom-device",
        type=str,
        default="cuda:0",
        help="Device for single-prompt UniCom inference.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for CSV UniCom inference.",
    )
    parser.add_argument("--n-samples-per-prompt", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)

    parser.add_argument(
        "--decoder-device",
        type=str,
        default="0",
        help="Decoder device. Supports single GPU ('0' or 'cuda:0') or multi GPU ('0,1').",
    )
    parser.add_argument("--decoder-batch-size", type=int, default=1)
    parser.add_argument("--decoder-infer-steps", type=int, default=100)
    parser.add_argument("--decoder-cfg-scale", type=float, default=1.0)
    parser.add_argument("--decoder-flux-cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--decoder-aba-mode",
        type=str,
        default="eval_compression_64_siglip",
        help="Decoder aba_mode. Keep this aligned with the exported decoder checkpoint.",
    )
    parser.add_argument(
        "--decoder-negative-condition-mode",
        type=str,
        default="siglip2",
        choices=["zeros", "pt", "siglip2"],
    )
    parser.add_argument(
        "--decoder-ckpt",
        type=Path,
        default=PROJECT_ROOT / "model_zoo" / "unicom_decoder_transformer.pt",
    )
    parser.add_argument(
        "--decoder-vae-path",
        type=Path,
        default=PROJECT_ROOT / "model_zoo" / "flux-vae",
    )
    parser.add_argument(
        "--decoder-siglip2-model-path",
        type=Path,
        default=PROJECT_ROOT / "model_zoo" / "siglip2-so400m-patch16-naflex",
    )
    parser.add_argument(
        "--keep-intermediate-config",
        action="store_true",
        help="Keep generated decoder CSV/YAML files under output-dir/_pipeline.",
    )

    parser.set_defaults(cot_do_sample=True)
    return parser.parse_args()


def run_command(cmd):
    printable = " ".join(str(part) for part in cmd)
    print(f"$ {printable}", flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def load_rows_with_features(csv_path: Path):
    if not csv_path.exists():
        return [], []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = []
        for row in reader:
            feature_path = row.get("feature_path")
            if feature_path:
                rows.append(row)
    return fieldnames, rows


def get_max_index(csv_path: Path):
    _, rows = load_rows_with_features(csv_path)
    if not rows:
        return -1
    return max(int(row["index"]) for row in rows)


def row_in_requested_range(row: dict, args) -> bool:
    index = int(row["index"])
    if args.start_index is not None and index < args.start_index:
        return False
    if args.end_index is not None and index >= args.end_index:
        return False

    sample_j = row.get("sample_j")
    if sample_j not in (None, "") and int(sample_j) >= args.n_samples_per_prompt:
        return False
    return True


def get_requested_latent_rows(results_csv_path: Path, args, previous_max_index: int | None = None):
    fieldnames, rows = load_rows_with_features(results_csv_path)
    if args.prompt is not None:
        if previous_max_index is None:
            return fieldnames, rows
        return fieldnames, [row for row in rows if int(row["index"]) > previous_max_index]
    return fieldnames, [row for row in rows if row_in_requested_range(row, args)]


def compute_expected_samples(args):
    if args.prompt is not None:
        return args.n_samples_per_prompt

    rows = load_csv_rows(args.csv_path)
    if args.start_index is not None:
        rows = [row for row in rows if int(row["index"]) >= args.start_index]
    if args.end_index is not None:
        rows = [row for row in rows if int(row["index"]) < args.end_index]
    return len(rows) * args.n_samples_per_prompt


def load_csv_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def has_complete_latent_cache(rows, expected_count: int):
    if expected_count <= 0 or len(rows) != expected_count:
        return False
    for row in rows:
        feature_path = row.get("feature_path")
        if not feature_path or not Path(feature_path).exists():
            return False
    return True


def write_decoder_csv(csv_path: Path, fieldnames, rows):
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_decoder_config(config_path: Path, args, decoder_csv_path: Path, decoder_output_dir: Path):
    config = {
        "base": str((DECODER_ROOT / "configs" / "base.yaml").resolve()),
        "device": args.decoder_device,
        "csv_path": str(decoder_csv_path.resolve()),
        "output_dir": str(decoder_output_dir.resolve()),
        "ckpt": str(args.decoder_ckpt.resolve()),
        "vae_path": str(args.decoder_vae_path.resolve()),
        "siglip2_model_path": str(args.decoder_siglip2_model_path.resolve()),
        "infer_steps": args.decoder_infer_steps,
        "aba_mode": args.decoder_aba_mode,
        "cfg_scale": args.decoder_cfg_scale,
        "flux_cfg_scale": args.decoder_flux_cfg_scale,
        "condition_mode": "pt",
        "condition_column": "feature_path",
        "negative_condition_mode": args.decoder_negative_condition_mode,
        "batch_size": args.decoder_batch_size,
        "num_shards": 1,
    }
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def build_decoder_output_filename(row: dict) -> str:
    index = row["index"]
    prompt = row["prompt"][:20].replace("/", "_")
    sample_j = row.get("sample_j")
    if sample_j not in (None, ""):
        return f"{index}_{sample_j}_{prompt}.png"
    return f"{index}_{prompt}.png"


def filter_rows_needing_decode(rows, decoder_output_dir: Path):
    pending_rows = []
    for row in rows:
        output_path = decoder_output_dir / build_decoder_output_filename(row)
        if not output_path.exists():
            pending_rows.append(row)
    return pending_rows


def build_unicom_command(args, latent_output_dir: Path):
    python_bin = sys.executable
    if args.prompt is not None:
        cmd = [
            python_bin,
            str((UNICOM_ROOT / "inference.py").resolve()),
            "--model-path",
            str(args.model_path.resolve()),
            "--prompt",
            args.prompt,
            "--output-dir",
            str(latent_output_dir.resolve()),
            "--device",
            args.unicom_device,
            "--seed",
            str(args.seed),
            "--n-samples-per-prompt",
            str(args.n_samples_per_prompt),
            "--image-size",
            args.image_size,
            "--diff-infer-steps",
            str(args.diff_infer_steps),
            "--guidance-scale",
            str(args.guidance_scale),
            "--flow-shift",
            str(args.flow_shift),
            "--cot-max-new-tokens",
            str(args.cot_max_new_tokens),
            "--bot-task",
            args.bot_task,
            "--use-system-prompt",
            args.use_system_prompt,
            "--verbose",
            str(args.verbose),
        ]
        if not args.cot_do_sample:
            cmd.append("--cot-no-do-sample")
        if args.drop_think:
            cmd.append("--drop-think")
        if args.image:
            cmd.extend(["--image", *args.image])
        return cmd

    cmd = [
        python_bin,
        str((UNICOM_ROOT / "inference_from_csv.py").resolve()),
        "--model-path",
        str(args.model_path.resolve()),
        "--csv-path",
        str(args.csv_path.resolve()),
        "--output-dir",
        str(latent_output_dir.resolve()),
        "--num-gpus",
        str(args.num_gpus),
        "--n-samples-per-prompt",
        str(args.n_samples_per_prompt),
        "--image-size",
        args.image_size,
        "--diff-infer-steps",
        str(args.diff_infer_steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--flow-shift",
        str(args.flow_shift),
        "--cot-max-new-tokens",
        str(args.cot_max_new_tokens),
        "--bot-task",
        args.bot_task,
        "--use-system-prompt",
        args.use_system_prompt,
        "--verbose",
        str(args.verbose),
    ]
    if not args.cot_do_sample:
        cmd.append("--cot-no-do-sample")
    if args.drop_think:
        cmd.append("--drop-think")
    if args.start_index is not None:
        cmd.extend(["--start-index", str(args.start_index)])
    if args.end_index is not None:
        cmd.extend(["--end-index", str(args.end_index)])
    return cmd


def main():
    args = parse_args()

    if args.prompt is not None and args.num_gpus != 1:
        print("Warning: --prompt mode uses UniCom single-GPU inference; --num-gpus is ignored.", flush=True)

    output_dir = args.output_dir.resolve()
    latent_output_dir = output_dir / "latents"
    decoder_output_dir = output_dir / "images"
    pipeline_dir = output_dir / "_pipeline"
    latent_output_dir.mkdir(parents=True, exist_ok=True)
    decoder_output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    results_csv_path = latent_output_dir / "results.csv"
    expected_count = compute_expected_samples(args)
    previous_max_index = get_max_index(results_csv_path) if args.prompt is not None else None
    fieldnames, existing_rows = get_requested_latent_rows(results_csv_path, args)
    can_reuse_latents = args.prompt is None and has_complete_latent_cache(existing_rows, expected_count)
    decoder_rows = existing_rows if can_reuse_latents else []

    if can_reuse_latents:
        print(
            f"Found complete latent cache in {results_csv_path}, skipping UniCom inference and running decoder only.",
            flush=True,
        )
    else:
        run_command(build_unicom_command(args, latent_output_dir))
        fieldnames, decoder_rows = get_requested_latent_rows(results_csv_path, args, previous_max_index)
        if len(decoder_rows) != expected_count:
            raise RuntimeError(
                f"UniCom inference did not produce all expected latents for the requested range: "
                f"expected {expected_count}, found {len(decoder_rows)} in {results_csv_path}"
            )

    decoder_rows = filter_rows_needing_decode(decoder_rows, decoder_output_dir)
    if not decoder_rows:
        print("All expected decoded images already exist, skipping decoder inference.", flush=True)
        print("\nPipeline finished.", flush=True)
        print(f"Latents CSV: {results_csv_path}", flush=True)
        print(f"Decoded images: {decoder_output_dir}", flush=True)
        return

    decoder_csv_path = pipeline_dir / "decoder_input.csv"
    decoder_config_path = pipeline_dir / "decoder_config.yaml"
    write_decoder_csv(decoder_csv_path, fieldnames, decoder_rows)
    write_decoder_config(decoder_config_path, args, decoder_csv_path, decoder_output_dir)

    decoder_cmd = [
        "bash",
        str((DECODER_ROOT / "scripts" / "run.sh").resolve()),
        "--config-file",
        str(decoder_config_path.resolve()),
    ]
    run_command(decoder_cmd)

    if not args.keep_intermediate_config:
        decoder_csv_path.unlink(missing_ok=True)
        decoder_config_path.unlink(missing_ok=True)
        if not any(pipeline_dir.iterdir()):
            pipeline_dir.rmdir()

    print("\nPipeline finished.", flush=True)
    print(f"Latents CSV: {results_csv_path}", flush=True)
    print(f"Decoded images: {decoder_output_dir}", flush=True)


if __name__ == "__main__":
    main()
