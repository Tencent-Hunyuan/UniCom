import argparse
import csv
import os
import time

import torch
import torch.multiprocessing as mp
from loguru import logger
from transformers import AutoModelForCausalLM


CSV_FIELDS = [
    "index", "sample_j", "prompt", "seed",
    "height", "width",
    "guidance_scale", "diff_steps", "flow_shift", "bot_task",
    "latent_shape", "feature_path", "time_s", "cot"
]


def setup_logger(worker_name: str):
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>"
        + worker_name
        + "</cyan> | <level>{message}</level>",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="UniCom CSV Inference")

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Input CSV with columns: index, seed, prompt")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs for data-parallel inference. "
                             "Each GPU loads the full model independently.")
    parser.add_argument("--n-samples-per-prompt", type=int, default=1,
                        help="Number of images to generate per prompt (uses seed+j for j-th sample)")
    parser.add_argument("--image-size", type=str, default="1024")
    parser.add_argument("--diff-infer-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--cot-max-new-tokens", type=int, default=1536,
                        help="Maximum new tokens for the CoT/text stage.")
    parser.add_argument("--cot-no-do-sample", action="store_false", dest="cot_do_sample",
                        help="Disable sampling for the CoT/text stage and use greedy decoding.")
    parser.add_argument("--bot-task", type=str, default="think_recaption",
                        choices=["vanilla", "recaption", "think_recaption"])
    parser.add_argument("--use-system-prompt", type=str, default="en_unified")
    parser.add_argument("--drop-think", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--start-index", type=int, default=None,
                        help="Start from this index (inclusive), useful for resuming")
    parser.add_argument("--end-index", type=int, default=None,
                        help="End at this index (exclusive)")

    parser.set_defaults(cot_do_sample=True)
    return parser.parse_args()


def load_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "index": int(row["index"]),
                "seed": int(row["seed"]),
                "prompt": row["prompt"],
            })
    return rows


def filter_rows(rows, start_index=None, end_index=None):
    if start_index is not None:
        rows = [r for r in rows if r["index"] >= start_index]
    if end_index is not None:
        rows = [r for r in rows if r["index"] < end_index]
    return rows


def load_existing_keys(result_csv_path):
    existing_keys = set()
    if os.path.exists(result_csv_path):
        with open(result_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add((int(row["index"]), int(row["sample_j"])))
    return existing_keys


def worker_fn(rank, world_size, args):
    setup_logger(f"unicom-csv-gpu{rank}")
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    abs_output_dir = os.path.abspath(args.output_dir)

    if args.image_size == "auto":
        image_size = "auto"
    else:
        s = int(args.image_size)
        image_size = (s, s)

    logger.info(f"Loading model on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.load_tokenizer(args.model_path)

    if args.drop_think:
        model.generation_config.drop_think = True
    model.generation_config.pad_token_id = model.tokenizer.pad_token_id
    model.generation_config.do_sample = args.cot_do_sample
    logger.info(f"Model loaded dtype={model.dtype}")

    rows = filter_rows(load_csv(args.csv_path), args.start_index, args.end_index)
    shard_rows = rows[rank::world_size]
    n_samples = args.n_samples_per_prompt
    shard_total = len(shard_rows) * n_samples
    global_total = len(rows) * n_samples

    result_csv_path = os.path.join(abs_output_dir, "results.csv")
    existing_keys = load_existing_keys(result_csv_path)

    if rank == 0:
        logger.info(
            f"Dispatch prompts={len(rows)} samples_per_prompt={n_samples} "
            f"total_samples={global_total} gpus={world_size}"
        )

    shard_csv_path = os.path.join(abs_output_dir, f".shard_{rank}.csv")
    csvfile = open(shard_csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
    writer.writeheader()

    done_count = 0
    t_total_start = time.time()

    for row in shard_rows:
        idx = row["index"]
        base_seed = row["seed"]
        prompt = row["prompt"]

        for j in range(n_samples):
            if (idx, j) in existing_keys:
                done_count += 1
                continue

            seed = base_seed + j
            sample_id = f"{idx}_{j}" if n_samples > 1 else str(idx)
            pt_path = os.path.join(abs_output_dir, f"{sample_id}.pt")

            if args.verbose >= 1:
                logger.info(
                    f"[{done_count + 1}/{shard_total}] idx={idx} sample_j={j} "
                    f"seed={seed} prompt='{prompt[:60]}'"
                )

            t_start = time.time()
            try:
                cot_text, outputs, resolved_image_size = model.generate_image(
                    prompt=prompt,
                    seed=seed,
                    image_size=image_size,
                    use_system_prompt=args.use_system_prompt,
                    bot_task=args.bot_task,
                    max_new_tokens=args.cot_max_new_tokens,
                    diff_infer_steps=args.diff_infer_steps,
                    diff_guidance_scale=args.guidance_scale,
                    flow_shift=args.flow_shift,
                    verbose=max(0, args.verbose - 1),
                )
            except Exception:
                logger.exception(f"Generation failed for idx={idx} sample_j={j} seed={seed}")
                done_count += 1
                continue

            t_elapsed = time.time() - t_start

            if args.verbose >= 1 and cot_text:
                cot_preview = cot_text[0]
                logger.info(f"CoT: {cot_preview} ({t_elapsed:.1f}s)")

            latents = outputs.samples if hasattr(outputs, "samples") else outputs
            latents = latents.squeeze(0)
            torch.save(latents.cpu(), pt_path)

            if isinstance(resolved_image_size, (tuple, list)):
                h, w = resolved_image_size
            else:
                h, w = "", ""

            writer.writerow({
                "index": idx,
                "sample_j": j,
                "prompt": prompt,
                "seed": seed,
                "height": h,
                "width": w,
                "guidance_scale": args.guidance_scale,
                "diff_steps": args.diff_infer_steps,
                "flow_shift": args.flow_shift,
                "bot_task": args.bot_task,
                "latent_shape": str(list(latents.shape)),
                "feature_path": pt_path,
                "time_s": f"{t_elapsed:.1f}",
                "cot": cot_text
            })
            csvfile.flush()
            done_count += 1

    csvfile.close()
    t_total = time.time() - t_total_start
    logger.info(
        f"Shard complete done={done_count}/{shard_total} total_time={t_total:.1f}s "
        f"avg_time={t_total / max(done_count, 1):.1f}s_per_sample"
    )


def merge_shard_csvs(output_dir, world_size):
    abs_output_dir = os.path.abspath(output_dir)
    result_csv_path = os.path.join(abs_output_dir, "results.csv")

    existing_rows = []
    existing_keys = set()
    if os.path.exists(result_csv_path):
        with open(result_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                existing_keys.add((row["index"], row["sample_j"]))

    new_rows = []
    for rank in range(world_size):
        shard_path = os.path.join(abs_output_dir, f".shard_{rank}.csv")
        if not os.path.exists(shard_path):
            continue
        with open(shard_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["index"], row["sample_j"])
                if key not in existing_keys:
                    new_rows.append(row)
                    existing_keys.add(key)
        os.remove(shard_path)

    all_rows = existing_rows + new_rows
    all_rows.sort(key=lambda r: (int(r["index"]), int(r["sample_j"])))

    with open(result_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    return result_csv_path, len(all_rows)


def main():
    args = parse_args()
    setup_logger("unicom-csv")
    os.makedirs(args.output_dir, exist_ok=True)

    available_gpus = torch.cuda.device_count()
    requested_gpus = min(args.num_gpus, available_gpus)
    if requested_gpus < 1:
        raise RuntimeError("No CUDA GPUs available")

    rows = filter_rows(load_csv(args.csv_path), args.start_index, args.end_index)
    total_samples = len(rows) * args.n_samples_per_prompt
    if total_samples < 1:
        raise ValueError("No samples to run after applying CSV and index filters")

    num_gpus = min(requested_gpus, total_samples)

    logger.info(
        f"Start CSV inference requested_gpus={args.num_gpus} available_gpus={available_gpus} "
        f"effective_gpus={num_gpus} total_samples={total_samples} output_dir={os.path.abspath(args.output_dir)}"
    )

    t_wall_start = time.time()

    if num_gpus == 1:
        worker_fn(0, 1, args)
    else:
        mp.spawn(worker_fn, args=(num_gpus, args), nprocs=num_gpus, join=True)

    result_csv_path, total_rows = merge_shard_csvs(args.output_dir, num_gpus)
    t_wall = time.time() - t_wall_start

    logger.info(f"All done samples={total_rows} wall_time={t_wall:.1f}s")
    logger.info(f"Results: {result_csv_path}")


if __name__ == "__main__":
    main()
