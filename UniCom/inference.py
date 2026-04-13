import argparse
import os
import csv
import time

import torch
from loguru import logger
from transformers import AutoModelForCausalLM


def setup_logger():
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>unicom-infer</cyan> | <level>{message}</level>",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="UniCom Inference")

    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to HF model directory (from convert_checkpoint.py)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--image", type=str, nargs="*", default=None,
                        help="Input image path(s) for editing tasks")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device for UniCom inference, e.g. cuda:0")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=str, default="1024",
                        help="Image size: integer (e.g. 1024) or 'auto'")
    parser.add_argument("--diff-infer-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--n-samples-per-prompt", type=int, default=1,
                        help="Number of images to generate for the input prompt (uses seed+j for j-th sample)")
    parser.add_argument("--cot-max-new-tokens", type=int, default=1536,
                        help="Maximum new tokens for the CoT/text stage.")
    parser.add_argument("--cot-no-do-sample", action="store_false", dest="cot_do_sample",
                        help="Disable sampling for the CoT/text stage and use greedy decoding.")
    parser.add_argument("--bot-task", type=str, default="think_recaption",
                        choices=["vanilla", "recaption", "think_recaption"])
    parser.add_argument("--use-system-prompt", type=str, default="en_unified",
                        choices=["en_vanilla","en_recaption", "en_unified"])
    parser.add_argument("--drop-think", action="store_true")
    parser.add_argument("--verbose", type=int, default=2,
                        help="Verbosity level: 0=quiet, 1=basic, 2=full")

    parser.set_defaults(cot_do_sample=True)
    return parser.parse_args()


def main():
    setup_logger()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.image_size == "auto":
        image_size = "auto"
    else:
        s = int(args.image_size)
        image_size = (s, s)

    logger.info("Start UniCom single inference")
    logger.info(f"model_path={args.model_path} device={args.device} output_dir={os.path.abspath(args.output_dir)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.load_tokenizer(args.model_path)

    if args.drop_think:
        model.generation_config.drop_think = True
    model.generation_config.pad_token_id = model.tokenizer.pad_token_id
    model.generation_config.do_sample = args.cot_do_sample

    logger.info(f"Model loaded dtype={model.dtype}")

    prompt = args.prompt
    image_input = args.image

    if args.verbose >= 1:
        logger.info(f"Prompt: {prompt}")
        if image_input:
            logger.info(f"Input images: {image_input}")
        logger.info(
            f"image_size={image_size} seed={args.seed} n_samples_per_prompt={args.n_samples_per_prompt} bot_task={args.bot_task} "
            f"diff_steps={args.diff_infer_steps} guidance_scale={args.guidance_scale} "
            f"flow_shift={args.flow_shift} cot_max_new_tokens={args.cot_max_new_tokens} "
            f"cot_do_sample={args.cot_do_sample}"
        )

    csv_path = os.path.join(args.output_dir, "results.csv")
    abs_output_dir = os.path.abspath(args.output_dir)

    next_index = 0
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    next_index = max(next_index, int(row[0]) + 1)

    write_header = not os.path.exists(csv_path)
    for j in range(args.n_samples_per_prompt):
        sample_seed = args.seed + j
        t_start = time.time()

        cot_text, outputs, resolved_image_size = model.generate_image(
            prompt=prompt,
            image=image_input,
            seed=sample_seed,
            image_size=image_size,
            use_system_prompt=args.use_system_prompt,
            bot_task=args.bot_task,
            max_new_tokens=args.cot_max_new_tokens,
            diff_infer_steps=args.diff_infer_steps,
            diff_guidance_scale=args.guidance_scale,
            flow_shift=args.flow_shift,
            verbose=args.verbose,
        )

        t_elapsed = time.time() - t_start

        if hasattr(outputs, 'samples'):
            latents = outputs.samples
        else:
            latents = outputs
        latents = latents.squeeze(0)

        current_index = next_index + j
        pt_path = os.path.join(abs_output_dir, f"{current_index}.pt")
        torch.save(latents.cpu(), pt_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["index", "prompt", "cot_text", "seed",
                                  "height", "width",
                                  "guidance_scale", "diff_steps", "flow_shift",
                                  "bot_task", "latent_shape", "feature_path", "time_s"])
                write_header = False
            cot_str = cot_text[0] if cot_text else ""
            if isinstance(resolved_image_size, (tuple, list)):
                h, w = resolved_image_size
            else:
                h, w = "", ""
            writer.writerow([
                current_index,
                prompt, cot_str, sample_seed,
                h, w, args.guidance_scale,
                args.diff_infer_steps, args.flow_shift,
                args.bot_task, str(list(latents.shape)),
                pt_path, f"{t_elapsed:.1f}",
            ])

        logger.info(
            f"Generation complete time={t_elapsed:.1f}s index={current_index} sample_j={j} "
            f"seed={sample_seed} latent_shape={tuple(latents.shape)} latent_dtype={latents.dtype}"
        )
        logger.info(f"Saved latent: {pt_path}")

    logger.info(f"Saved csv: {csv_path}")


if __name__ == "__main__":
    main()
