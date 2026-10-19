#!/usr/bin/env python3
"""vLLM evaluator matching ConflictQA code/mcq/uni_internal=true_para.py.

This keeps outputs outside the source data directory so reruns do not overwrite
./data/MMKC/ConflictQA/data/mcq/*/uni_internal=true_para.json.
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams


def build_text_prompt(row):
    return f"""{row['counter_memory']}

Question: {row['question']}

Answer in 1–5 words. Do not use a full sentence."""


def build_image_prompt(row):
    return f"""Question: {row['question']}

Answer in 1–5 words. Do not use a full sentence."""


def generate_three(mllm, messages, sampling_params, num_samples):
    answers = []
    for _ in range(num_samples):
        output = mllm.chat(messages, sampling_params=sampling_params)[0].outputs[0].text.strip().strip(".")
        answers.append(output)
    return answers


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--image_root", default="./data/MMKC/ConflictQA/image/gpt4/counter")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--mode", choices=["text", "image", "both"], default="both")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "uni_internal=true_para_vllm.json"

    with open(args.data_path, encoding="utf-8") as f:
        data = json.load(f)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    mllm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 10},
        allowed_local_media_path="./data/MMKC/ConflictQA/image",
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    if args.mode in {"text", "both"}:
        for row in tqdm(data, desc="text"):
            messages = [{"role": "user", "content": [{"type": "text", "text": build_text_prompt(row)}]}]
            row["output-text"] = generate_three(mllm, messages, sampling_params, args.num_samples)

    if args.mode in {"image", "both"}:
        image_root = Path(args.image_root)
        for row in tqdm(data, desc="image"):
            image_path = image_root / f"counter_{row['original_idx']}.jpg"
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                    {"type": "text", "text": build_image_prompt(row)},
                ],
            }]
            row["output-image"] = generate_three(mllm, messages, sampling_params, args.num_samples)

    output_file.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    print(output_file)


if __name__ == "__main__":
    main()
