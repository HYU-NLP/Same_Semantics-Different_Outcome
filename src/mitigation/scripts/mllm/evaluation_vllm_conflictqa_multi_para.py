#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams


def chat3(llm, messages, sampling_params, num_samples):
    answers = []
    for _ in range(num_samples):
        out = llm.chat(messages, sampling_params=sampling_params)[0].outputs[0].text.strip().strip(".")
        answers.append(out)
    return answers


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_path", default="./data/MMKC/ConflictQA/data/mcq/qwen7b/mcq_internal=true_para_v2.json")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=5)
    args = parser.parse_args()

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 10},
        allowed_local_media_path="./data/MMKC/ConflictQA/image",
    )
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )

    with open(args.data_path, encoding="utf-8") as f:
        data = json.load(f)

    for d in tqdm(data, desc="itq1"):
        query = d["question"]
        original_idx2 = d["original_idx2"]
        counter_memory = d["counter_memory"]
        counter_image_path2 = f"./data/MMKC/ConflictQA/image/chat_gpt4/counter/counter_{original_idx2}.jpg"
        prompt = f"""{counter_memory}

Question: {query}

Answer in 1–5 words. Do not use a full sentence."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{counter_image_path2}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        d["output-itq1"] = chat3(llm, messages, sampling_params, args.num_samples)

    for d in tqdm(data, desc="tiq1"):
        query = d["question"]
        original_idx2 = d["original_idx2"]
        counter_memory = d["counter_memory"]
        counter_image_path2 = f"./data/MMKC/ConflictQA/image/chat_gpt4/counter/counter_{original_idx2}.jpg"
        prompt1 = f"{counter_memory}"
        prompt2 = f"""Question: {query}

Answer in 1–5 words. Do not use a full sentence."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt1},
                    {"type": "image_url", "image_url": {"url": f"file://{counter_image_path2}"}},
                    {"type": "text", "text": prompt2},
                ],
            }
        ]
        d["output-tiq1"] = chat3(llm, messages, sampling_params, args.num_samples)

    for d in tqdm(data, desc="itq2"):
        query = d["question"]
        original_idx = d["original_idx"]
        counter_memory2 = d["counter_memory2"]
        counter_image_path = f"./data/MMKC/ConflictQA/image/gpt4/counter/counter_{original_idx}.jpg"
        prompt = f"""{counter_memory2}

Question: {query}

Answer in 1–5 words. Do not use a full sentence."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{counter_image_path}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        d["output-itq2"] = chat3(llm, messages, sampling_params, args.num_samples)

    for d in tqdm(data, desc="tiq2"):
        query = d["question"]
        original_idx = d["original_idx"]
        counter_memory2 = d["counter_memory2"]
        counter_image_path = f"./data/MMKC/ConflictQA/image/gpt4/counter/counter_{original_idx}.jpg"
        prompt1 = f"{counter_memory2}"
        prompt2 = f"""Question: {query}

Answer in 1–5 words. Do not use a full sentence."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt1},
                    {"type": "image_url", "image_url": {"url": f"file://{counter_image_path}"}},
                    {"type": "text", "text": prompt2},
                ],
            }
        ]
        d["output-tiq2"] = chat3(llm, messages, sampling_params, args.num_samples)

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "multi_internal=true_para_vllm.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(out_file)


if __name__ == "__main__":
    main()
