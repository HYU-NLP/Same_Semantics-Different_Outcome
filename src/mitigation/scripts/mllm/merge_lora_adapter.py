#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base model for vLLM inference."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tokenizer.save_pretrained(output)
    processor = AutoProcessor.from_pretrained(args.adapter, trust_remote_code=True)
    processor.save_pretrained(output)
    print(output)


if __name__ == "__main__":
    main()
