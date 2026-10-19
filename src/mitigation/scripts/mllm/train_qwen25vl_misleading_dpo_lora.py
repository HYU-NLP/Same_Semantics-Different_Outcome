#!/usr/bin/env python3
"""Minimal multimodal DPO LoRA training for Qwen2.5-VL.

This is intentionally narrow: prompt + image is shared, chosen is the factual
answer, rejected is the visually suggested counter answer.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def read_jsonl(path, max_samples=None, seed=123):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    random.Random(seed).shuffle(rows)
    if max_samples:
        rows = rows[:max_samples]
    return rows


def build_prompt(processor, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode(processor, question, image_path, answer, max_len, device):
    prompt = build_prompt(processor, question)
    eos = processor.tokenizer.eos_token or ""
    full_text = (prompt + str(answer).strip() + eos).rstrip()
    image = Image.open(image_path).convert("RGB")
    prompt_inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_len,
    )
    prompt_len = int(prompt_inputs["attention_mask"].sum().item())
    inputs = processor(
        text=[full_text],
        images=[image],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_len,
    )
    return {k: v.to(device) for k, v in inputs.items()}, prompt_len


def sequence_logprob(model, inputs, prompt_len):
    out = model(**inputs)
    logits = out.logits[:, :-1, :]
    labels = inputs["input_ids"][:, 1:]
    attn = inputs["attention_mask"][:, 1:].bool()
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    response_mask = attn.clone()
    response_mask[:, : max(prompt_len - 1, 0)] = False
    denom = response_mask.sum().clamp_min(1)
    return (token_logps * response_mask).sum() / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=1834)
    parser.add_argument("--max_len", type=int, default=4500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.dataset, args.max_samples, args.seed)
    processor = AutoProcessor.from_pretrained(args.model)

    policy = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=None
    ).to(device)
    ref = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=None
    ).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(policy, lora)
    policy.train()
    policy.gradient_checkpointing_enable()
    policy.print_trainable_parameters()

    optim = AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr)
    total_steps = math.ceil(len(rows) * args.epochs / args.grad_accum)
    pbar = tqdm(total=len(rows) * args.epochs, desc="DPO samples")
    global_step = 0
    running = []

    for _epoch in range(args.epochs):
        for i, row in enumerate(rows):
            chosen_inputs, chosen_prompt_len = encode(
                processor, row["question"], row["image"], row["chosen"], args.max_len, device
            )
            rejected_inputs, rejected_prompt_len = encode(
                processor, row["question"], row["image"], row["rejected"], args.max_len, device
            )

            pi_chosen = sequence_logprob(policy, chosen_inputs, chosen_prompt_len)
            pi_rejected = sequence_logprob(policy, rejected_inputs, rejected_prompt_len)
            with torch.no_grad():
                ref_chosen = sequence_logprob(ref, chosen_inputs, chosen_prompt_len)
                ref_rejected = sequence_logprob(ref, rejected_inputs, rejected_prompt_len)

            logits = args.beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
            loss = -F.logsigmoid(logits) / args.grad_accum
            loss.backward()
            running.append(float(loss.item() * args.grad_accum))

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(rows):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    pbar.set_postfix(loss=sum(running[-50:]) / max(len(running[-50:]), 1), step=global_step)
            pbar.update(1)

    pbar.close()
    policy.save_pretrained(out_dir / "checkpoint")
    processor.save_pretrained(out_dir / "checkpoint")
    (out_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "n": len(rows),
                "epochs": args.epochs,
                "lr": args.lr,
                "beta": args.beta,
                "grad_accum": args.grad_accum,
                "total_steps": total_steps,
                "loss_tail_mean": sum(running[-50:]) / max(len(running[-50:]), 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
