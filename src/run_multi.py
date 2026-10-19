"""Merged ConflictQA multi-evidence experiments from code/vlm/0506* and 0507*.

Select an original filename (without .py) using --variant. Original prompts,
evidence assignments, original_idx image paths, output keys, and model
settings are preserved. Output filenames omit the original date prefix. Each invocation runs both orders for ONE variant.
The *2 variants intentionally share output names with their original versions;
use different --output_dir values when retaining both runs.

Example:
    python src/run_multi.py --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --variant 0506_multi_image=true_counter2 \
        --input_path data/ConflictQA/internal_5_mcq_all_none_of_them_counter.json \
        --output_dir results/qwen7b/multi --image_root data/ConflictQA/image

Use --dry_run to inspect messages without loading a model.
"""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random

QUESTION_PROMPT = (
    "Query: {query} \n"
    "        \n"
    "        Answer using 1–5 words only. Do not use a full sentence."
)
EVIDENCE_PROMPT = "{evidence}\n        " + QUESTION_PROMPT


@dataclass(frozen=True)
class Experiment:
    subset: str
    mode: str
    first_source: str
    parallel: int
    max_model_len: int
    output_name: str

    @property
    def orders(self):
        return {"mixed": ("itq", "tiq"), "image": ("ii1", "ii2"),
                "text": ("tt1", "tt2")}[self.mode]


# mixed: first_source is the IMAGE source, fixed across itq/tiq.
# same modality: first_source is first in order 1 and second in order 2.
EXPERIMENTS = {
    "0506_multi_image=true_counter": Experiment("counter", "mixed", "counter", 2, 4096, "image=true_counter.json"),
    "0506_multi_image=true_counter2": Experiment("counter", "mixed", "counter", 1, 8192, "image=true_counter.json"),
    "0506_multi_image=true_para": Experiment("para", "mixed", "para", 2, 4096, "image=true_para.json"),
    "0506_multi_image=true_para2": Experiment("para", "mixed", "para", 1, 8192, "image=true_para.json"),
    "0506_multi_text=true_counter": Experiment("counter", "mixed", "para", 2, 4096, "text=true_counter.json"),
    "0506_multi_text=true_counter2": Experiment("counter", "mixed", "para", 1, 8192, "text=true_counter.json"),
    "0506_multi_text=true_para": Experiment("para", "mixed", "counter", 1, 4096, "text=true_para.json"),
    "0506_multi_text=true_para2": Experiment("para", "mixed", "counter", 1, 8192, "text=true_para.json"),
    "0507_multi_image_counter": Experiment("counter", "image", "counter", 1, 4096, "multi_image_counter.json"),
    "0507_multi_image_para": Experiment("para", "image", "para", 1, 4096, "multi_image_para.json"),
    "0507_multi_text_counter": Experiment("counter", "text", "counter", 1, 4096, "multi_text_counter.json"),
    "0507_multi_text_para": Experiment("para", "text", "para", 1, 4096, "multi_text_para.json"),
}


def build_messages(sample, experiment, order, image_root):
    if order not in experiment.orders:
        raise ValueError(f"Invalid order {order} for {experiment.mode}")

    def text(source):
        key = "parametric_memory" if source == "para" else "counter_memory"
        return sample[key]

    def image(source):
        path = image_root / "gpt4" / source / f"{source}_{sample['original_idx']}.jpg"
        return {"type": "image_url", "image_url": {"url": f"file://{path}"}}

    first = experiment.first_source
    second = "counter" if first == "para" else "para"
    question = {"type": "text", "text": QUESTION_PROMPT.format(query=sample["question"])}

    if experiment.mode == "mixed":
        if order == "itq":
            content = [image(first), {"type": "text", "text": EVIDENCE_PROMPT.format(
                evidence=text(second), query=sample["question"])}]
        else:
            content = [{"type": "text", "text": text(second)}, image(first), question]
    else:
        if order == experiment.orders[1]:
            first, second = second, first
        if experiment.mode == "image":
            content = [image(first), image(second), question]
        else:
            content = [{"type": "text", "text": text(first)},
                       {"type": "text", "text": text(second)}, question]
    return [{"role": "user", "content": content}]


def normalize_answer(text):
    from word2number import w2n

    answer = text.strip().strip(".")
    try:
        answer = str(w2n.word_to_num(answer))
    except Exception:
        pass
    return answer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=EXPERIMENTS, required=True)
    parser.add_argument("--model_id")
    parser.add_argument("--model_name", help="Optional run label; not used to construct paths")
    parser.add_argument("--input_path", type=Path, required=True, help="Prepared input JSON")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for result JSON files")
    parser.add_argument("--image_root", type=Path, required=True, help="Image directory containing gpt4/")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.model_id:
        parser.error("--model_id is required for inference")
    experiment = EXPERIMENTS[args.variant]
    args.image_root = args.image_root.resolve()
    filename = experiment.output_name
    if args.dry_run:
        filename = f"{args.variant.split('_', 1)[1]}_messages.json"
    args.output_path = args.output_dir / filename
    if args.output_path.resolve() == args.input_path.resolve():
        parser.error("Output must not overwrite input")
    return args


def main():
    args = parse_args()
    experiment = EXPERIMENTS[args.variant]
    random.seed(42)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
    with args.input_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if not args.dry_run:
        from tqdm import tqdm
        from vllm import LLM, SamplingParams

        model = LLM(
            model=args.model_id,
            tensor_parallel_size=experiment.parallel,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=experiment.max_model_len,
            limit_mm_per_prompt={"image": 10},
            allowed_local_media_path=str(args.image_root),
        )
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=20)

    for order in experiment.orders:
        rows = data if args.dry_run else tqdm(data, desc=order)
        for sample in rows:
            messages = build_messages(sample, experiment, order, args.image_root)
            if args.dry_run:
                sample[f"messages-{order}"] = messages
                continue
            answers = []
            for _ in range(3):
                result = model.chat(messages, sampling_params=sampling_params)
                answers.append(normalize_answer(result[0].outputs[0].text))
            sample[f"output-{order}"] = answers

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=4)
    print(f"Saved {len(data)} samples to {args.output_path}")


if __name__ == "__main__":
    main()
