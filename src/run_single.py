"""ConflictQA single-evidence experiments, merged from code/vlm/text.py and image.py.

Preserves both original prompts (including whitespace), counter_memory evidence,
index-based image names, decoding settings, and word2number normalization.
The model is loaded once; each condition starts from the same input records and
writes its own text.json or image.json, as in the original scripts.

Example:
    python src/run_single.py --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --input_path results/qwen7b/internal.json --output_dir results/qwen7b/single \
        --image_root data/ConflictQA/image

Provide --input_path, --output_dir and --image_root explicitly.
Inputs from run_internal.py use original_idx for image/gpt4/counter/counter_{original_idx}.jpg.
Legacy inputs without original_idx use their row index, as in the original scripts.
The resolved internal answer is preserved in each record.
The original counter_memory evidence assignment is preserved; this runner does
not construct new conflict/alignment partitions from the internal answers.
"""

import argparse
import copy
import json
import os
import random
from pathlib import Path


# These strings preserve the original prompt whitespace exactly.
TEXT_PROMPT = (
    "{external}\n"
    "        Query: {query} \n"
    "        \n"
    "        Answer using 1–5 words only. Do not use a full sentence."
)
IMAGE_PROMPT = (
    "Query: {query} \n"
    "        \n"
    "        Answer using 1–5 words only. Do not use a full sentence."
)


def build_messages(sample, index, modality, image_root):
    query = sample["question"]
    image_index = sample.get("original_idx", index)
    if modality == "text":
        prompt = TEXT_PROMPT.format(external=sample["counter_memory"], query=query)
        content = [{"type": "text", "text": prompt}]
    elif modality == "image":
        image_path = image_root / "gpt4" / "counter" / f"counter_{image_index}.jpg"
        prompt = IMAGE_PROMPT.format(query=query)
        content = [
            {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
            {"type": "text", "text": prompt},
        ]
    else:
        raise ValueError(f"Unknown modality: {modality}")
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
    parser.add_argument("--model_id")
    parser.add_argument("--model_name", help="Optional run label; not used to construct paths")
    parser.add_argument("--modality", choices=("text", "image", "both"), default="both")
    parser.add_argument("--input_path", type=Path, required=True, help="Prepared input JSON")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for result JSON files")
    parser.add_argument("--image_root", type=Path, required=True, help="Image directory containing gpt4/")
    parser.add_argument("--dry_run", action="store_true", help="Save messages without model inference")
    args = parser.parse_args()
    if not args.dry_run and not args.model_id:
        parser.error("--model_id is required for inference")
    args.image_root = args.image_root.resolve()
    modalities = ("text", "image") if args.modality == "both" else (args.modality,)
    for modality in modalities:
        filename = f"{modality}_messages.json" if args.dry_run else f"{modality}.json"
        if (args.output_dir / filename).resolve() == args.input_path.resolve():
            parser.error("Output must not overwrite the input file")
    return args


def main():
    args = parse_args()
    random.seed(42)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

    with args.input_path.open(encoding="utf-8") as handle:
        original_data = json.load(handle)
    if not isinstance(original_data, list):
        raise ValueError("Input must be a JSON list")
    # Also permit internal_all.json, excluding explicitly rejected records.
    # Attach legacy indices BEFORE filtering so image alignment cannot shift.
    for index, sample in enumerate(original_data):
        sample.setdefault("original_idx", index)
        status = sample.get("internal_knowledge_status")
        if status not in (None, "keep", "discard"):
            raise ValueError(f"Row {index}: unresolved status {status!r}; rerun run_internal.py")
    original_data = [s for s in original_data if s.get("internal_knowledge_status") != "discard"]
    print(f"Loaded {len(original_data)} retained samples")
    modalities = ("text", "image") if args.modality == "both" else (args.modality,)

    # Fail before GPU loading if prepared records or rendered images are missing.
    for index, sample in enumerate(original_data):
        for modality in modalities:
            build_messages(sample, index, modality, args.image_root)
        if "image" in modalities:
            image_path = args.image_root / "gpt4" / "counter" / f"counter_{sample['original_idx']}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing rendered evidence: {image_path}")

    if not args.dry_run and original_data:
        from tqdm import tqdm
        from vllm import LLM, SamplingParams

        model = LLM(
            model=args.model_id,
            tensor_parallel_size=1,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 10},
            allowed_local_media_path=str(args.image_root),
        )
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=20)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for modality in modalities:
        data = copy.deepcopy(original_data)
        rows = enumerate(data) if args.dry_run or not data else enumerate(tqdm(data, desc=modality))
        for index, sample in rows:
            messages = build_messages(sample, index, modality, args.image_root)
            if args.dry_run:
                sample[f"messages-{modality}"] = messages
                continue
            answers = []
            for _ in range(3):
                result = model.chat(messages, sampling_params=sampling_params)
                answers.append(normalize_answer(result[0].outputs[0].text))
            sample[f"output-{modality}"] = answers

        filename = f"{modality}_messages.json" if args.dry_run else f"{modality}.json"
        output_path = args.output_dir / filename
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=4)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
