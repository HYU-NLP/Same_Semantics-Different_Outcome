"""Run safety evaluations with explicit local paths.

Example:
    python src/run_safety.py --model_id MODEL --model_name MODEL_LABEL \
        --data_name CATEGORY --num RUN_ID \
        --data_dir data/MM-SafetyBench/data/processed_questions \
        --image_root data/MM-SafetyBench/image --output_dir results/safety
Output: OUTPUT_DIR/RUN_ID.json. Image paths retain their original layout.
"""
import os
import json


import argparse
from pathlib import Path

import random
random.seed(42)




def iter_local_image_paths(messages):
    for message in messages:
        for content in message.get("content", []):
            image_url = content.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url", "")
            elif isinstance(image_url, str):
                url = image_url
            else:
                continue

            if url.startswith("file://"):
                yield url[len("file://"):]


def chat_or_skip(mllm, messages, sampling_params, modality, index):
    missing_paths = [path for path in iter_local_image_paths(messages) if not os.path.exists(path)]
    if missing_paths:
        print(f"\n[SKIP] {modality} index={index}: missing image(s): {', '.join(missing_paths)}")
        return None

    try:
        return mllm.chat(messages, sampling_params=sampling_params)[0].outputs[0].text.strip().strip(".")
    except ValueError as e:
        message = str(e)
        if "decoder prompt" in message and "maximum model length" in message:
            print(f"\n[SKIP] {modality} index={index}: {message}")
            return None
        raise
    except FileNotFoundError as e:
        print(f"\n[SKIP] {modality} index={index}: {e}")
        return None


def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--num", type=str, required=True)
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing CATEGORY.json")
    parser.add_argument("--image_root", type=Path, required=True, help="Root containing category image directories")
    parser.add_argument("--output_dir", type=Path, required=True, help="Results directory")
    args = parser.parse_args()
    args.image_root = args.image_root.resolve()

    from vllm import LLM, SamplingParams
    from tqdm import tqdm

    # 모델 로드
    mllm = LLM(
        model=args.model_id,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 10},  # 최대 이미지 개수
        allowed_local_media_path=str(args.image_root),
    )
    

    path = args.data_dir / f"{args.data_name}.json"

    with open(path, "r", encoding="utf-8") as f:
        processed = json.load(f)

    processed = list(processed.values())

    processed[0]

    ##############################################################################################################################
    
    modality = 'question'
    for i, d in enumerate(tqdm(processed)):
        
        question = d["Question"]

        messages = [
            {"role": "user", "content": [{"type": "text", "text": question}]}
        ]
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

        output = chat_or_skip(mllm, messages, sampling_params, modality, i)
        d[f'{modality}'] = output
        print('\n1')
    
    modality = 'changed_question'
    for i, d in enumerate(tqdm(processed)):
        
        question = d["Question"]

        messages = [
            {"role": "user", "content": [{"type": "text", "text": question}]}
        ]
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

        output = chat_or_skip(mllm, messages, sampling_params, modality, i)
        d[f'{modality}'] = output
        print('\n2')

    modality = 'SD'
    for i, d in enumerate(tqdm(processed)):
        
        question = d["Rephrased Question(SD)"]
        image_path = f"{args.image_root}/{args.data_name}/SD/{i}.jpg"

        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{image_path}"}}]},
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

        output = chat_or_skip(mllm, messages, sampling_params, modality, i)
        d[f'{modality}'] = output
        print('\n3')

    modality = 'TYPO'
    for i, d in enumerate(tqdm(processed)):
        
        question = d["Rephrased Question"]
        image_path = f"{args.image_root}/{args.data_name}/TYPO/{i}.jpg"

        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{image_path}"}}]},
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

        output = chat_or_skip(mllm, messages, sampling_params, modality, i)
        d[f'{modality}'] = output
        print('\n4')

    modality = 'SD_TYPO'
    for i, d in enumerate(tqdm(processed)):
        
        question = d["Rephrased Question(SD)"]
        image_path = f"{args.image_root}/{args.data_name}/SD_TYPO/{i}.jpg"

        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{image_path}"}}]},
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

        output = chat_or_skip(mllm, messages, sampling_params, modality, i)
        d[f'{modality}'] = output
        print('\n5')

    modality = 'My_TYPO'
    for i, d in enumerate(tqdm(processed)):
        
        image_path = f"{args.image_root}/{args.data_name}/TYPO/{i}.jpg"
        # question_path = f"{args.image_root}/{args.data_name}/rephrased_question/{i}.jpg"
        question_path = f"{args.image_root}/{args.data_name}/rephrased_question_my/{i}.jpg"

        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{image_path}"}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{question_path}"}}]},
        ]
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

        output = chat_or_skip(mllm, messages, sampling_params, modality, i)
        d[f'{modality}'] = output
        print('\n6')

    modality = 'My_SD'
    for i, d in enumerate(tqdm(processed)):
        
        image_path = f"{args.image_root}/{args.data_name}/SD/{i}.jpg"
        # question_path = f"{args.image_root}/{args.data_name}/rephrased_question_sd/{i}.jpg"
        question_path = f"{args.image_root}/{args.data_name}/rephrased_question_sd_my/{i}.jpg"

        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{image_path}"}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{question_path}"}}]},
        ]
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

        output = chat_or_skip(mllm, messages, sampling_params, modality, i)
        d[f'{modality}'] = output
        print('\n7')

    modality = 'My_SD_TYPO'
    for i, d in enumerate(tqdm(processed)):
        
        image_path = f"{args.image_root}/{args.data_name}/SD_TYPO/{i}.jpg"
        # question_path = f"{args.image_root}/{args.data_name}/rephrased_question/{i}.jpg"
        question_path = f"{args.image_root}/{args.data_name}/rephrased_question_my/{i}.jpg"

        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{image_path}"}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{question_path}"}}]},
        ]
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

        output = chat_or_skip(mllm, messages, sampling_params, modality, i)
        d[f'{modality}'] = output
        print('\n8')
        
    modality = 'My_question'
    for i, d in enumerate(tqdm(processed)):
        
        question_path = f"{args.image_root}/{args.data_name}/question/{i}.jpg"

        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{question_path}"}}]},
        ]
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

        output = chat_or_skip(mllm, messages, sampling_params, modality, i)
        d[f'{modality}'] = output
        print('\n9')
    
    ##############################################################################################################################

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/{args.num}.json", "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=4)

    print('✅ Done')

if __name__ == "__main__":
    main()