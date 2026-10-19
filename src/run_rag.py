"""Run the existing MS MARCO text experiment with explicit paths.

Example:
    python src/run_rag.py --model_id Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --image_root data/MS_MARCO/image --output_path results/rag/text.json
MS MARCO is loaded through datasets; its cache can be set with HF_DATASETS_CACHE.
The source experiment's sampling and active gold-passage-only prompt are unchanged.
"""
import argparse
from pathlib import Path
import os
import json
import unicodedata
import re

def compute_rouge(sample):
    import numpy as np
    from rouge_score import rouge_scorer
    rr = []
    rouge_types=["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_types)
    for i, d in enumerate(sample):
        answer = d['answers'][0].lower().rstrip('.')
        try:
            output = d["output"].lower().rstrip('.')
            rouge_score = scorer.score(answer, output)
            for k in rouge_score:
                rouge_score[k] = rouge_score[k].fmeasure
            rr.append(rouge_score['rougeL'])
        except:
            continue
    print(np.average(rr))

def filename(s: str) -> str:
    # 파일명에 쓰기 애매한 문자 제거/치환 (필요시 규칙 더 강화)
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\-.]", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()

    import multiprocessing as mp
    from datasets import load_dataset
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    mp.set_start_method("spawn", force=True)

    from vllm import LLM, SamplingParams
    import torch

    print("GPU count seen by torch:", torch.cuda.device_count())

    ds = load_dataset("microsoft/ms_marco", "v2.1")

    train_filtered = ds["train"].filter(lambda d: d["passages"]["is_selected"].count(1) == 1 and len(d["passages"]["is_selected"]) == 10)

    subset = train_filtered.shuffle(seed=1004).select(range(70))

    model_id = args.model_id

    mllm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        trust_remote_code=True,
        # max_model_len=32768, # qwen
        max_model_len=4096, # llava
        # distributed_executor_backend="ray",
        limit_mm_per_prompt={"image": 10},  # 최대 이미지 개수
        allowed_local_media_path=str(args.image_root.resolve()),
    )

    outs_text = []

    for i, d in enumerate(subset):
        try:
            i = str(i)
            query = d['query']
            query_type = d['query_type']

            g_idx = d["passages"]["is_selected"].index(1)
            g_passages = d['passages']['passage_text'][g_idx]
            g_url = d['passages']['url'][g_idx]
            answer = d['answers'][0]

            ################################################
            idxs_l = d['passages']['is_selected']
            not_gold_idxs = [i for i, v in enumerate(idxs_l) if v == 0]
            idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9 = not_gold_idxs
            pass1 = d['passages']['passage_text'][idx1]
            pass2 = d['passages']['passage_text'][idx2]
            pass3 = d['passages']['passage_text'][idx3]
            pass4 = d['passages']['passage_text'][idx4]
            pass5 = d['passages']['passage_text'][idx5]
            pass6 = d['passages']['passage_text'][idx6]
            pass7 = d['passages']['passage_text'][idx7]
            pass8 = d['passages']['passage_text'][idx8]
            pass9 = d['passages']['passage_text'][idx9]
        
            ################################################
            instruction = "Answer the question with the shortest possible phrase or sentence. Use ONLY the passages. Do not add any extra facts."

            prompt = f"{instruction}\n[Passages]: {g_passages}\n[Query]: {query}"
            # prompt = f"{instruction}\n[Passages]: {g_passages}\n{pass1}\n{pass2}\n{pass3}\n{pass4}\n{pass5}\n{pass6}\n{pass7}\n{pass8}\n{pass9}\n[Query]: {query}" # top
            # prompt = f"{instruction}\n[Passages]: {pass1}\n{pass2}\n{pass3}\n{g_passages}\n{pass4}\n{pass5}\n{pass6}\n{pass7}\n{pass8}\n{pass9}\n[Query]: {query}" # 3
            # prompt = f"{instruction}\n[Passages]: {pass1}\n{pass2}\n{pass3}\n{pass4}\n{g_passages}\n{pass5}\n{pass6}\n{pass7}\n{pass8}\n{pass9}\n[Query]: {query}" # 4
            # prompt = f"{instruction}\n[Passages]: {pass1}\n{pass2}\n{pass3}\n{pass4}\n{pass5}\n{g_passages}\n{pass6}\n{pass7}\n{pass8}\n{pass9}\n[Query]: {query}" # 5
            # prompt = f"{instruction}\n[Passages]: {pass1}\n{pass2}\n{pass3}\n{pass4}\n{pass5}\n{pass6}\n{g_passages}\n{pass7}\n{pass8}\n{pass9}\n[Query]: {query}" # 6
            # prompt = f"{instruction}\n[Passages]: {pass1}\n{pass2}\n{pass3}\n{pass4}\n{pass5}\n{pass6}\n{pass7}\n{pass8}\n{pass9}\n{g_passages}\n[Query]: {query}" # bottom
        
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt,}]}]
            sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

            output = mllm.chat(messages, sampling_params=sampling_params)[0].outputs[0].text
            outs_text.append(output)
            ################################################
    
        except: outs_text.append(None)

    subset_text = subset.add_column("output", outs_text)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(list(subset_text), ensure_ascii=False, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
