"""Extract ConflictQA internal knowledge with five closed-book MCQ responses.

Protocol (Section 3.1): ask the closed-book MCQ five times and retain only
unanimous valid selections. Use the chosen answer as internal_answer. For
unanimous None-of-the-above selections, request an open-ended answer (Figure 9)
and use it to replace internal_answer.

The paper does not specify the open-ended repeat count or an aggregation rule.
This implementation makes one generation call, with no additional voting or
consistency filter. This is a minimal implementation choice, not a claim that
the paper explicitly prescribes one call. Empty generations raise an error.

Choice construction, parsing and MCQ decoding follow code/mcq/5_mcq.py.
The Figure 9 prompt also appears in code/mcq/5_internal=false_o.py; vLLM
open-ended decoding settings follow code/vlm/0508_internal.py.

Example:
    python src/run_internal.py --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --input_path /path/to/conflictqa_with_short_answers.json \
        --output_dir results/qwen7b/internal

Outputs:
- original_internal_5_mcq.json: all MCQ records, saved before fallback inference.
- internal_all.json: all records, with keep/discard status and raw answers.
- internal.json: retained records only. original_idx preserves image alignment.
These files do not construct the downstream para/counter evidence partitions.
run_single.py reads this internal.json directly and uses original_idx for images.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random

CHOICE_LETTERS = ["A", "B", "C", "D"]
NONE_OF_THEM = "None of them"

def first_ground_truth(sample):
    ground_truth = sample.get("ground_truth", "")
    if isinstance(ground_truth, list):
        return ground_truth[0] if ground_truth else ""
    return ground_truth

def normalize_letter(output):
    output = str(output).strip().upper()
    for letter in CHOICE_LETTERS:
        if output == letter or output.startswith(f"{letter}.") or output.startswith(f"{letter})"):
            return letter
    return output[:1] if output[:1] in CHOICE_LETTERS else ""

def build_options(sample):
    options = [
        {"tag": "ground_truth", "text": first_ground_truth(sample)},
        {"tag": "memory_short_answer", "text": sample.get("memory_short_answer", "")},
        {"tag": "counter_short_answer", "text": sample.get("counter_short_answer", "")},
        {"tag": "none_of_them", "text": NONE_OF_THEM},
    ]

    random.shuffle(options)
    return [
        {"letter": letter, "tag": option["tag"], "text": option["text"]}
        for letter, option in zip(CHOICE_LETTERS, options)
    ]


def validate_samples(data):
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON list")
    for index, sample in enumerate(data):
        required = ("question", "ground_truth", "memory_short_answer", "counter_short_answer")
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"Row {index}: missing {missing}; supply data with short answers")
        values = [sample['question'], first_ground_truth(sample),
                  sample['memory_short_answer'], sample['counter_short_answer']]
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"Row {index}: question and answer candidates must be nonempty strings")


def collect_mcq(sample, model, sampling_params):
    """Original 5_mcq.py sample loop; fixed to five responses."""
    query = sample['question']
    shuffled_options = build_options(sample)
    choices_str = '\n'.join((f"{option['letter']}. {option['text']}" for option in shuffled_options))
    prompt = f'Answer the following multiple-choice question based on your own knowledge.\n\nQuestion: {query}\n\nChoices:\n{choices_str}\n\nInstructions:\n- Choose the option that best answers the question.\n- Choose "{NONE_OF_THEM}" only if none of the other choices is correct.\n- Answer with only one letter: A, B, C, or D.\n'
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
    raw_answers = []
    parsed_answers = []
    for _ in range(5):
        output = model.chat(messages, sampling_params=sampling_params)[0].outputs[0].text.strip().strip('.')
        raw_answers.append(output)
        parsed_answers.append(normalize_letter(output))
    tag_to_letter = {option['tag']: option['letter'] for option in shuffled_options}
    letter_to_tag = {option['letter']: option['tag'] for option in shuffled_options}
    letter_to_text = {option['letter']: option['text'] for option in shuffled_options}
    tagged_answers = [{'raw_answer': raw_answer, 'letter': letter, 'tag': letter_to_tag.get(letter, 'invalid'), 'choice_text': letter_to_text.get(letter, '')} for (raw_answer, letter) in zip(raw_answers, parsed_answers)]
    sample['internal_knowledge_5_mcq'] = raw_answers
    sample['internal_knowledge_5_mcq_tags'] = [letter_to_tag.get(letter, 'invalid') for letter in parsed_answers]
    sample['internal_knowledge_5_mcq_tagged'] = tagged_answers
    sample['internal_knowledge_5_mcq_choices'] = shuffled_options
    sample['ground_truth_choice'] = first_ground_truth(sample)
    sample['ground_truth_letter'] = tag_to_letter['ground_truth']
    sample['memory_short_answer_letter'] = tag_to_letter['memory_short_answer']
    sample['counter_short_answer_letter'] = tag_to_letter['counter_short_answer']
    sample['none_of_them_letter'] = tag_to_letter['none_of_them']


def finalize_sample(sample, model, sampling_params):
    """Resolve the MCQ answer, using Figure 9 only after unanimous None."""
    tags = sample["internal_knowledge_5_mcq_tags"]
    sample["internal_answer"] = None
    sample["internal_knowledge"] = []
    sample["internal_answer_source"] = None
    sample["internal_knowledge_status"] = "discard"
    sample.pop("internal_rejection_reason", None)
    sample.pop("internal_knowledge_5", None)
    if len(tags) != 5 or len(set(tags)) != 1 or tags[0] == "invalid":
        sample["internal_rejection_reason"] = "inconsistent_or_invalid_mcq"
        return

    selected_tag = tags[0]
    if selected_tag == "none_of_them":
        query = sample["question"]
        prompt = f"""Answer the following question based on your own knowledge.

Question: {query}

Answer in 1–5 words. Do not use a full sentence."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        # No repeat or aggregation rule is specified for this step in the paper.
        answer = (
            model.chat(messages, sampling_params=sampling_params)[0]
            .outputs[0].text.strip().strip(".")
        )
        sample["internal_knowledge"] = [answer]
        answer_source = "open_ended_after_none"
    else:
        choice = next(c for c in sample["internal_knowledge_5_mcq_choices"]
                      if c["tag"] == selected_tag)
        answer = choice["text"]
        sample["internal_knowledge"] = [answer]
        answer_source = selected_tag

    if not answer.strip():
        raise ValueError("Empty internal answer: generation failed; no additional filtering rule is applied")
    sample["internal_answer"] = answer
    sample["internal_answer_source"] = answer_source
    sample["internal_knowledge_status"] = "keep"


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_id', required=True)
    parser.add_argument('--input_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    for name in ('original_internal_5_mcq.json', 'internal_all.json', 'internal.json'):
        if args.input_path.resolve() == (args.output_dir / name).resolve():
            parser.error('Input must differ from all output paths')
    return args


def main():
    args = parse_args()
    data = json.loads(args.input_path.read_text(encoding='utf-8'))
    validate_samples(data)
    data = copy.deepcopy(data)
    for index, sample in enumerate(data):
        sample.setdefault('original_idx', index)
    random.seed(42)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'

    from tqdm import tqdm
    from vllm import LLM, SamplingParams

    model = LLM(model=args.model_id, tensor_parallel_size=1, dtype='bfloat16',
                gpu_memory_utilization=0.85, enforce_eager=True, trust_remote_code=True,
                max_model_len=4096, limit_mm_per_prompt={'image': 10})
    mcq_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=5)
    open_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=20)

    # Complete MCQ for the whole dataset before any fallback model calls.
    for sample in tqdm(data, desc='MCQ (five responses)'):
        collect_mcq(sample, model, mcq_params)
    save_json(args.output_dir / 'original_internal_5_mcq.json', data)

    for sample in tqdm(data, desc='Resolve internal answers'):
        finalize_sample(sample, model, open_params)
    save_json(args.output_dir / 'internal_all.json', data)
    retained = [sample for sample in data if sample['internal_knowledge_status'] == 'keep']
    save_json(args.output_dir / 'internal.json', retained)
    print(f'Retained {len(retained)} / {len(data)}; discarded {len(data) - len(retained)}')



if __name__ == '__main__':
    main()
