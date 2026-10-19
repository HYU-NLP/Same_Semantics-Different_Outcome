# Same Semantics, Different Outcome: On the Modality Robustness of Multimodal LLMs under Knowledge Conflict

*Accepted to Findings of EMNLP 2026.*

## Overview

The experiments use **ConflictQA** and **NQ-Swap** to compare textual and visual evidence. Single-evidence experiments compare a passage with its rendered image. Multi-evidence experiments vary the source-to-modality assignment and presentation order, including image–text–question (`itq`) and text–image–question (`tiq`). The repository also includes scripts for RAG, safety, and mitigation experiments.

## Repository Structure

```text
.
├── README.md
├── data/
│   └── conflictQA-popQA-gpt4-filtered.json
└── src/
    ├── prompts.py
    ├── run_internal.py
    ├── run_single.py
    ├── run_multi.py
    ├── run_rag.py
    ├── run_safety.py
    └── mitigation/
        ├── openrlhf/
        ├── scripts/mllm/
        └── requirements.txt
```

## Datasets

### ConflictQA

```python
import json
from pathlib import Path

data_path = Path("data/conflictQA-popQA-gpt4-filtered.json")
with data_path.open(encoding="utf-8") as handle:
    data = json.load(handle)

# Preserve the source index before filtering or reordering examples.
for index, sample in enumerate(data):
    sample["original_idx"] = index
```

| Field | Description |
|---|---|
| `question` | Question to answer. |
| `popularity` | Popularity metadata supplied with the source example. |
| `ground_truth` | Reference factual answer or answer aliases. |
| `memory_answer` | Source memory-side answer. |
| `parametric_memory` | Passage describing the source model's memory-side evidence. |
| `counter_answer` | Competing counter-memory answer. |
| `counter_memory` | Passage supporting the counter-memory answer. |
| `original_idx` | Row index added locally to preserve image correspondence. |


### NQ-Swap

```python
from datasets import load_dataset

dataset = load_dataset("pminervini/NQ-Swap")
data = list(dataset["dev"])

# Preserve image alignment before filtering or reordering examples.
for index, sample in enumerate(data):
    sample["original_idx"] = index
```


#### Data fields

| Field | Description |
|---|---|
| `question` | Question to answer. |
| `org_context` | Original passage. |
| `org_answer` | Original answer candidates, stored as a list. |
| `sub_context` | Passage with substituted answer entities. |
| `sub_answer` | Substituted answer candidates, stored as a list. |
| `original_idx` | Row index added locally to preserve image correspondence. |

#### Experiment preparation

The original NQ-Swap MCQ script uses three choices: the first `org_answer`, the first `sub_answer`, and `None of them`. The choice order is shuffled once per example and reused for five responses.

For text-as-image experiments, render the original and substituted passages separately while preserving their indices:

```text
image/
├── org/
│   └── org_{original_idx}.jpg
└── sub/
    └── sub_{original_idx}.jpg
```

Single-evidence comparisons present the same selected passage as text or an image. Multi-evidence comparisons supply the selected textual and visual sources in both orders: image–text–question (`itq`) and text–image–question (`tiq`). The source assignment depends on the experimental condition; loading the dataset alone does not construct all conflict settings.

## Running the ConflictQA Experiments

Run the commands below from the repository root. Paths under `prepared/`, `images/`, and `results/` are examples; replace them with your own paths. No machine-specific absolute paths are required.

### Environment

Inference uses Python, a CUDA-capable GPU, and a vLLM installation compatible with the selected model. The main runners also use `tqdm` and `word2number`; dataset loading uses `datasets`. Install PyTorch and vLLM for your CUDA environment, then install the supporting packages:

```bash
pip install tqdm word2number datasets
```

A locked inference environment is not included. Training dependencies are listed separately in [src/mitigation/requirements.txt](src/mitigation/requirements.txt); those requirements are not a shared environment specification for all inference models.

### 1. Prepare the inputs

Set `original_idx` to each source row's index before filtering or reordering. Render the passages into this image layout:

```text
images/
└── gpt4/
    ├── para/
    │   └── para_{original_idx}.jpg
    └── counter/
        └── counter_{original_idx}.jpg
```

### 2. Extract internal knowledge

```bash
python src/run_internal.py \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --input_path prepared/conflictqa_with_short_answers.json \
  --output_dir results/qwen7b/internal
```

One invocation collects **five MCQ responses per example**; If all five select `None of them`, the script asks an open-ended, closed-book question and uses that answer as `internal_answer`. 


### 3. Run single-evidence experiments

```bash
python src/run_single.py \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --input_path results/qwen7b/internal/internal.json \
  --image_root images \
  --modality both \
  --output_dir results/qwen7b/single
```

Each condition collects three responses per example and writes `text.json` and/or `image.json`.

### 4. Run multi-evidence experiments

```bash
python src/run_multi.py \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --variant multi_text=true_para \
  --input_path prepared/internal_para.json \
  --image_root images \
  --output_dir results/qwen7b/multi
```

Mixed-modality conditions run both `itq` and `tiq`. Image-only conditions run `ii1` and `ii2`; text-only conditions run `tt1` and `tt2`, reversing the evidence order. Each order collects three responses. 

## Additional Experiments

| Component | Entry point | Scope |
|---|---|---|
| RAG | [run_rag.py](src/run_rag.py) | Loads MS MARCO; the active configuration uses gold-passage text. |
| Safety | [run_safety.py](src/run_safety.py) | Generates responses for the configured textual and image conditions. |
| Conflict SFT | [train_sft.sh](src/mitigation/scripts/mllm/train_sft.sh) | Training script with configurable model, dataset, and output paths. |
| DPO preparation | [prepare_misleading_image_dpo_data.py](src/mitigation/scripts/mllm/prepare_misleading_image_dpo_data.py) | Prepares preference data, including NQ-Swap-oriented inputs. |
| DPO training | [train_qwen25vl_misleading_dpo_lora.py](src/mitigation/scripts/mllm/train_qwen25vl_misleading_dpo_lora.py) | LoRA-based preference training. |
| Steering | [evaluation_hf_conflictqa_multi_text_steer.py](src/mitigation/scripts/mllm/evaluation_hf_conflictqa_multi_text_steer.py) | ConflictQA inference with activation steering. |
