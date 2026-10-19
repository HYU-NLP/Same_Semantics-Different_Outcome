#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-VL-7B-Instruct}"
SEED="${SEED:-123}"
EXP_NAME="${EXP_NAME:-Qwen2-ConflictQA_true_para}"
DATASET="${DATASET:-./data/MMKC/Training/data/conflictqa_true_para.jsonl}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
GPU="${GPU:-0}"
DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
OUTPUT_DIR="$OUTPUT_ROOT/$EXP_NAME"
mkdir -p "$OUTPUT_DIR"

export WANDB_PROJECT="${WANDB_PROJECT:-counter-bias-sft}"
export WANDB_NAME="${WANDB_NAME:-${EXP_NAME}_seed${SEED}}"
export WANDB_DIR="$OUTPUT_DIR/wandb"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_WATCH="${WANDB_WATCH:-false}"

echo "EXP_NAME=$EXP_NAME"
echo "MODEL_NAME=$MODEL_NAME"
echo "DATASET=$DATASET"
echo "MAX_EPOCHS=$MAX_EPOCHS"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "LORA_RANK=$LORA_RANK"
echo "USE_FLASH_ATTN=$USE_FLASH_ATTN"
echo "OUTPUT_DIR=$OUTPUT_DIR"

if [[ -n "$DEEPSPEED_INCLUDE" ]]; then
  DEEPSPEED_ENV=()
  DEEPSPEED_CMD=(deepspeed --include "$DEEPSPEED_INCLUDE")
else
  DEEPSPEED_ENV=(CUDA_VISIBLE_DEVICES="$GPU")
  DEEPSPEED_CMD=(deepspeed)
fi

TRAIN_ARGS=(
    --image_text_to_text \
    --max_len 4500 \
    --dataset "$DATASET" \
    --input_key question,image \
    --output_key answer \
    --apply_chat_template \
    --train_batch_size 8 \
    --micro_train_batch_size 1 \
    --max_samples "$MAX_SAMPLES" \
    --pretrain "$MODEL_NAME" \
    --save_path "$OUTPUT_DIR/checkpoint" \
    --seed "$SEED" \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 2 \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --max_epochs "$MAX_EPOCHS" \
    --bf16 \
    --learning_rate "$LEARNING_RATE" \
    --load_checkpoint \
    --gradient_checkpointing \
    --use_tensorboard "$OUTPUT_DIR/tensorboard" \
    --use_wandb "${WANDB_API_KEY:-}" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_NAME"
)

if [[ "$USE_FLASH_ATTN" == "1" ]]; then
  TRAIN_ARGS+=(--flash_attn)
fi

env "${DEEPSPEED_ENV[@]}" "${DEEPSPEED_CMD[@]}" --module openrlhf.cli.train_sft "${TRAIN_ARGS[@]}" \
    2>&1 | tee "$OUTPUT_DIR/training.log"
