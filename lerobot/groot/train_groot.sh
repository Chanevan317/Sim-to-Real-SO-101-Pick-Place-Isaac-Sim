#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on the SO-101 sim dataset with LeRobot.
#
# Run in the lerobot env after `pip install -e ".[groot]"` in the lerobot repo
# and after running fix_dataset_metadata.py on the dataset.
#
# Settings can be overridden through env vars, extra args are forwarded to
# lerobot-train:
#   GPU=1 NUM_EPISODES=100 BATCH_SIZE=4 ./train_groot.sh
#   NUM_GPUS=2 STEPS=60000 ./train_groot.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load HF_TOKEN (needed for the gated nvidia/Cosmos-Reason2-2B repo) from a
# gitignored .env next to this script, if present. See .env.example.
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/../lerobot_dataset}"
# Dataset name: only a label (LeRobot does not store it in the dataset), so it follows the folder name.
REPO_ID="${REPO_ID:-local/$(basename "$DATASET_ROOT")}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/../train/outputs/so101_groot}"
# Batch size 8 peaks at ~13.6 GB with LOW_MEMORY=1 (tested on an RTX 4080 SUPER).
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
# Train on the first N episodes only (episode_index 0..N-1). Unset = all episodes.
NUM_EPISODES="${NUM_EPISODES:-}"
# How many GPUs to train on. Each GPU holds a full model copy and gets
# BATCH_SIZE samples, so 2 GPUs process 2 x BATCH_SIZE samples per step.
NUM_GPUS="${NUM_GPUS:-1}"
# Which GPU(s) to use, e.g. GPU=1, or GPU=0,1 for two GPUs.
if [[ "$NUM_GPUS" -gt 1 ]]; then
  GPU="${GPU:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
else
  GPU="${GPU:-0}"
fi
# LOW_MEMORY=1 fits a 16 GB GPU: weights stay in bf16 (--policy.dtype=bfloat16)
# instead of fp32, and the 1.1B-parameter diffusion transformer (DiT) is
# frozen, so only the ~0.5B state/action encoders and VL projector train.
# LOW_MEMORY=0 is NVIDIA's full recipe (fp32 weights, DiT trained), which
# needs roughly 40 GB or more.
LOW_MEMORY="${LOW_MEMORY:-1}"
# Dataloader worker processes. The dataset's camera frames are AV1 video,
# decoded on CPU (pyav), so this needs to scale with batch size to keep the
# GPU fed; raise it on machines with more cores.
NUM_WORKERS="${NUM_WORKERS:-4}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MEMORY_ARGS=()
if [[ "$LOW_MEMORY" == "1" ]]; then
  # `model_params_fp32` is dead in current LeRobot (silently discarded in
  # configuration_groot.py's __post_init__); `dtype` is the flag that actually
  # controls whether the model gets cast to fp32.
  MEMORY_ARGS=(--policy.dtype=bfloat16 --policy.tune_diffusion_model=false)
fi

EPISODE_ARGS=()
if [[ -n "$NUM_EPISODES" ]]; then
  EPISODE_ARGS=(--dataset.episodes="[$(seq -s, 0 $((NUM_EPISODES - 1)))]")
fi

LAUNCHER=(lerobot-train)
if [[ "$NUM_GPUS" -gt 1 ]]; then
  LAUNCHER=(accelerate launch --multi_gpu --num_processes="$NUM_GPUS" "$(which lerobot-train)")
fi

"${LAUNCHER[@]}" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.image_transforms.enable=true \
  "${EPISODE_ARGS[@]}" \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.embodiment_tag=new_embodiment \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.use_bf16=true \
  "${MEMORY_ARGS[@]}" \
  --policy.push_to_hub=false \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --use_policy_training_preset=true \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --log_freq=10 \
  --num_workers="$NUM_WORKERS" \
  --output_dir="$OUTPUT_DIR" \
  --job_name=so101_groot \
  --wandb.enable=false \
  "$@"
