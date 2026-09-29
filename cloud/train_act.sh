#!/usr/bin/env bash
# ACT baseline. ~52M params; fits any 16 GB GPU at batch 32.
#   cloud/train_act.sh data/lerobot/panda_stack outputs/train/act 40000
set -euo pipefail
DATA=${1:-data/lerobot/panda_stack}
OUT=${2:-outputs/train/act}
STEPS=${3:-40000}

cd "$(dirname "$0")/.."
pixi run -e train lerobot-train \
  --dataset.repo_id=local/panda_stack --dataset.root="$DATA" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=50 --policy.n_action_steps=50 \
  --output_dir="$OUT" --job_name=act \
  --batch_size=32 --steps="$STEPS" --log_freq=100 --save_freq=5000 --num_workers=8 \
  --wandb.enable=false
