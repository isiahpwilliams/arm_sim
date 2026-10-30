#!/usr/bin/env bash
# SmolVLA fine-tune from lerobot/smolvla_base (pretrained VLM + action expert).
# Do not use --policy.type=smolvla here: that trains the backbone from scratch.
# The base expects cameras named camera1..3; rename_map maps ours onto the first two.
# Batch 32 fits a 24 GB GPU; use 16 on an L4.
#   cloud/train_smolvla.sh data/lerobot/panda_stack outputs/train/smolvla 20000
set -euo pipefail
DATA=${1:-data/lerobot/panda_stack}
OUT=${2:-outputs/train/smolvla}
STEPS=${3:-20000}

cd "$(dirname "$0")/.."
pixi run -e train lerobot-train \
  --dataset.repo_id=local/panda_stack --dataset.root="$DATA" \
  --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --output_dir="$OUT" --job_name=smolvla \
  --batch_size=32 --steps="$STEPS" --log_freq=100 --save_freq=2500 --num_workers=8 \
  --wandb.enable=false
