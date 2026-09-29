#!/usr/bin/env bash
# Closed-loop evaluation of a checkpoint in the MuJoCo sim (headless, CPU rendering is fine).
#   cloud/eval.sh outputs/train/smolvla/checkpoints/last/pretrained_model 50
set -euo pipefail
CKPT=${1:?checkpoint dir}
SEEDS=${2:-50}
cd "$(dirname "$0")/.."
MUJOCO_GL=egl pixi run -e train python sim/eval_policy.py "$CKPT" --seeds "$SEEDS" --device cuda \
  --video "outputs/eval_$(basename "$(dirname "$(dirname "$CKPT")")").mp4"
