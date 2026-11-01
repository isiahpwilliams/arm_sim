#!/usr/bin/env bash
# Closed-loop evaluation of a checkpoint in the MuJoCo sim. Headless, but EGL is
# mandatory here: without MUJOCO_GL=egl MuJoCo dies with "an OpenGL platform
# library has not been loaded".
#   cloud/eval.sh outputs/train/smolvla/checkpoints/last/pretrained_model 50
#   cloud/eval.sh outputs/train/act_gb/checkpoints/040000/pretrained_model 50 --pair cube_green,cube_blue
set -euo pipefail
CKPT=${1:?checkpoint dir}
SEEDS=${2:-50}
cd "$(dirname "$0")/.."

# $CKPT is .../train/<run>/checkpoints/<step>/pretrained_model, so the step is one
# dirname up and the run is three. Naming the video after two was always
# "checkpoints", so every eval silently overwrote outputs/eval_checkpoints.mp4.
STEP=$(basename "$(dirname "$CKPT")")
RUN=$(basename "$(dirname "$(dirname "$(dirname "$CKPT")")")")

MUJOCO_GL=egl pixi run -e train python sim/eval_policy.py "$CKPT" --seeds "$SEEDS" --device cuda \
  --video "outputs/eval_${RUN}_${STEP}.mp4" "${@:3}"
