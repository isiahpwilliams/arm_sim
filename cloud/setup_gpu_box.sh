#!/usr/bin/env bash
# One-time setup on a rented Linux x86_64 box with an NVIDIA GPU (Ubuntu 22.04+).
# Installs pixi and the `train` environment (LeRobot + CUDA torch), nothing else.
set -euo pipefail

if ! command -v pixi >/dev/null; then
  curl -fsSL https://pixi.sh/install.sh | sh
  export PATH="$HOME/.pixi/bin:$PATH"
fi

cd "$(dirname "$0")/.."
pixi install -e train
pixi run -e train python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
