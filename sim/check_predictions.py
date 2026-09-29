"""Offline check: how far a checkpoint's predicted actions are from the recorded ones.

    pixi run -e train python sim/check_predictions.py outputs/train/act_single/checkpoints/004000/pretrained_model --episodes 2 9 19

Reports mean absolute error per action dimension in raw units (radians, gripper 0..255)
on frames the policy trained on. Separates "pipeline bug" (huge errors) from
"needs more training" (small but not small enough).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from eval_policy import load_policy

OBS_KEYS = ("observation.state", "observation.images.front", "observation.images.wrist", "task")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--root", type=Path, default=Path("data/lerobot/panda_stack"))
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--every", type=int, default=20, help="sample every N frames")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    policy, pre, post = load_policy(args.checkpoint, args.device)
    ds = LeRobotDataset("local/panda_stack", root=args.root, video_backend="pyav", episodes=args.episodes)
    errs = []
    for idx in range(0, len(ds), args.every):
        s = ds[idx]
        batch = {k: (v[None] if torch.is_tensor(v) else [v]) for k, v in s.items() if k in OBS_KEYS}
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(pre(batch))
        pred = post(chunk[:, 0])[0].cpu().numpy()
        errs.append(np.abs(pred - s["action"].numpy()))
    e = np.mean(errs, axis=0)
    print(f"{len(errs)} frames from episodes {args.episodes}")
    print("mean |error| joints (rad):", e[:7].round(3))
    print(f"mean |error| gripper (0-255): {e[7]:.1f}")
    print(f"joint mean: {e[:7].mean():.3f} rad = {np.degrees(e[:7].mean()):.1f} deg")


if __name__ == "__main__":
    main()
