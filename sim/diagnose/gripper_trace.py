"""Does re-planning every step stop the gripper from ever committing to close?

The gripper command is effectively binary (255 open, 0 closed). Temporal ensembling
blends ~50 overlapping chunk predictions; if some say open and some say close, the
average is a half-closed gripper that grasps nothing. Trace the command either way.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "sim")
from arm_env import PandaCubeEnv            # noqa: E402
from tasks import sample_task               # noqa: E402
from eval_policy import load_policy, make_batch  # noqa: E402

CKPT = Path("outputs/train/act_gb/checkpoints/last/pretrained_model")
SEEDS = [5, 17, 46]          # three the baseline solved
CAMS = ["front", "wrist"]


def trace(ensemble):
    policy, pre, post = load_policy(CKPT, "cuda", temporal_ensemble=ensemble)
    env = PandaCubeEnv()
    for seed in SEEDS:
        obs = env.reset(seed=seed)
        src, dst, instruction = sample_task(np.random.default_rng(seed))
        policy.reset()
        grip = []
        for _ in range(200):
            frames = {c: env.render(c) for c in CAMS}
            with torch.inference_mode():
                a = post(policy.select_action(pre(make_batch(obs, frames, instruction, src, dst))))[0].cpu().numpy()
            grip.append(float(a[7]))
            obs = env.step(a)
        g = np.array(grip)
        below = np.flatnonzero(g < 128.0)
        print(f"  seed {seed:3d}  min={g.min():6.1f}  first<128: "
              f"{below[0] if below.size else 'never':>5}  steps<128: {below.size:3d}  "
              f"steps<30: {int((g < 30).sum()):3d}")


print(f"=== baseline (n_action_steps=50, no ensembling) ===")
trace(None)
print(f"=== temporal ensembling 0.01 (replan every step) ===")
trace(0.01)
