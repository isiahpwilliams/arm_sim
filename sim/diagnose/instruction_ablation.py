"""Does the policy's output depend on what it was TOLD at all?

The mirror of image_ablation.py. Hold the images and the proprioceptive state fixed
and vary ONLY the instruction:
  true       -- the instruction that actually belongs with this frame
  paraphrase -- the same task in different words (one of the other four templates)
  swap src   -- names a DIFFERENT source cube: a genuinely different task
  reversed   -- source and destination exchanged: also a different task
  blank      -- empty string, no instruction at all

Two readings matter, and they are different questions:
  * if "swap src" barely moves the prediction, the policy is not using the
    instruction to choose a cube -- its cube selection was always chance.
  * if "paraphrase" moves it about as much as "swap src", the policy is reacting to
    token noise rather than meaning, so it never learned the colour binding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "sim")
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from tasks import COLOUR, TEMPLATES                          # noqa: E402
from eval_policy import load_policy                          # noqa: E402

IMG_KEYS = ("observation.images.front", "observation.images.wrist")
COLOURS = list(COLOUR.values())


def parse(task: str) -> tuple[str, str]:
    """(src colour, dst colour) -- every template puts src before dst."""
    found = sorted(((task.index(c), c) for c in COLOURS if c in task))
    assert len(found) == 2, task
    return found[0][1], found[1][1]


def variants(task: str) -> dict[str, str]:
    src, dst = parse(task)
    other = next(c for c in COLOURS if c not in (src, dst))
    base = next(t for t in TEMPLATES if t.format(src=src, dst=dst) == task)
    para = next(t for t in TEMPLATES if t is not base)
    return {
        "paraphrase": para.format(src=src, dst=dst),
        "swap src": base.format(src=other, dst=dst),
        "reversed": base.format(src=dst, dst=src),
        "blank": "",
    }


def predict(policy, pre, post, state, task, imgs):
    batch = {"observation.state": state[None], "task": [task]}
    for k, v in imgs.items():
        batch[k] = v[None]
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(pre(batch))
    return post(chunk[:, 0])[0].cpu().numpy()


def main() -> None:
    ckpt = Path(sys.argv[1])
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/lerobot/panda_stack")
    repo = sys.argv[3] if len(sys.argv) > 3 else "local/panda_stack"
    policy, pre, post = load_policy(ckpt, "cuda")
    eps = [0, 5, 11, 23]
    ds = LeRobotDataset(repo, root=root, video_backend="pyav", episodes=eps)
    frames = [ds[i] for i in range(0, len(ds), 15)]
    print(f"checkpoint: {ckpt}\ndataset: {root}\n{len(frames)} frames from episodes {eps}\n")

    shifts: dict[str, list] = {}
    for s in frames:
        imgs = {k: s[k] for k in IMG_KEYS}
        state, task = s["observation.state"], s["task"]
        p_true = predict(policy, pre, post, state, task, imgs)
        for name, alt in variants(task).items():
            p_alt = predict(policy, pre, post, state, task=alt, imgs=imgs)
            shifts.setdefault(name, []).append(np.abs(p_true - p_alt)[:7])

    acts = np.array([f["action"].numpy()[:7] for f in frames])
    deg = np.degrees
    print("how far the PREDICTION moves when only the INSTRUCTION changes (deg):")
    for name in ("paraphrase", "swap src", "reversed", "blank"):
        print(f"  true -> {name:11s}: {deg(np.mean(shifts[name])):6.2f}")
    print("\nscale reference:")
    print(f"  std of recorded actions across these frames : {deg(acts.std(0).mean()):6.2f} deg")
    print(f"  mean |action - mean action|                 : {deg(np.abs(acts - acts.mean(0)).mean()):6.2f} deg")
    print("=== INSTRUCTION ABLATION DONE ===")


if __name__ == "__main__":
    main()
