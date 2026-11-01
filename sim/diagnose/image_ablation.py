"""Does the policy's output depend on what the cameras see at all?

Hold the proprioceptive state and the instruction fixed, and vary ONLY the images:
  true     -- the frame that actually belongs with this state
  swapped  -- the frame from a different episode (a different scene entirely)
  grey     -- a flat 0.5 image carrying no information

If the policy is reading the cameras, swapping the scene should move its prediction a
lot. If the prediction barely moves, the images are not driving the action and the
policy is running off proprioception plus the clock -- which no amount of extra
demonstration data will fix.

Scale reference: we also report the spread of the recorded actions themselves, so
"a small change" can be judged against how much the action varies in the first place.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "sim")
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from eval_policy import load_policy                          # noqa: E402

OBS_KEYS = ("observation.state", "observation.images.front", "observation.images.wrist", "task")
IMG_KEYS = ("observation.images.front", "observation.images.wrist")


def predict(policy, pre, post, state, task, imgs):
    batch = {"observation.state": state[None], "task": [task]}
    for k, v in imgs.items():
        batch[k] = v[None]
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(pre(batch))
    return post(chunk[:, 0])[0].cpu().numpy()


def main() -> None:
    ckpt = Path(sys.argv[1])
    policy, pre, post = load_policy(ckpt, "cuda")
    eps = [0, 5, 11, 23]
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/lerobot/panda_stack")
    ds = LeRobotDataset("local/panda_stack", root=root,
                        video_backend="pyav", episodes=eps)
    print(f"dataset: {root}")

    frames = [ds[i] for i in range(0, len(ds), 15)]
    rng = np.random.default_rng(0)

    d_true, d_swap, d_grey, delta_swap, delta_grey = [], [], [], [], []
    for n, s in enumerate(frames):
        true_imgs = {k: s[k] for k in IMG_KEYS}
        # A frame from a different episode: different cube layout, different task.
        other = frames[int(rng.integers(len(frames)))]
        while other["episode_index"].item() == s["episode_index"].item():
            other = frames[int(rng.integers(len(frames)))]
        swap_imgs = {k: other[k] for k in IMG_KEYS}
        grey_imgs = {k: torch.full_like(s[k], 0.5) for k in IMG_KEYS}

        rec = s["action"].numpy()
        state, task = s["observation.state"], s["task"]
        p_true = predict(policy, pre, post, state, task, true_imgs)
        p_swap = predict(policy, pre, post, state, task, swap_imgs)
        p_grey = predict(policy, pre, post, state, task, grey_imgs)

        d_true.append(np.abs(p_true - rec)[:7])
        d_swap.append(np.abs(p_swap - rec)[:7])
        d_grey.append(np.abs(p_grey - rec)[:7])
        delta_swap.append(np.abs(p_true - p_swap)[:7])
        delta_grey.append(np.abs(p_true - p_grey)[:7])

    acts = np.array([f["action"].numpy()[:7] for f in frames])
    deg = np.degrees
    print(f"{len(frames)} frames from episodes {eps}\n")
    print("error against the recorded action (deg, mean over 7 joints):")
    print(f"  true images     : {deg(np.mean(d_true)):6.2f}")
    print(f"  swapped scene   : {deg(np.mean(d_swap)):6.2f}")
    print(f"  grey images     : {deg(np.mean(d_grey)):6.2f}")
    print("\nhow far the PREDICTION moves when only the image changes (deg):")
    print(f"  true -> swapped : {deg(np.mean(delta_swap)):6.2f}")
    print(f"  true -> grey    : {deg(np.mean(delta_grey)):6.2f}")
    print("\nscale reference:")
    print(f"  std of recorded actions across these frames : {deg(acts.std(0).mean()):6.2f} deg")
    print(f"  mean |action - mean action|                 : {deg(np.abs(acts - acts.mean(0)).mean()):6.2f} deg")


if __name__ == "__main__":
    main()
