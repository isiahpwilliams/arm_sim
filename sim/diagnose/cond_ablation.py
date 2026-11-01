"""Does a --task-cond policy read the (src, dst) one-hot appended to its state?

The mirror of instruction_ablation.py for policies with no language input: hold the
images and the proprioceptive state fixed, vary only the 6 conditioning dims, and
measure how far the predicted action moves. Compare the shift against what changing
the images does -- if the label moves the action far less than the pixels do, the
conditioning channel is not reaching behaviour.

    MUJOCO_GL=egl pixi run -e train python sim/diagnose/cond_ablation.py <ckpt> data/lerobot/panda_stack_cond
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "sim")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from eval_policy import load_policy                          # noqa: E402

IMG_KEYS = ("observation.images.front", "observation.images.wrist")
N_CUBES = 3


def predict(policy, pre, post, state, task, imgs):
    batch = {"observation.state": state[None], "task": [task]}
    for k, v in imgs.items():
        batch[k] = v[None]
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(pre(batch))
    return post(chunk[:, 0])[0].cpu().numpy()


def variants(state: torch.Tensor) -> dict[str, torch.Tensor]:
    """Same proprioception, different task label."""
    src, dst = state[8:8 + N_CUBES], state[8 + N_CUBES:]
    out = {}
    swapped = state.clone()
    swapped[8:8 + N_CUBES], swapped[8 + N_CUBES:] = dst, src
    out["swap src/dst"] = swapped
    rolled = state.clone()
    rolled[8:8 + N_CUBES] = torch.roll(src, 1)
    out["other src"] = rolled
    zeroed = state.clone()
    zeroed[8:] = 0.0
    out["blank label"] = zeroed
    ones = state.clone()
    ones[8:] = 1.0
    out["all cubes"] = ones
    return out


def main() -> None:
    ckpt = Path(sys.argv[1])
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/lerobot/panda_stack_cond")
    policy, pre, post = load_policy(ckpt, "cuda")
    eps = [0, 5, 11, 23]
    ds = LeRobotDataset("local/panda_stack_cond", root=root, video_backend="pyav", episodes=eps)
    # The label only disambiguates where proprioception does not. By mid-trajectory
    # qpos already encodes which cube the arm is heading to, so sampling the whole
    # episode understates the label's role; --first N keeps only each episode's
    # opening frames, where every episode starts from the same home pose.
    first = int(sys.argv[sys.argv.index("--first") + 1]) if "--first" in sys.argv else 0
    if first:
        starts = {}
        for i in range(len(ds)):
            e = ds.hf_dataset[i]["episode_index"]
            e = int(e) if not hasattr(e, "item") else int(e.item())
            starts.setdefault(e, []).append(i)
        idx = [i for v in starts.values() for i in v[:first]]
        print(f"restricted to the first {first} frames of each episode")
    else:
        idx = list(range(0, len(ds), 15))
    frames = [ds[i] for i in idx]
    if frames[0]["observation.state"].shape[0] != 8 + 2 * N_CUBES:
        raise SystemExit("this dataset has no task one-hot; export it with --task-cond")

    deltas: dict[str, list] = {}
    img_delta = []
    rng = np.random.default_rng(0)
    for n, s in enumerate(frames):
        imgs = {k: s[k] for k in IMG_KEYS}
        state, task = s["observation.state"], s["task"]
        p_true = predict(policy, pre, post, state, task, imgs)
        for name, alt in variants(state).items():
            p_alt = predict(policy, pre, post, alt, task, imgs)
            deltas.setdefault(name, []).append(np.abs(p_true - p_alt)[:7])
        # Reference: the same measurement for the images, on the same frames.
        other = frames[int(rng.integers(len(frames)))]
        while other["episode_index"].item() == s["episode_index"].item():
            other = frames[int(rng.integers(len(frames)))]
        p_img = predict(policy, pre, post, state, task, {k: other[k] for k in IMG_KEYS})
        img_delta.append(np.abs(p_true - p_img)[:7])

    acts = np.array([f["action"].numpy()[:7] for f in frames])
    deg = np.degrees
    print(f"{len(frames)} frames from episodes {eps}\n")
    print("how far the PREDICTION moves when only the TASK LABEL changes (deg):")
    for name, d in deltas.items():
        print(f"  true -> {name:<12}: {deg(np.mean(d)):6.2f}")
    print("\nreference, same frames:")
    print(f"  true -> swapped scene images : {deg(np.mean(img_delta)):6.2f}")
    print(f"  std of recorded actions      : {deg(acts.std(0).mean()):6.2f}")


if __name__ == "__main__":
    main()
