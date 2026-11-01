"""A/B the policy's action error on codec-decoded dataset frames vs raw renderer output.

Everything is held constant -- same checkpoint, same sampled (episode, frame) pairs,
same recorded observation.state, same instruction -- except the image source.
Run from the repo root with the `train` env and MUJOCO_GL=egl.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "sim")
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from arm_env import PandaCubeEnv  # noqa: E402
from eval_policy import load_policy  # noqa: E402

CAMS = ("front", "wrist")


def predict(policy, pre, post, images: dict, state: torch.Tensor, task: str) -> np.ndarray:
    batch = {"observation.state": state[None], "task": [task]}
    for cam, img in images.items():
        batch[f"observation.images.{cam}"] = img[None]
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(pre(batch))
    return post(chunk[:, 0])[0].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--root", type=Path, default=Path("data/lerobot/panda_stack"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--episodes", type=int, nargs="+", default=[2, 9, 19])
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    policy, pre, post = load_policy(args.checkpoint, args.device)
    ds = LeRobotDataset("local/panda_stack", root=args.root,
                        video_backend="pyav", episodes=args.episodes)

    # Collect the sampled frames and their codec-decoded images.
    samples = []
    for i in range(0, len(ds), args.every):
        s = ds[i]
        samples.append({
            "ep": int(s["episode_index"]),
            "frame": int(s["frame_index"]),
            "state": s["observation.state"],
            "action": s["action"].numpy(),
            "task": s["task"],
            "codec": {c: s[f"observation.images.{c}"] for c in CAMS},
        })
    print(f"{len(samples)} sampled frames from episodes {args.episodes}", flush=True)

    # Replay each episode once, capturing raw renders at exactly those frame indices.
    env = PandaCubeEnv()
    wanted: dict[int, set[int]] = {}
    for s in samples:
        wanted.setdefault(s["ep"], set()).add(s["frame"])

    raw_frames: dict[tuple[int, int], dict] = {}
    max_dev = 0.0
    for ep, frames in sorted(wanted.items()):
        ep_dir = args.raw / f"episode_{ep:06d}"
        meta = json.loads((ep_dir / "meta.json").read_text())
        steps = np.load(ep_dir / "steps.npz")
        actions, rec_tcp = steps["action"], steps["tcp_pos"]
        obs = env.reset(seed=meta["seed"])
        for t in range(len(actions)):
            if t in frames:
                # Confirm the replay is still on the recorded trajectory.
                max_dev = max(max_dev, float(np.linalg.norm(obs["tcp_pos"] - rec_tcp[t])))
                raw_frames[(ep, t)] = {
                    c: torch.from_numpy(env.render(c).copy()).permute(2, 0, 1).float() / 255.0
                    for c in CAMS
                }
            obs = env.step(actions[t])
    print(f"max TCP deviation of replay from recording: {max_dev * 1000:.2f} mm", flush=True)

    err_codec, err_raw, img_absdiff = [], [], []
    for s in samples:
        raw = raw_frames.get((s["ep"], s["frame"]))
        if raw is None:
            continue
        p_codec = predict(policy, pre, post, s["codec"], s["state"], s["task"])
        p_raw = predict(policy, pre, post, raw, s["state"], s["task"])
        err_codec.append(np.abs(p_codec - s["action"]))
        err_raw.append(np.abs(p_raw - s["action"]))
        img_absdiff.append(np.mean([
            float(torch.mean(torch.abs(raw[c] - s["codec"][c]))) for c in CAMS
        ]))

    ec, er = np.mean(err_codec, axis=0), np.mean(err_raw, axis=0)
    print(f"\ncompared on {len(err_codec)} frames")
    print(f"mean |pixel difference| raw vs codec (0-1 scale): {np.mean(img_absdiff):.4f}")
    print("\n                        joints (deg)   gripper (0-255)")
    print(f"  codec-decoded frames  {np.degrees(ec[:7].mean()):12.2f}   {ec[7]:13.1f}")
    print(f"  raw renderer frames   {np.degrees(er[:7].mean()):12.2f}   {er[7]:13.1f}")
    ratio = np.degrees(er[:7].mean()) / max(np.degrees(ec[:7].mean()), 1e-9)
    print(f"\n  raw / codec joint-error ratio: {ratio:.1f}x")
    print(f"  per-joint codec (deg): {np.degrees(ec[:7]).round(2)}")
    print(f"  per-joint raw   (deg): {np.degrees(er[:7]).round(2)}")


if __name__ == "__main__":
    main()
