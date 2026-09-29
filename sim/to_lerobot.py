"""Convert data/raw episodes into a LeRobot v3 dataset (run in the `train` env).

    pixi run -e train to-lerobot -- --raw data/raw --out data/lerobot/panda_stack

Exports observation.images.<cam>, observation.state (7 joints + gripper opening),
action (7 joint targets + gripper command), and task. Privileged cube poses are
deliberately left out.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
STATE_NAMES = JOINT_NAMES + ["gripper_opening_m"]
ACTION_NAMES = [f"{j}_target" for j in JOINT_NAMES] + ["gripper_cmd"]


def read_video(path: Path) -> np.ndarray:
    reader = imageio.get_reader(path)
    frames = np.stack([np.asarray(f) for f in reader])
    reader.close()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/lerobot/panda_stack"))
    ap.add_argument("--repo-id", default="local/panda_stack")
    ap.add_argument("--cameras", nargs="+", default=["front", "wrist"])
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N episodes")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    episodes = sorted(p for p in args.raw.glob("episode_*") if (p / "meta.json").exists())
    if args.limit:
        episodes = episodes[: args.limit]
    if not episodes:
        raise SystemExit(f"no episodes under {args.raw}")
    first = json.loads((episodes[0] / "meta.json").read_text())
    fps = int(first["fps"])
    h, w = first["image_hw"]

    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out} exists; pass --overwrite to replace it")
        shutil.rmtree(args.out)

    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (8,), "names": ACTION_NAMES},
    }
    for cam in args.cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=fps, features=features, root=args.out,
        robot_type="franka_panda_sim", use_videos=True)

    total_frames = 0
    for ep_dir in episodes:
        meta = json.loads((ep_dir / "meta.json").read_text())
        steps = np.load(ep_dir / "steps.npz")
        T = len(steps["action"])
        vids = {cam: read_video(ep_dir / f"{cam}.mp4") for cam in args.cameras}
        for cam, v in vids.items():
            if len(v) != T:
                raise RuntimeError(f"{ep_dir.name}: {cam}.mp4 has {len(v)} frames, steps has {T}")

        state = np.concatenate([steps["qpos"], steps["gripper"][:, None]], axis=1).astype(np.float32)
        action = steps["action"].astype(np.float32)
        for t in range(T):
            frame = {
                "observation.state": state[t],
                "action": action[t],
                "task": meta["instruction"],
            }
            for cam in args.cameras:
                frame[f"observation.images.{cam}"] = vids[cam][t]
            ds.add_frame(frame)
        ds.save_episode()
        total_frames += T
        print(f"{ep_dir.name}: {T:3d} frames  \"{meta['instruction']}\"")

    ds.finalize()
    print(f"\nwrote {len(episodes)} episodes / {total_frames} frames to {args.out}")
    print("features:", list(features))


if __name__ == "__main__":
    main()
