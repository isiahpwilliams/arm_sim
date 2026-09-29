"""Record expert demonstrations to data/raw/episode_XXXXXX/{meta.json, steps.npz, <cam>.mp4}.

    pixi run collect -- --episodes 200 --start-seed 10000 --cameras front wrist

steps.npz holds per-step arrays: action (T,8), qpos/qvel (T,7), gripper, tcp_pos,
tcp_quat, cube_<name> (privileged), phase, time. Failed episodes are skipped
unless --keep-failures is passed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from arm_env import CUBE_NAMES, PandaCubeEnv
from expert import StackingExpert
from tasks import sample_task

MAX_STEPS = 400
SETTLE_STEPS = 10


def record_episode(env, expert, seed, cameras):
    obs = env.reset(seed=seed)
    src, dst, instruction = sample_task(np.random.default_rng(seed))
    expert.reset(obs, src, dst)

    frames = {c: [] for c in cameras}
    rows = {k: [] for k in ["action", "qpos", "qvel", "gripper", "tcp_pos", "tcp_quat", "phase", "time"]}
    rows.update({f"cube_{c}": [] for c in CUBE_NAMES})

    for _ in range(MAX_STEPS):
        for c in cameras:
            frames[c].append(env.render(c))
        action = expert.act(obs)
        rows["action"].append(action)
        rows["qpos"].append(obs["qpos"]); rows["qvel"].append(obs["qvel"])
        rows["gripper"].append(obs["gripper"])
        rows["tcp_pos"].append(obs["tcp_pos"]); rows["tcp_quat"].append(obs["tcp_quat"])
        rows["phase"].append(min(expert.phase_i, len(expert.phases)))
        rows["time"].append(obs["time"])
        for c in CUBE_NAMES:
            rows[f"cube_{c}"].append(obs["cubes"][c])
        obs = env.step(action)
        if expert.done:
            for _ in range(SETTLE_STEPS):
                obs = env.step(action)
            break

    success = expert.success(obs)
    meta = {
        "seed": seed, "instruction": instruction, "src": src, "dst": dst,
        "success": success, "length": len(rows["action"]),
        "fps": env.control_hz, "cameras": list(cameras),
        "image_hw": list(frames[cameras[0]][0].shape[:2]),
    }
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    return meta, arrays, frames


def write_episode(out_dir: Path, meta, arrays, frames):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    np.savez_compressed(out_dir / "steps.npz", **arrays)
    for cam, imgs in frames.items():
        w = imageio.get_writer(out_dir / f"{cam}.mp4", fps=int(meta["fps"]), codec="libx264",
                               quality=8, macro_block_size=None)
        for im in imgs:
            w.append_data(im)
        w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--cameras", nargs="+", default=["front", "wrist"])
    ap.add_argument("--keep-failures", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env = PandaCubeEnv()
    expert = StackingExpert(env)
    index_path = args.out / "index.jsonl"
    existing = sum(1 for _ in open(index_path)) if index_path.exists() else 0

    kept = failed = 0
    with open(index_path, "a") as index:
        for i in range(args.episodes):
            seed = args.start_seed + i
            meta, arrays, frames = record_episode(env, expert, seed, args.cameras)
            if not meta["success"] and not args.keep_failures:
                failed += 1
                print(f"seed {seed:5d}  FAIL  (skipped)  \"{meta['instruction']}\"")
                continue
            ep_id = existing + kept
            meta["episode"] = ep_id
            write_episode(args.out / f"episode_{ep_id:06d}", meta, arrays, frames)
            index.write(json.dumps(meta) + "\n"); index.flush()
            kept += 1
            print(f"seed {seed:5d}  {'OK  ' if meta['success'] else 'FAIL'}  ep {ep_id:5d}  "
                  f"{meta['length']:3d} steps  \"{meta['instruction']}\"")
    print(f"kept {kept}, failed {failed}, total episodes on disk: {existing + kept}")


if __name__ == "__main__":
    main()
