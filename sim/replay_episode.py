"""Replay a recorded episode's actions open-loop and check the outcome reproduces.

    pixi run replay -- data/raw/episode_000000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from arm_env import CUBE_HALF, PandaCubeEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=Path)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    meta = json.loads((args.episode / "meta.json").read_text())
    steps = np.load(args.episode / "steps.npz")
    actions = steps["action"]
    print(f"episode {meta['episode']}  seed {meta['seed']}  \"{meta['instruction']}\"  {len(actions)} steps")

    env = PandaCubeEnv()
    obs = env.reset(seed=meta["seed"])
    writer = None if args.no_video else imageio.get_writer(
        args.episode / "replay.mp4", fps=int(meta["fps"]), codec="libx264", quality=8, macro_block_size=None)

    max_tcp_err = 0.0
    for t, a in enumerate(actions):
        max_tcp_err = max(max_tcp_err, float(np.linalg.norm(obs["tcp_pos"] - steps["tcp_pos"][t])))
        if writer is not None:
            writer.append_data(np.concatenate([env.render(c) for c in env.cameras], axis=1))
        obs = env.step(a)
    for _ in range(10):
        obs = env.step(actions[-1])
    if writer is not None:
        writer.close()

    s, d = obs["cubes"][meta["src"]], obs["cubes"][meta["dst"]]
    stacked = np.linalg.norm(s[:2] - d[:2]) < 0.015 and abs(s[2] - (d[2] + 2 * CUBE_HALF)) < 0.008
    print(f"max TCP deviation from recording: {max_tcp_err * 1000:.2f} mm")
    print(f"final: {meta['src']} at {s.round(3)}, {meta['dst']} at {d.round(3)} -> stacked={stacked}")
    print("REPRODUCIBLE" if stacked and max_tcp_err < 0.002 else "MISMATCH")


if __name__ == "__main__":
    main()
