"""Run the scripted expert on N seeds and report the success rate.

    pixi run expert -- --seeds 20 --video outputs/expert.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from arm_env import PandaCubeEnv
from expert import StackingExpert
from tasks import sample_task

MAX_STEPS = 400


def run_episode(env, expert, seed, writer=None, verbose=False):
    obs = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    src, dst, instruction = sample_task(rng)
    expert.reset(obs, src, dst)
    last_phase = None
    for step in range(MAX_STEPS):
        if verbose and expert.phase != last_phase:
            print(f"  step {step:3d}  -> {expert.phase:12s} tcp={obs['tcp_pos'].round(3)}")
            last_phase = expert.phase
        action = expert.act(obs)
        obs = env.step(action)
        if writer is not None:
            writer.append_data(np.concatenate([env.render(c) for c in env.cameras], axis=1))
        if expert.done:
            for _ in range(10):
                obs = env.step(action)
                if writer is not None:
                    writer.append_data(np.concatenate([env.render(c) for c in env.cameras], axis=1))
            break
    if verbose:
        s_, d_ = obs["cubes"][src], obs["cubes"][dst]
        print(f"  final: {src}={s_.round(3)} {dst}={d_.round(3)} tcp={obs['tcp_pos'].round(3)} "
              f"xy_err={np.linalg.norm(s_[:2]-d_[:2])*1000:.1f}mm z_err={(s_[2]-d_[2]-0.04)*1000:.1f}mm")
    return expert.success(obs), instruction, step + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--video", type=Path, default=None, help="record the first seed to this mp4")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    env = PandaCubeEnv()
    expert = StackingExpert(env)
    results = []
    for seed in range(args.start, args.start + args.seeds):
        writer = None
        if args.video is not None and seed == args.start:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.video, fps=20, codec="libx264", quality=8, macro_block_size=None)
        ok, instruction, n = run_episode(env, expert, seed, writer, args.verbose)
        if writer is not None:
            writer.close()
        results.append(ok)
        print(f"seed {seed:4d}  {'OK  ' if ok else 'FAIL'}  {n:3d} steps  \"{instruction}\"")
    print(f"success: {sum(results)}/{len(results)}")


if __name__ == "__main__":
    main()
