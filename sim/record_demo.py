"""Wave the arm over randomised cubes and record all cameras to outputs/demo.mp4.

    pixi run record-demo -- --seconds 4 --seed 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from arm_env import PandaCubeEnv, HOME_QPOS, GRIPPER_OPEN, GRIPPER_CLOSED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--amplitude", type=float, default=0.4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env = PandaCubeEnv(control_hz=20.0, render_size=(240, 320))
    obs = env.reset(seed=args.seed)
    cams = env.cameras
    print("cameras:", cams)
    print("cubes:", {k: v.round(3).tolist() for k, v in obs["cubes"].items()})

    n_steps = int(args.seconds * env.control_hz)
    writer = imageio.get_writer(args.out / "demo.mp4", fps=int(env.control_hz), codec="libx264",
                                quality=8, macro_block_size=None)
    phase = np.arange(7) * 0.5
    for k in range(n_steps):
        t = k / env.control_hz
        # Damp the pitch joints so the arm stays clear of the table.
        wave = args.amplitude * np.sin(t + phase) * np.array([1, 0.3, 1, 0.3, 1, 0.3, 1])
        gripper = GRIPPER_OPEN if int(t) % 2 == 0 else GRIPPER_CLOSED
        obs = env.step(np.concatenate([HOME_QPOS + wave, [gripper]]))

        frames = [env.render(c) for c in cams]
        writer.append_data(np.concatenate(frames, axis=1))

        if k % int(env.control_hz) == 0:
            print(f"t={obs['time']:4.1f}s  hand={obs['ee_pos'].round(3)}  "
                  f"q1={obs['qpos'][0]:+.2f}  gripper={obs['gripper']*100:.1f}cm")
        if k in (0, n_steps // 2):
            for c, f in zip(cams, frames):
                imageio.imwrite(args.out / f"frame_{k:04d}_{c}.png", f)
    writer.close()
    print(f"wrote {args.out / 'demo.mp4'} ({n_steps} frames, {len(cams)} cameras tiled)")


if __name__ == "__main__":
    main()
