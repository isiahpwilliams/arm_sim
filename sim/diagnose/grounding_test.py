"""Does the policy aim at the cube it was told to pick, or at a fixed average spot?

For each seed we roll the policy out and freeze the moment it decides to close the
gripper (action[7] crossing from open toward closed). At that instant we record where
the TCP actually is and where the three cubes actually are. Three questions fall out:

  1. grounding gain -- regress tcp_xy_at_close on src_cube_xy across seeds.
     slope ~1 means the aim point follows the cube; slope ~0 means it goes to the
     same place no matter where the cube is, i.e. vision is not driving the reach.
  2. target identity -- is the cube nearest the gripper at close time the SOURCE cube
     the instruction named, or some other cube? Catches language-grounding failures.
  3. residual error -- how far off is the aim point when it does track.

Run from the repo root so sim/ imports resolve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "sim")

from arm_env import CUBE_NAMES, PandaCubeEnv  # noqa: E402
from tasks import sample_task                 # noqa: E402
import eval_policy                                              # noqa: E402
from eval_policy import load_policy, make_batch, MAX_STEPS  # noqa: E402

CLOSE_THRESHOLD = 128.0          # gripper command below this = closing (255 open, 0 closed)
SPAWN_MEAN = np.array([0.50, 0.0])   # centre of the cube spawn box: the "ignores vision" null


def main() -> None:
    ckpt = Path(sys.argv[1])
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    if "--zero-state" in sys.argv:
        eval_policy.ZERO_STATE = True
        print("proprioception blanked (--zero-state)")
    device = "cuda"

    policy, pre, post = load_policy(ckpt, device)
    env = PandaCubeEnv()
    cameras = ["front", "wrist"]

    # A single-pair policy only ever saw one (src, dst); scoring it on scenes whose
    # instruction names a different pair would measure a task it was never taught.
    pair = None
    if "--pair" in sys.argv:
        pair = tuple(sys.argv[sys.argv.index("--pair") + 1].split(","))
        print(f"restricted to scenes whose task is {pair[0]} -> {pair[1]}")

    seeds, s = [], 0
    while len(seeds) < n_seeds:
        if pair is None or sample_task(np.random.default_rng(s))[:2] == pair:
            seeds.append(s)
        s += 1

    rows = []
    for seed in seeds:
        obs = env.reset(seed=seed)
        src, dst, instruction = sample_task(np.random.default_rng(seed))
        policy.reset()

        cubes0 = {c: obs["cubes"][c][:2].copy() for c in CUBE_NAMES}
        close_tcp = None
        for step in range(MAX_STEPS):
            frames = {c: env.render(c) for c in cameras}
            with torch.inference_mode():
                action = post(policy.select_action(pre(make_batch(obs, frames, instruction, src, dst))))
            a = action[0].cpu().numpy()
            if close_tcp is None and a[7] < CLOSE_THRESHOLD:
                # Record before stepping: this is the pose the policy chose to grasp from.
                close_tcp = obs["tcp_pos"].copy()
                close_step = step
                break
            obs = env.step(a)

        if close_tcp is None:
            print(f"seed {seed:3d}  never commanded a close")
            continue

        tcp_xy = close_tcp[:2]
        d = {c: float(np.linalg.norm(tcp_xy - cubes0[c])) for c in CUBE_NAMES}
        nearest = min(d, key=d.get)
        rows.append(dict(seed=seed, src=src, dst=dst, tcp=tcp_xy, src_xy=cubes0[src],
                         d_src=d[src], d_dst=d[dst], nearest=nearest, step=close_step,
                         z=float(close_tcp[2])))
        print(f"seed {seed:3d}  step{close_step:4d}  src={src.replace('cube_',''):5s} "
              f"src_xy=({cubes0[src][0]:.3f},{cubes0[src][1]:+.3f})  "
              f"tcp_xy=({tcp_xy[0]:.3f},{tcp_xy[1]:+.3f})  tcp_z={close_tcp[2]:.3f}  "
              f"err_src={100*d[src]:5.1f}cm  err_dst={100*d[dst]:5.1f}cm  nearest={nearest.replace('cube_','')}")

    if not rows:
        print("no closes recorded -- nothing to regress")
        return

    tcp = np.array([r["tcp"] for r in rows])
    srcxy = np.array([r["src_xy"] for r in rows])

    print(f"\n=== {len(rows)} grasp attempts ===")
    for i, axis in enumerate("xy"):
        # tcp_axis = a + b * src_axis ; b is the grounding gain on that axis.
        b, a = np.polyfit(srcxy[:, i], tcp[:, i], 1)
        r = np.corrcoef(srcxy[:, i], tcp[:, i])[0, 1]
        print(f"  {axis}: grounding gain b={b:+.2f}  (1.0 = tracks cube, 0.0 = ignores it)   r={r:+.2f}")

    err_src = np.array([r["d_src"] for r in rows])
    err_null = np.linalg.norm(tcp - SPAWN_MEAN, axis=1)
    spread = np.linalg.norm(srcxy - SPAWN_MEAN, axis=1)
    print(f"  mean |tcp - src cube|      : {100*err_src.mean():5.1f} cm   (median {100*np.median(err_src):.1f})")
    print(f"  mean |tcp - spawn centre|  : {100*err_null.mean():5.1f} cm")
    print(f"  mean |src cube - centre|   : {100*spread.mean():5.1f} cm   <- how far cubes actually spread")
    print(f"  tcp spread (std x,y)       : {100*tcp.std(0)[0]:5.1f}, {100*tcp.std(0)[1]:5.1f} cm")
    print(f"  cube spread (std x,y)      : {100*srcxy.std(0)[0]:5.1f}, {100*srcxy.std(0)[1]:5.1f} cm")
    right = sum(r["nearest"] == r["src"] for r in rows)
    print(f"  closed nearest the SOURCE cube: {right}/{len(rows)}")
    print(f"  mean close height tcp_z    : {100*np.mean([r['z'] for r in rows]):5.1f} cm")


if __name__ == "__main__":
    main()
