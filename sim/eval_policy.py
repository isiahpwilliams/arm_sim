"""Run a trained LeRobot policy closed-loop in PandaCubeEnv and report success rate.

    pixi run -e train eval -- outputs/train/act/checkpoints/last/pretrained_model --seeds 20 --video outputs/eval.mp4

Evaluation seeds default to 0..N-1, which are never used for data collection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

from arm_env import PandaCubeEnv
from expert import StackingExpert
from tasks import sample_task

MAX_STEPS = 400


def load_policy(path: Path, device: str, n_action_steps: int | None = None,
                temporal_ensemble: float | None = None):
    """Policy plus its normalisation pipelines (saved beside the weights)."""
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    if n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    if temporal_ensemble is not None:      # ACT: re-plan every step, blend overlapping chunks
        cfg.temporal_ensemble_coeff = temporal_ensemble
        cfg.n_action_steps = 1
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def make_batch(obs: dict, frames: dict[str, np.ndarray], instruction: str) -> dict:
    batch = {
        "observation.state": torch.as_tensor(
            np.concatenate([obs["qpos"], [obs["gripper"]]]), dtype=torch.float32)[None],
        "task": [instruction],
    }
    for cam, img in frames.items():
        batch[f"observation.images.{cam}"] = (torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)[None]
    return batch


def run_episode(env, checker, policy, pre, post, cameras, seed, writer=None):
    """Returns (success, instruction, max source-cube lift in m, final source->target xy distance in m)."""
    obs = env.reset(seed=seed)
    src, dst, instruction = sample_task(np.random.default_rng(seed))
    checker.reset(obs, src, dst)
    policy.reset()
    max_lift = 0.0
    for _ in range(MAX_STEPS):
        frames = {c: env.render(c) for c in cameras}
        if writer is not None:
            writer.append_data(np.concatenate([frames[c] for c in cameras], axis=1))
        with torch.inference_mode():
            action = post(policy.select_action(pre(make_batch(obs, frames, instruction))))
        obs = env.step(action[0].cpu().numpy())
        max_lift = max(max_lift, float(obs["cubes"][src][2]) - 0.02)
        if checker.success(obs):
            break
    dist = float(np.linalg.norm(obs["cubes"][src][:2] - obs["cubes"][dst][:2]))
    return checker.success(obs), instruction, max_lift, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--cameras", nargs="+", default=["front", "wrist"])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--video", type=Path, default=None, help="record the first N seeds, tiled")
    ap.add_argument("--video-episodes", type=int, default=3)
    ap.add_argument("--n-action-steps", type=int, default=None, help="override how many chunk steps run open-loop")
    ap.add_argument("--temporal-ensemble", type=float, default=None, help="ACT temporal ensembling coeff, e.g. 0.01")
    ap.add_argument("--pair", default=None,
                    help="only evaluate scenes whose task is SRC,DST (e.g. cube_red,cube_blue); "
                         "scans seeds upward until --seeds matching scenes have run")
    args = ap.parse_args()
    pair = tuple(args.pair.split(",")) if args.pair else None

    policy, pre, post = load_policy(args.checkpoint, args.device, args.n_action_steps, args.temporal_ensemble)
    env = PandaCubeEnv()
    checker = StackingExpert(env)   # only its success() test is used
    writer = None
    if args.video is not None:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=20, codec="libx264", quality=8, macro_block_size=None)

    results, lifts, dists = [], [], []
    seed = args.start
    while len(results) < args.seeds:
        seed += 1
        if pair is not None and sample_task(np.random.default_rng(seed - 1))[:2] != pair:
            continue
        i = len(results)
        w = writer if (writer is not None and i < args.video_episodes) else None
        ok, instruction, lift, dist = run_episode(env, checker, policy, pre, post, args.cameras, seed - 1, w)
        results.append(ok); lifts.append(lift); dists.append(dist)
        print(f"seed {seed - 1:4d}  {'OK  ' if ok else 'FAIL'}  lift={lift * 100:4.1f}cm  final_dist={dist * 100:4.1f}cm  \"{instruction}\"")
    if writer is not None:
        writer.close()
    grasped = sum(l > 0.03 for l in lifts)
    print(f"success: {sum(results)}/{len(results)}  ({100 * np.mean(results):.0f}%)   "
          f"grasped+lifted: {grasped}/{len(lifts)}   mean final src->dst distance: {100 * np.mean(dists):.1f} cm")


if __name__ == "__main__":
    main()
