"""Are the images still distinguishable after preprocessing reaches the model?

A policy that ignores vision might be ignoring it because the images arrive degenerate
(saturated, constant, or identical between scenes). Compare the preprocessed tensors
for two different scenes: if they differ, the pipeline is fine and the policy is
choosing to ignore real information.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, "sim")
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from eval_policy import load_policy                          # noqa: E402

IMG_KEYS = ("observation.images.front", "observation.images.wrist")

ckpt = Path(sys.argv[1])
policy, pre, post = load_policy(ckpt, "cuda")
ds = LeRobotDataset("local/panda_stack", root=Path("data/lerobot/panda_stack"),
                    video_backend="pyav", episodes=[0, 11])

a, b = ds[10], ds[len(ds) - 10]
print("raw dataset tensors:")
for k in IMG_KEYS:
    print(f"  {k}: shape={tuple(a[k].shape)} dtype={a[k].dtype} "
          f"min={a[k].min():.3f} max={a[k].max():.3f} mean={a[k].mean():.3f}")
print(f"  mean |A - B| raw (front): {(a[IMG_KEYS[0]] - b[IMG_KEYS[0]]).abs().mean():.4f}")

out = {}
for name, s in (("A", a), ("B", b)):
    batch = {"observation.state": s["observation.state"][None], "task": [s["task"]]}
    for k in IMG_KEYS:
        batch[k] = s[k][None]
    out[name] = pre(batch)

print("\nafter the policy's own preprocessor:")
for k in out["A"]:
    va = out["A"][k]
    if not torch.is_tensor(va) or va.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        continue
    vb = out["B"][k]
    print(f"  {k}: shape={tuple(va.shape)} min={va.min():.3f} max={va.max():.3f} "
          f"mean={va.mean():.3f} std={va.std():.3f}  |A-B|={(va.float()-vb.float()).abs().mean():.4f}")
