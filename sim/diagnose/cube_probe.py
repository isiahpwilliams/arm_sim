"""Perception probe: can a small CNN read the three cubes' xy off the front camera?

No language, no proprioception, no action expert -- just image -> 6 numbers
(red/blue/green cube xy in metres). This separates "the pixels do not carry the
cube position" from "SmolVLA will not extract it".

Labels come from data/raw/*/steps.npz, which stored privileged cube xyz per step
during collection, so nothing has to be re-simulated.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn

RAW = Path("data/raw")
CUBES = ["cube_cube_red", "cube_cube_blue", "cube_cube_green"]


def load_split(eps, stride):
    """Returns images (N,240,320,3) uint8, labels (N,6) float32, step index (N,)."""
    imgs, labels, steps = [], [], []
    for ep in eps:
        d = RAW / f"episode_{ep:06d}"
        if not d.exists():
            continue
        z = np.load(d / "steps.npz")
        xy = np.concatenate([z[c][:, :2] for c in CUBES], axis=1)  # (T,6)
        frames = imageio.mimread(d / "front.mp4", memtest=False)
        T = min(len(frames), len(xy))
        idx = np.arange(0, T, stride)
        imgs.append(np.stack([frames[i] for i in idx]))
        labels.append(xy[idx])
        steps.append(idx)
    return (np.concatenate(imgs), np.concatenate(labels).astype(np.float32),
            np.concatenate(steps))


class CubeNet(nn.Module):
    """Plain conv stack, flatten, FC. Trained from scratch -- no pretrained features."""

    def __init__(self):
        super().__init__()
        chans = [3, 32, 64, 128, 256, 256]
        layers = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.BatchNorm2d(b), nn.ReLU()]
        self.conv = nn.Sequential(*layers)          # 240x320 -> 8x10
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 8 * 10, 256), nn.ReLU(), nn.Linear(256, 6))

    def forward(self, x):
        return self.head(self.conv(x))


def batches(imgs, labels, bs, device, shuffle):
    bs = min(bs, len(imgs))
    order = np.random.permutation(len(imgs)) if shuffle else np.arange(len(imgs))
    for i in range(0, len(order) - (bs - 1 if shuffle else 0), bs):
        sel = order[i:i + bs]
        x = torch.from_numpy(imgs[sel]).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        yield x, torch.from_numpy(labels[sel]).to(device)


def per_cube_cm(pred, true):
    """Mean Euclidean xy error per cube, in cm."""
    d = (pred - true).reshape(-1, 3, 2)
    return 100.0 * np.linalg.norm(d, axis=2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-eps", type=int, default=400)
    ap.add_argument("--val-eps", type=int, default=100)
    ap.add_argument("--stride", type=int, default=4, help="keep every Nth frame")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tr_x, tr_y, _ = load_split(range(args.train_eps), args.stride)
    va_x, va_y, va_s = load_split(range(700, 700 + args.val_eps), args.stride)
    print(f"train {tr_x.shape}  val {va_x.shape}   decoded in {time.time() - t0:.0f}s", flush=True)

    # Normalise targets so all six outputs are on comparable scales.
    mu, sd = tr_y.mean(0), tr_y.std(0)
    print(f"label mean {np.round(mu, 3)}  std {np.round(sd, 3)}")
    print(f"predict-the-mean baseline on val: {per_cube_cm(np.tile(mu, (len(va_y), 1)), va_y):.1f} cm")

    net = CubeNet().to(device)
    print(f"probe params: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs * (len(tr_x) // min(args.batch, len(tr_x)))))
    mu_t = torch.as_tensor(mu, device=device)
    sd_t = torch.as_tensor(sd, device=device)

    for ep in range(args.epochs):
        net.train()
        tot = n = 0
        for x, y in batches(tr_x, tr_y, args.batch, device, True):
            loss = nn.functional.mse_loss(net(x), (y - mu_t) / sd_t)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(x); n += len(x)

        net.eval()
        preds = []
        with torch.inference_mode():
            for x, _ in batches(va_x, va_y, args.batch, device, False):
                preds.append((net(x) * sd_t + mu_t).cpu().numpy())
        p = np.concatenate(preds)
        v = va_y[:len(p)]; s = va_s[:len(p)]
        early = s < 60
        print(f"epoch {ep + 1:2d}  train mse {tot / n:.4f}   "
              f"val {per_cube_cm(p, v):5.2f} cm   "
              f"approach-phase (step<60) {per_cube_cm(p[early], v[early]):5.2f} cm", flush=True)

    # Which axis is harder: x is the depth axis for the front camera.
    d = np.abs(p - v).reshape(-1, 3, 2).mean(axis=(0, 1))
    print(f"mean |err| x (depth) {100 * d[0]:.2f} cm   y (lateral) {100 * d[1]:.2f} cm")
    for i, c in enumerate(CUBES):
        e = 100 * np.linalg.norm((p - v).reshape(-1, 3, 2)[:, i], axis=1).mean()
        print(f"  {c:16s} {e:5.2f} cm")
    print("=== PROBE DONE ===")


if __name__ == "__main__":
    main()
