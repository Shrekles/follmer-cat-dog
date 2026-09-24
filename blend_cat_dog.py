"""Cat→dog blends along a single Brownian path.

Each row is one Brownian path B. Column j is the Föllmer sample driven by that same B,
with class guidance  w · [(1−λ_j) ∇log p(cat|y) + λ_j ∇log p(dog|y)],  λ_j from 0 to 1.
Because every column shares ω, the pose, layout and colours stay fixed while the class moves.
The endpoints (λ = 0, 1) are the w-tilted cat and dog laws (exact conditionals only when w = 1).

    python blend_cat_dog.py --ckpt checkpoints/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth \
        --classifier checkpoints/cifar10_noisy_classifier.pt --w 6 --n 8 --k 7 --steps 500
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from follmer import BrownianPath, StackedPath, follmer_sample, symmetric_grid
from noisy_classifier import load_classifier
from run_cat_dog import CIFAR, save_grid
from score_sde_backend import ScoreSDEDenoiser, get_device, load_score_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="score_sde_pytorch")
    p.add_argument("--config", default="vp/cifar10_ddpmpp_continuous")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--classifier", required=True)
    p.add_argument("--a", default="cat", choices=CIFAR)
    p.add_argument("--b", default="dog", choices=CIFAR)
    p.add_argument("--w", type=float, nargs="+", default=[6.0], help="guidance scale(s); one grid per value")
    p.add_argument("--n", type=int, default=8, help="Brownian paths (rows)")
    p.add_argument("--k", type=int, default=7, help="blend steps per row (columns)")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/blend")
    args = p.parse_args()

    device = get_device()
    config, sde, eps, score_fn = load_score_model(args.repo, args.config, args.ckpt, device)
    clf = load_classifier(args.classifier, device)
    den = ScoreSDEDenoiser(sde, eps, score_fn, classifier=clf, guidance=1.0)  # w lives in the weights
    shape = (config.data.num_channels, config.data.image_size, config.data.image_size)
    grid = symmetric_grid(args.steps)
    ia, ib = CIFAR.index(args.a), CIFAR.index(args.b)
    lams = torch.linspace(0, 1, args.k)

    # One network batch: for each w, n paths × k blend weights, every column of a row on the same B.
    paths, weights = [], []
    for w in args.w:
        for lam in lams:
            paths.append(BrownianPath(grid, (args.n,) + shape, seed=args.seed))
            W = torch.zeros(args.n, 10)
            W[:, ia], W[:, ib] = w * (1 - lam), w * lam
            weights.append(W)
    weights = torch.cat(weights).to(device)
    print(f"device={device}  {len(args.w)}×{args.k}×{args.n} = {weights.shape[0]} samples, {args.steps} steps")
    t0 = time.time()
    res = follmer_sample(den, StackedPath(paths), device, labels=weights, verbose=True)
    print(f"done in {time.time() - t0:.0f}s")

    os.makedirs(args.out, exist_ok=True)
    x = res.x1.view(len(args.w), args.k, args.n, *shape)
    with torch.no_grad():  # classifier's p(a), p(b) on the (nearly clean) samples, as a rough label check
        s = torch.full((args.n,), eps, device=device)
        probs = torch.stack([torch.stack([F.softmax(clf(x[i, j].to(device), s), -1)[:, [ia, ib]].cpu()
                                          for j in range(args.k)]) for i in range(len(args.w))])
    for i, w in enumerate(args.w):
        rows = [x[i, :, r] for r in range(args.n)]  # row r: path r, columns λ = 0 … 1
        save_grid(rows, f"{args.out}/blend_w{w:g}.png", config.data.centered)
        pa, pb = probs[i, :, :, 0].mean(1), probs[i, :, :, 1].mean(1)
        print(f"w={w:g}  mean p({args.a}) by λ: {[round(v, 2) for v in pa.tolist()]}")
        print(f"       mean p({args.b}) by λ: {[round(v, 2) for v in pb.tolist()]}")
    torch.save({"x": x, "lams": lams, "w": args.w, "probs": probs}, f"{args.out}/blend.pt")


if __name__ == "__main__":
    main()
