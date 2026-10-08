"""One Brownian path, three processes, watched over time.

    row 1: cat  driven by B            (forward time)
    row 2: dog  driven by the same B   (forward time)
    row 3: dog  driven by B̂_t = B_1 − B_{1−t}   (time-reversed path)

Columns show the model's current guess of the final image, x̂_1(X_t, t) = E[X_1 | X_t], at
increasing t, and the last column is the sample X_1. Several candidate paths are run; one
figure is saved per path, and the path whose three endpoints the classifier finds most clearly
cat / dog / dog is also saved as `best.png`. That is a selection step, so the figure is
illustrative, not a random draw.

    python classifier_guided/single_path.py --ckpt checkpoints/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth \
        --classifier classifier_guided/checkpoints/cifar10_noisy_classifier.pt --w 6 --n 8 --steps 400
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import argparse
import time

import torch
import torch.nn.functional as F

from follmer import BrownianPath, StackedPath, follmer_sample, symmetric_grid
from noisy_classifier import load_classifier
from run_cat_dog import CIFAR, save_grid
from score_sde_backend import ScoreSDEDenoiser, get_device, load_score_model

TS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="score_sde_pytorch")
    p.add_argument("--config", default="vp/cifar10_ddpmpp_continuous")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--classifier", required=True)
    p.add_argument("--w", type=float, default=6.0)
    p.add_argument("--n", type=int, default=8, help="candidate paths")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--out", default="classifier_guided/results/single_path")
    args = p.parse_args()

    device = get_device()
    config, sde, eps, score_fn = load_score_model(args.repo, args.config, args.ckpt, device)
    clf = load_classifier(args.classifier, device)
    den = ScoreSDEDenoiser(sde, eps, score_fn, classifier=clf, guidance=args.w)
    shape = (config.data.num_channels, config.data.image_size, config.data.image_size)
    cat, dog = CIFAR.index("cat"), CIFAR.index("dog")

    B = BrownianPath(symmetric_grid(args.steps), (args.n,) + shape, seed=args.seed)
    runs = [("cat | B", B, cat), ("dog | B", B, dog), ("dog | B^", B.reversed(), dog)]
    labels = torch.cat([torch.full((args.n,), c) for _, _, c in runs])
    t0 = time.time()
    res = follmer_sample(den, StackedPath([r[1] for r in runs]), device, labels=labels, snapshot_ts=TS, verbose=True)
    print(f"done in {time.time() - t0:.0f}s")

    # frames[r][j] : (n, C, H, W) for run r at column j (snapshots, then the final sample)
    frames = [[res.snapshots[t].split(args.n)[r] for t in TS] + [res.x1.split(args.n)[r]] for r in range(3)]
    finals = [f[-1] for f in frames]
    with torch.no_grad():
        s = torch.full((args.n,), eps, device=device)
        pr = [F.softmax(clf(x.to(device), s), -1).cpu() for x in finals]
    score = torch.stack([pr[0][:, cat], pr[1][:, dog], pr[2][:, dog]]).min(0).values

    os.makedirs(args.out, exist_ok=True)
    dist = lambda a, b: (a - b).flatten(1).pow(2).sum(1)
    for i in range(args.n):
        rows = [torch.stack([f[i] for f in frames[r]]) for r in range(3)]
        save_grid(rows, f"{args.out}/path_{i}.png", config.data.centered)
        print(f"path {i}: p(cat)={pr[0][i, cat]:.2f} p(dog|B)={pr[1][i, dog]:.2f} p(dog|B^)={pr[2][i, dog]:.2f}   "
              f"|cat−dog(B)|²={dist(finals[0], finals[1])[i]:.0f}  |cat−dog(B^)|²={dist(finals[0], finals[2])[i]:.0f}")
    best = int(score.argmax())
    rows = [torch.stack([f[best] for f in frames[r]]) for r in range(3)]
    save_grid(rows, f"{args.out}/best.png", config.data.centered)
    print(f"best path: {best}  (columns: t = {list(TS)} then X_1)")
    torch.save({"frames": frames, "ts": TS, "probs": pr, "w": args.w}, f"{args.out}/single_path.pt")


if __name__ == "__main__":
    main()
