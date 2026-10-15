"""Forward and backward, cats and dogs, all on the same Brownian sample paths.

For each path B (and its time reversal B̂_t = B_1 − B_{1−t}):

    forward cat   = cat model driven by B        backward cat  = cat model driven by B̂
    forward dog   = dog model driven by B        backward dog  = dog model driven by B̂

The coupling in the proof is (forward cat, backward dog); by symmetry (backward cat, forward dog)
is the same coupling with the roles of B and B̂ swapped. The script reports E|X − Y|² for every
pairing of the four, and saves the four final images per path and their evolution over time.

    python separate_models/run_fwd_bwd.py --n 16 --steps 400
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import argparse
import itertools
import json
import time

import torch
import torchvision
from PIL import Image

from follmer import BrownianPath, StackedPath, follmer_sample, symmetric_grid
from score_sde_backend import ScoreSDEDenoiser, get_device, load_score_model

ROWS = ["fwd cat", "fwd dog", "bwd cat", "bwd dog"]
WANT = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def save_big(t, path, nrow, scale):
    torchvision.utils.save_image(((t + 1) / 2).clamp(0, 1), path, nrow=nrow, padding=1)
    im = Image.open(path)
    im.resize((im.width * scale, im.height * scale), Image.NEAREST).save(path.replace(".png", "_big.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="score_sde_pytorch")
    p.add_argument("--config", default="vp/cifar10_ddpmpp_continuous")
    p.add_argument("--cat", default="separate_models/checkpoints/cat.pth")
    p.add_argument("--dog", default="separate_models/checkpoints/dog.pth")
    p.add_argument("--n", type=int, default=16, help="Brownian paths")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="separate_models/results/fwd_bwd")
    args = p.parse_args()

    device = get_device()
    config, sde, eps, s_cat = load_score_model(args.repo, args.config, args.cat, device)
    _, _, _, s_dog = load_score_model(args.repo, args.config, args.dog, device)
    den = {"cat": ScoreSDEDenoiser(sde, eps, s_cat), "dog": ScoreSDEDenoiser(sde, eps, s_dog)}
    shape = (config.data.num_channels, config.data.image_size, config.data.image_size)
    grid = symmetric_grid(args.steps)
    print(f"device={device}  paths={args.n}  steps={args.steps}", flush=True)

    final = {r: [] for r in ROWS}
    traj = {r: [] for r in ROWS}           # each: (len(WANT)+1, bs, C, H, W)
    for b0 in range(0, args.n, args.batch):
        bs = min(args.batch, args.n - b0)
        B = BrownianPath(grid, (bs,) + shape, seed=args.seed * 1_000_003 + b0)   # same seeds as run_pairs.py
        t0 = time.time()
        for animal in ("cat", "dog"):
            res = follmer_sample(den[animal], StackedPath([B, B.reversed()]), device, snapshot_ts=WANT)
            for i, direction in enumerate(("fwd", "bwd")):
                r = f"{direction} {animal}"
                x1 = res.x1.split(bs)[i]
                final[r].append(x1)
                traj[r].append(torch.stack([res.snapshots[t].split(bs)[i] for t in WANT] + [x1]))
        print(f"paths {b0 + bs}/{args.n} done in {time.time() - t0:.0f}s", flush=True)

    final = {r: torch.cat(v) for r, v in final.items()}
    traj = {r: torch.cat(v, 1) for r, v in traj.items()}
    n = args.n
    cost = lambda a, b: (final[a] - final[b]).flatten(1).pow(2).sum(1)
    table = {}
    for a, b in itertools.combinations(ROWS, 2):
        c = cost(a, b)
        table[f"{a} | {b}"] = {"mean": c.mean().item(), "stderr": (c.std() / n ** 0.5).item()}

    os.makedirs(args.out, exist_ok=True)
    k = min(n, 16)
    save_big(torch.cat([final[r][:k] for r in ROWS]), f"{args.out}/four_rows.png", k, 4)
    # one block per path: 4 rows over time
    for name, paths in (("paths_1-8", range(0, min(8, n))), ("paths_9-16", range(8, min(16, n)))):
        if not len(paths):
            continue
        blocks = [torchvision.utils.make_grid(((torch.cat([traj[r][:, i] for r in ROWS]) + 1) / 2).clamp(0, 1),
                                              nrow=len(WANT) + 1, padding=1) for i in paths]
        H, W = blocks[0].shape[1:]
        canvas = torch.ones(3, (H + 6) * len(blocks) - 6, W)
        for j, b in enumerate(blocks):
            canvas[:, j * (H + 6): j * (H + 6) + H] = b
        torchvision.utils.save_image(canvas, f"{args.out}/{name}.png")
        im = Image.open(f"{args.out}/{name}.png")
        im.resize((im.width * 3, im.height * 3), Image.NEAREST).save(f"{args.out}/{name}_big.png")
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump({"args": vars(args), "rows": ROWS, "E|X-Y|^2": table}, f, indent=2)
    torch.save({"final": final, "traj": traj, "ts": WANT + [1.0]}, f"{args.out}/samples.pt")

    print(f"\nrows of four_rows.png: {ROWS}")
    for key, v in table.items():
        print(f"  {key:22s} E|X-Y|^2 = {v['mean']:7.0f} ± {v['stderr']:.0f}")


if __name__ == "__main__":
    main()
