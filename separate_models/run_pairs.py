"""The coupling from the math, with two separately trained unconditional models.

    μ = law of the cat model's samples,   ν = law of the dog model's samples
    B = Brownian motion on [0,1],          B̂_t = B_1 − B_{1−t}  (its time reversal)

    X = Föllmer_μ(B)     the cat model driven by B      (adapted to the forward filtration)
    Y = Föllmer_ν(B̂)    the dog model driven by B̂     (adapted to the reversed filtration)

(X, Y) is the coupling in Courtade–Fathi–Mikulincer (arXiv:2407.09465); its cost satisfies
    W₂²(μ, ν) ≤ E|X − Y|² ≤ 2D(μ‖γ) + 2D(ν‖γ)        (μ centered).
For comparison the dog model is also driven by the same forward path B, by −B, and by an
independent path. No classifier and no labels are used anywhere.

    python separate_models/run_pairs.py --n 16 --steps 400

Outputs (separate_models/results/pairs/): pairs_big.png, summary.json, viewer.html (time slider).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "classifier_guided"))

import argparse
import json
import time

import torch

from follmer import BrownianPath, StackedPath, follmer_sample, symmetric_grid
from make_viewer import build_html
from score_sde_backend import ScoreSDEDenoiser, get_device, load_score_model

DRIVERS = ["reversed", "same", "antithetic", "independent"]


def empirical_w2(X, Y):
    from scipy.optimize import linear_sum_assignment
    C = torch.cdist(X.flatten(1).double(), Y.flatten(1).double()).pow(2).numpy()
    r, c = linear_sum_assignment(C)
    return float(C[r, c].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="score_sde_pytorch")
    p.add_argument("--config", default="vp/cifar10_ddpmpp_continuous")
    p.add_argument("--mu", default="separate_models/checkpoints/cat.pth", help="model for μ (driven by B)")
    p.add_argument("--nu", default="separate_models/checkpoints/dog.pth", help="model for ν (driven by B̂ etc.)")
    p.add_argument("--names", nargs=2, default=["cat", "dog"])
    p.add_argument("--n", type=int, default=16, help="Brownian paths")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--frames", type=int, default=80, help="time frames kept for the viewer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="separate_models/results/pairs")
    args = p.parse_args()

    device = get_device()
    config, sde, eps, score_mu = load_score_model(args.repo, args.config, args.mu, device)
    _, _, _, score_nu = load_score_model(args.repo, args.config, args.nu, device)
    den_mu, den_nu = ScoreSDEDenoiser(sde, eps, score_mu), ScoreSDEDenoiser(sde, eps, score_nu)
    shape = (config.data.num_channels, config.data.image_size, config.data.image_size)
    grid = symmetric_grid(args.steps)
    ts = [j / args.frames for j in range(1, args.frames)]
    a, b = args.names
    print(f"device={device}  μ={args.mu}  ν={args.nu}  paths={args.n}  steps={args.steps}", flush=True)

    X, EX, Y, EY = [], [], {k: [] for k in DRIVERS}, {k: [] for k in DRIVERS}
    guess, state = [], []          # viewer frames, rows: [μ|B, ν|B, ν|B̂]
    for b0 in range(0, args.n, args.batch):
        bs = min(args.batch, args.n - b0)
        B = BrownianPath(grid, (bs,) + shape, seed=args.seed * 1_000_003 + b0)
        drivers = {"reversed": B.reversed(), "same": B, "antithetic": B.negated(),
                   "independent": BrownianPath(grid, (bs,) + shape, seed=args.seed * 1_000_003 + b0 + 500_000)}
        t0 = time.time()
        rx = follmer_sample(den_mu, B, device, snapshot_ts=ts)
        ry = follmer_sample(den_nu, StackedPath([drivers[k] for k in DRIVERS]), device, snapshot_ts=ts)
        X.append(rx.x1); EX.append(rx.energy)
        for i, k in enumerate(DRIVERS):
            Y[k].append(ry.x1.split(bs)[i]); EY[k].append(ry.energy.split(bs)[i])
        i_rev, i_same = DRIVERS.index("reversed"), DRIVERS.index("same")
        rows = lambda fx, fy: torch.stack([fx, fy.split(bs)[i_same], fy.split(bs)[i_rev]])
        guess.append(torch.stack([rows(rx.snapshots[t], ry.snapshots[t]) for t in ts] + [rows(rx.x1, ry.x1)]))
        state.append(torch.stack([rows(rx.states[t], ry.states[t]) for t in ts] + [rows(rx.x1, ry.x1)]))
        print(f"paths {b0 + bs}/{args.n} done in {time.time() - t0:.0f}s", flush=True)

    X = torch.cat(X)
    Y = {k: torch.cat(v) for k, v in Y.items()}
    cost = {k: (X - Y[k]).flatten(1).pow(2).sum(1) for k in DRIVERS}
    D_mu, D_nu = torch.cat(EX).mean().item(), torch.cat(EY["reversed"]).mean().item()
    summary = {"args": vars(args), "dim": X[0].numel(),
               "E|X-Y|^2": {k: cost[k].mean().item() for k in DRIVERS},
               "stderr": {k: (cost[k].std() / len(X) ** 0.5).item() for k in DRIVERS},
               "empirical_OT(X, Y_independent)": empirical_w2(X, Y["independent"]),
               "D_mu_est": D_mu, "D_nu_est": D_nu, "2D_mu+2D_nu": 2 * D_mu + 2 * D_nu}

    os.makedirs(args.out, exist_ok=True)
    import torchvision
    from PIL import Image
    k = min(len(X), 16)
    grid_img = torch.cat([X[:k], Y["reversed"][:k], Y["same"][:k]])
    torchvision.utils.save_image(((grid_img + 1) / 2).clamp(0, 1), f"{args.out}/pairs.png", nrow=k, padding=1)
    im = Image.open(f"{args.out}/pairs.png")
    im.resize((im.width * 4, im.height * 4), Image.NEAREST).save(f"{args.out}/pairs_big.png")
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({"X": X, "Y": Y, "summary": summary}, f"{args.out}/samples.pt")
    torch.save({"guess": torch.cat(guess, 2), "state": torch.cat(state, 2), "ts": ts + [1.0],
                "steps": args.steps, "centered": config.data.centered}, f"{args.out}/frames.pt")
    build_html(args.out, runs=[f"{a} model · driven by B", f"{b} model · driven by the same B", f"{b} model · driven by reversed B̂"],
               note=f"{len(X)} Brownian paths, {args.steps} Euler steps. Two separate unconditional models "
                    f"({a}-only and {b}-only), no classifier.")

    print(f"\npairs_big.png rows: {a}(B), {b}(B̂), {b}(B)")
    print(f"energy ½∫|v|² ≈ D(·‖γ):  {a}: {D_mu:.0f}   {b}: {D_nu:.0f}   2D+2D ≈ {2 * D_mu + 2 * D_nu:.0f}")
    print(f"finite-sample OT cost (n={len(X)}): {summary['empirical_OT(X, Y_independent)']:.0f}")
    for k_ in DRIVERS:
        print(f"  E|X-Y|^2  {k_:12s} = {summary['E|X-Y|^2'][k_]:8.0f} ± {summary['stderr'][k_]:.0f}")


if __name__ == "__main__":
    main()
