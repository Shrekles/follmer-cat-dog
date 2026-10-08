"""Cat/dog Föllmer coupling experiment with a pretrained score_sde_pytorch model.

For each Brownian path ω = B on [0,1]:
    X = Föllmer_cat(B)                                          (μ = cats)
    Y = Föllmer_dog(·) driven by one of
        reversed    : B̂_t = B_1 − B_{1−t}   (time-reversal coupling of arXiv:2407.09465)
        same        : B                      (synchronous, σ = 1)
        antithetic  : −B                     (σ = −1)
        independent : an independent B'      (σ = 0)
and we report E|X − Y|² for each coupling. We also report the path energies ½∫|v|²dt
(D(μ‖γ) estimates) and a finite-sample OT cost between the cat and dog sample clouds.

All runs for one path go through the network in one batch: the score model is
unconditional, and only the classifier-guidance term depends on the class.

    python classifier_guided/run_cat_dog.py --ckpt checkpoints/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth \
        --classifier classifier_guided/checkpoints/cifar10_noisy_classifier.pt --n 16 --steps 1000
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import argparse
import json
import time

import torch

from follmer import BrownianPath, StackedPath, follmer_sample, symmetric_grid
from noisy_classifier import load_classifier
from score_sde_backend import ScoreSDEDenoiser, get_device, load_score_model

CIFAR = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def save_grid(rows, path, centered):
    import torchvision
    imgs = torch.cat(rows, 0)
    imgs = (imgs + 1) / 2 if centered else imgs
    torchvision.utils.save_image(imgs.clamp(0, 1), path, nrow=rows[0].shape[0], padding=1)


def empirical_w2(X, Y):
    """Exact OT between the two n-point empirical measures (a biased, finite-sample proxy for W2²)."""
    from scipy.optimize import linear_sum_assignment
    C = torch.cdist(X.flatten(1).double(), Y.flatten(1).double()).pow(2).numpy()
    r, c = linear_sum_assignment(C)
    return float(C[r, c].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="score_sde_pytorch")
    p.add_argument("--config", default="vp/cifar10_ddpmpp_continuous")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--classifier", required=True)
    p.add_argument("--mu", default="cat", choices=CIFAR)
    p.add_argument("--nu", default="dog", choices=CIFAR)
    p.add_argument("--guidance", type=float, default=1.0, help="1.0 = exact Bayes conditional score")
    p.add_argument("--couplings", nargs="+", default=["reversed", "same", "antithetic", "independent"])
    p.add_argument("--n", type=int, default=16, help="number of Brownian paths")
    p.add_argument("--batch", type=int, default=8, help="paths per network batch (x (1+#couplings))")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clip", type=float, default=None, help="clamp x̂_1 to [-clip, clip] (breaks exactness)")
    p.add_argument("--out", default="classifier_guided/results/cat_dog")
    args = p.parse_args()

    device = get_device()
    config, sde, eps, score_fn = load_score_model(args.repo, args.config, args.ckpt, device)
    clf = load_classifier(args.classifier, device)
    den = ScoreSDEDenoiser(sde, eps, score_fn, classifier=clf, guidance=args.guidance)
    shape = (config.data.num_channels, config.data.image_size, config.data.image_size)
    grid = symmetric_grid(args.steps)
    mu_c, nu_c = CIFAR.index(args.mu), CIFAR.index(args.nu)
    print(f"device={device}  {args.mu}(={mu_c}) vs {args.nu}(={nu_c})  steps={args.steps}  "
          f"t-range usable by model: [{den.snr.t_lo:.2e}, 1-{1 - den.snr.t_hi:.1e}]")

    X_all, Y_all = [], {c: [] for c in args.couplings}
    E_X, E_Y = [], {c: [] for c in args.couplings}
    snaps_all = []
    for b0 in range(0, args.n, args.batch):
        bs = min(args.batch, args.n - b0)
        B = BrownianPath(grid, (bs,) + shape, seed=args.seed * 1_000_003 + b0)
        drivers = {"reversed": B.reversed(), "same": B, "antithetic": B.negated(),
                   "independent": BrownianPath(grid, (bs,) + shape, seed=args.seed * 1_000_003 + b0 + 500_000)}
        paths = [B] + [drivers[c] for c in args.couplings]
        labels = torch.tensor([mu_c] * bs + [nu_c] * (bs * len(args.couplings)))
        t0 = time.time()
        res = follmer_sample(den, StackedPath(paths), device, labels=labels, clip=args.clip,
                             snapshot_ts=(0.25, 0.5, 0.75, 0.9), verbose=True)
        xs, es = res.x1.split(bs), res.energy.split(bs)
        X_all.append(xs[0]); E_X.append(es[0])
        for i, c in enumerate(args.couplings):
            Y_all[c].append(xs[i + 1]); E_Y[c].append(es[i + 1])
        snaps_all.append({t: v.split(bs) for t, v in res.snapshots.items()})
        print(f"paths {b0 + bs}/{args.n} done in {time.time() - t0:.0f}s")

    X = torch.cat(X_all)
    Y = {c: torch.cat(v) for c, v in Y_all.items()}
    d = X[0].numel()
    summary = {"args": vars(args), "dim": d,
               "D_mu_est": torch.cat(E_X).mean().item(),
               "D_nu_est": {c: torch.cat(v).mean().item() for c, v in E_Y.items()},
               "E|X-Y|^2": {c: (X - Y[c]).flatten(1).pow(2).sum(1).mean().item() for c in args.couplings},
               "E|X-Y|^2_stderr": {c: ((X - Y[c]).flatten(1).pow(2).sum(1).std() / len(X) ** 0.5).item() for c in args.couplings}}
    ref = Y.get("independent", Y[args.couplings[0]])
    summary["empirical_OT(X, Y_independent)"] = empirical_w2(X, ref)
    summary["mean_X"], summary["mean_Y"] = X.mean().item(), ref.mean().item()

    os.makedirs(args.out, exist_ok=True)
    k = min(len(X), 16)
    save_grid([X[:k]] + [Y[c][:k] for c in args.couplings], f"{args.out}/samples.png", config.data.centered)
    # trajectory of the denoised prediction x̂_1(X_t, t): first path batch, cat(B) vs dog(B̂)
    if "reversed" in args.couplings:
        s0 = snaps_all[0]
        j = 1 + args.couplings.index("reversed")
        ts = sorted(s0)
        cat_row = torch.cat([s0[t][0][:4] for t in ts] + [X[:4]])
        dog_row = torch.cat([s0[t][j][:4] for t in ts] + [Y["reversed"][:4]])
        save_grid([cat_row, dog_row], f"{args.out}/trajectory_reversed.png", config.data.centered)
    torch.save({"X": X, "Y": Y, "summary": summary}, f"{args.out}/samples.pt")
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nrows of samples.png: {args.mu}(B), then {args.nu} under {args.couplings}")
    print(f"dim = {d}")
    print(f"energy ½∫|v|² (≈ D(·‖γ)):  {args.mu}: {summary['D_mu_est']:.1f}   "
          f"{args.nu}: {summary['D_nu_est'][args.couplings[0]]:.1f}")
    print(f"2D(mu)+2D(nu) ≈ {2 * summary['D_mu_est'] + 2 * summary['D_nu_est'][args.couplings[0]]:.1f}")
    print(f"finite-sample OT cost (n={len(X)}): {summary['empirical_OT(X, Y_independent)']:.1f}")
    for c in args.couplings:
        print(f"  E|X-Y|^2  {c:12s} = {summary['E|X-Y|^2'][c]:9.1f}  ± {summary['E|X-Y|^2_stderr'][c]:.1f}")


if __name__ == "__main__":
    main()
