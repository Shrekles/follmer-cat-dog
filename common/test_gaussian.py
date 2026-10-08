"""Sanity check on the extremal case of the paper: μ = N(0, αI), ν = N(0, I/α).

Theory (per dimension):
  W2²(μ,ν) = 2D(μ‖γ) + 2D(ν‖γ) = α + 1/α − 2       (equality case of symmetrized Talagrand)
  X = Föllmer_μ(B),  Y = Föllmer_ν(B̂)   ⇒  Y = X/α exactly, so E|X−Y|² = W2²
  linear couplings B² = σB¹ + ...       ⇒  E|X−Y|² = α + 1/α − 4σ log α / (α − 1/α)
"""
import argparse
import math

import torch

from follmer import BrownianPath, GaussianDenoiser, StackedPath, follmer_sample, symmetric_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--steps", type=int, default=1000)
    args = p.parse_args()

    a, d, n = args.alpha, args.dim, args.n
    grid = symmetric_grid(args.steps)
    B = BrownianPath(grid, (n, d), seed=0)
    paths = {"X ~ mu, driven by B": B,
             "Y ~ nu, reversed  (B^)": B.reversed(),
             "Y ~ nu, same      (B)": B,
             "Y ~ nu, antithetic(-B)": B.negated(),
             "Y ~ nu, independent": BrownianPath(grid, (n, d), seed=1)}
    names = list(paths)
    labels = torch.tensor([0] * n + [1] * (4 * n))
    res = follmer_sample(GaussianDenoiser([a, 1 / a]), StackedPath(list(paths.values())), "cpu", labels=labels)
    out = dict(zip(names, res.x1.split(n)))
    en = dict(zip(names, res.energy.split(n)))

    X = out[names[0]]
    D = lambda s: 0.5 * (s - 1 - math.log(s))
    r = 4 * math.log(a) / (a - 1 / a)
    theory = {"reversed": a + 1 / a - 2, "same": a + 1 / a - r, "antithetic": a + 1 / a + r, "independent": a + 1 / a}

    print(f"alpha={a}, dim={d}, n={n}, steps={args.steps}")
    print(f"Var(X)/dim = {X.var(0).mean():.4f}  (target {a:.4f})")
    print(f"energy_X/dim = {en[names[0]].mean() / d:.4f}   D(mu||g)/dim = {D(a):.4f}")
    print(f"energy_Y/dim = {en[names[1]].mean() / d:.4f}   D(nu||g)/dim = {D(1 / a):.4f}")
    print(f"2D(mu)+2D(nu) per dim = W2^2 per dim = {2 * D(a) + 2 * D(1 / a):.4f}\n")
    print(f"{'coupling':28s} {'E|X-Y|^2/dim':>14s} {'theory':>10s}")
    for name, key in zip(names[1:], theory):
        Y = out[name]
        print(f"{name:28s} {(X - Y).pow(2).sum(1).mean() / d:14.4f} {theory[key]:10.4f}")
    print(f"\nmax |Y_reversed - X/alpha| = {(out[names[1]] - X / a).abs().max():.2e}  (theory: 0)")


if __name__ == "__main__":
    main()
