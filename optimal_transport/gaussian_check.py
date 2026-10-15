"""Do the composed flows give the Brenier map? Check on Gaussians, where everything is exact.

μ = N(0, C1), ν = N(0, C2) in d dimensions. Closed forms:
    W2² = tr C1 + tr C2 − 2 tr (C1^½ C2 C1^½)^½
    Brenier map  T*(x) = C1^-½ (C1^½ C2 C1^½)^½ C1^-½ x
The composed map T = decode_ν ∘ encode_μ uses exact scores, so any gap from T* is the method's,
not a learning error. For context the script also runs the two Föllmer couplings (cat on B with
dog on B̂, and both on the same B) with exact drifts.

Cases: (a) C2 = C1⁻¹, the equality case of the symmetrized Talagrand inequality;
       (b) C2 commutes with C1;   (c) C2 does not commute with C1.

    python optimal_transport/gaussian_check.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)

import torch

from follmer import BrownianPath, follmer_sample, symmetric_grid
from pf_ode import transport
from score_sde_backend import import_score_sde

torch.set_default_dtype(torch.float64)


def msqrt(A, inv=False):
    w, V = torch.linalg.eigh(A)
    return V @ torch.diag(w ** (-0.5 if inv else 0.5)) @ V.T


def random_spd(d, lo, hi, g):
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    return Q @ torch.diag(torch.linspace(lo, hi, d)) @ Q.T


def vp_gaussian_score(C, sde):
    I = torch.eye(C.shape[0])

    def score(x, s):  # x: (n, d, 1, 1); data N(0, C) under the VP SDE has p_s = N(0, m²C + σ²I)
        m, sd = sde.marginal_prob(torch.ones(1, 1, 1, 1), s[:1].double())
        P = torch.linalg.inv(m.item() ** 2 * C + sd.item() ** 2 * I)
        return -(x[:, :, 0, 0] @ P)[:, :, None, None]
    return score


def follmer_gaussian(C):
    I = torch.eye(C.shape[0])

    def den(x, t, labels=None):  # E[X1 | Xt = x] for X1 ~ N(0, C)
        return (x[:, :, 0, 0] @ (C @ torch.linalg.inv(t * C + (1 - t) * I)).T)[:, :, None, None]
    return den


def main():
    d, n = 16, 4000
    sde_lib, _ = import_score_sde(os.path.join(HERE, "..", "score_sde_pytorch"))
    sde = sde_lib.VPSDE(beta_min=0.1, beta_max=20, N=1000)
    g = torch.Generator().manual_seed(0)
    C1 = random_spd(d, 0.3, 3.0, g)
    w, V = torch.linalg.eigh(C1)
    cases = {
        "(a) C2 = C1^-1 (equality case)": torch.linalg.inv(C1),
        "(b) C2 commutes with C1": V @ torch.diag(torch.linspace(2.5, 0.4, d)) @ V.T,
        "(c) C2 does not commute": random_spd(d, 0.4, 2.5, g),
    }
    x = torch.randn(n, d, generator=g) @ msqrt(C1).T
    grid = symmetric_grid(400)
    B = BrownianPath(grid, (n, d, 1, 1), seed=1)
    X_follmer = follmer_sample(follmer_gaussian(C1), B, "cpu").x1[:, :, 0, 0].double()

    print(f"d={d}, n={n}.  All costs are E|X−Y|² per dimension.\n")
    print(f"{'case':34s} {'W2² (exact)':>11s} {'composed flows':>15s} {'|T−T*|²':>9s} {'Föllmer B vs B̂':>16s} {'Föllmer same B':>15s}")
    for name, C2 in cases.items():
        r = msqrt(C1)
        w2 = (torch.trace(C1) + torch.trace(C2) - 2 * torch.trace(msqrt(r @ C2 @ r))).item() / d
        T_star = msqrt(C1, inv=True) @ msqrt(r @ C2 @ r) @ msqrt(C1, inv=True)
        y, _ = transport(vp_gaussian_score(C1, sde), vp_gaussian_score(C2, sde), sde, x[:, :, None, None], steps=400)
        y = y[:, :, 0, 0]
        cost = (x - y).pow(2).sum(1).mean().item() / d
        gap = (y - x @ T_star.T).pow(2).sum(1).mean().item() / d
        Yr = follmer_sample(follmer_gaussian(C2), B.reversed(), "cpu").x1[:, :, 0, 0].double()
        Ys = follmer_sample(follmer_gaussian(C2), B, "cpu").x1[:, :, 0, 0].double()
        rev = (X_follmer - Yr).pow(2).sum(1).mean().item() / d
        same = (X_follmer - Ys).pow(2).sum(1).mean().item() / d
        print(f"{name:34s} {w2:11.4f} {cost:15.4f} {gap:9.4f} {rev:16.4f} {same:15.4f}")
    print("\n|T−T*|² is the mean squared distance between the composed map and the exact Brenier map.")


if __name__ == "__main__":
    main()
