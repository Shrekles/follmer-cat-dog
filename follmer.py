"""Föllmer-process sampler driven by an explicit Brownian path.

Setting (Courtade–Fathi–Mikulincer, arXiv:2407.09465): on Wiener space over [0,1],
B is a standard Brownian motion and  B̂_t := B_1 - B_{1-t}  is its time reversal.
For a target law μ, the Föllmer process

    dX_t = v(X_t, t) dt + dB_t,   X_0 = 0,   v(x, t) = (E[X_1 | X_t = x] - x) / (1 - t)

has X_1 ~ μ, and D(μ‖γ) = ½ E ∫_0^1 |v(X_t, t)|² dt. So a "sampler" is just a
deterministic map  ω ↦ X_1(ω)  from Brownian paths to samples. Everything that
the learned model provides is the denoiser  x̂_1(x, t) = E[X_1 | X_t = x].

A Brownian path is represented by its increments on a time grid that is symmetric
about t = 1/2, so that the increments of B̂ on the same grid are exactly the
increments of B read backwards. The time reversal is then exact at the discrete level.
"""
import math
from dataclasses import dataclass, field

import torch


def symmetric_grid(n_steps: int) -> torch.Tensor:
    """t_k = (1 - cos(pi k / N)) / 2. Dense near both ends, and t_{N-k} = 1 - t_k."""
    k = torch.arange(n_steps + 1, dtype=torch.float64)
    t = (1 - torch.cos(math.pi * k / n_steps)) / 2
    t[0], t[-1] = 0.0, 1.0
    return t


class BrownianPath:
    """Increments ΔB_k = B_{t_{k+1}} - B_{t_k} on a symmetric grid, generated lazily.

    Increment k is regenerated from a per-(seed, k) generator, so the whole path never
    has to be held in memory and `reversed()` is just re-indexing k -> N-1-k.
    """

    def __init__(self, grid, shape, seed, sign=1.0, reverse=False):
        self.grid = grid
        self.dt = (grid[1:] - grid[:-1])
        self.N = len(self.dt)
        self.shape = tuple(shape)
        self.seed = int(seed)
        self.sign = float(sign)
        self.reverse = reverse
        assert torch.allclose(self.dt, self.dt.flip(0)), "grid must be symmetric for exact time reversal"

    def increment(self, k: int) -> torch.Tensor:
        j = self.N - 1 - k if self.reverse else k
        g = torch.Generator().manual_seed((self.seed << 20) + j)
        z = torch.randn(self.shape, generator=g)
        return self.sign * math.sqrt(self.dt[j].item()) * z

    def reversed(self):
        """B̂_t = B_1 - B_{1-t}: same ω, increments read in reverse order."""
        return BrownianPath(self.grid, self.shape, self.seed, self.sign, not self.reverse)

    def negated(self):
        """-B (the antithetic / σ = -1 linear coupling)."""
        return BrownianPath(self.grid, self.shape, self.seed, -self.sign, self.reverse)

    def endpoint(self) -> torch.Tensor:
        return sum(self.increment(k) for k in range(self.N))


class StackedPath:
    """Several paths concatenated along the batch dimension, integrated in one pass."""

    def __init__(self, paths):
        self.paths = paths
        self.grid = paths[0].grid
        self.N = paths[0].N
        self.sizes = [p.shape[0] for p in paths]

    def increment(self, k):
        return torch.cat([p.increment(k) for p in self.paths], 0)


@dataclass
class FollmerResult:
    x1: torch.Tensor                 # samples X_1, shape (B, ...)
    energy: torch.Tensor             # ½ ∫ |v|² dt per sample  (estimates D(μ‖γ) in expectation)
    energy_no_last: torch.Tensor     # same, excluding the final (singular) step
    snapshots: dict = field(default_factory=dict)  # t -> x̂_1(X_t, t)


@torch.no_grad()
def follmer_sample(denoiser, path, device, labels=None, snapshot_ts=(), clip=None, verbose=False):
    """Euler–Maruyama for dX = v dt + dB driven by `path`.

    denoiser(x, t: float, labels) -> E[X_1 | X_t = x].
    The last step has Δt = 1 - t_{N-1}, so it lands exactly on x̂_1 (+ the last tiny ΔB):
    this is the usual "final denoising step", obtained here for free.
    """
    grid = path.grid
    shape = (sum(path.sizes),) + path.paths[0].shape[1:] if isinstance(path, StackedPath) else path.shape
    x = torch.zeros(shape, device=device)
    energy = torch.zeros(shape[0], dtype=torch.float64)  # on CPU: MPS has no float64
    energy_nl = torch.zeros_like(energy)
    snaps, pending = {}, sorted(snapshot_ts)
    for k in range(path.N):
        t, dt = grid[k].item(), (grid[k + 1] - grid[k]).item()
        x1_hat = denoiser(x, t, labels)
        if clip is not None:
            x1_hat = x1_hat.clamp(-clip, clip)
        if pending and t >= pending[0]:
            snaps[pending.pop(0)] = x1_hat.detach().cpu()
        v = (x1_hat - x) / (1.0 - t)
        e = 0.5 * dt * v.flatten(1).pow(2).sum(1).cpu().double()
        energy += e
        if k < path.N - 1:
            energy_nl += e
        x = x + v * dt + path.increment(k).to(device)
        if verbose and (k % max(1, path.N // 10) == 0):
            print(f"  step {k}/{path.N}  t={t:.4f}")
    return FollmerResult(x.cpu(), energy, energy_nl, snaps)


class GaussianDenoiser:
    """Exact denoiser for μ = N(0, a·I), with a per-label variance a.

    X_t = t X_1 + sqrt(t(1-t)) Z  ⇒  E[X_1 | X_t = x] = a x / (t a + 1 - t).
    """

    def __init__(self, variances):
        self.variances = torch.as_tensor(variances, dtype=torch.float32)

    def __call__(self, x, t, labels):
        a = self.variances.to(x.device)[labels].view(-1, *([1] * (x.dim() - 1)))
        return a * x / (t * a + 1 - t)
