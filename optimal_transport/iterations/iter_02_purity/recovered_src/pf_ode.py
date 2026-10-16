"""Probability-flow ODE of a score_sde model: a deterministic map between images and noise.

For a forward SDE  dx = f(x,s) ds + g(s) dW  with marginals p_s, the ODE

    dx/ds = f(x, s) − ½ g(s)² ∇log p_s(x)

has the same marginals p_s. Integrating it from s≈0 to s=1 maps an image to (approximately
Gaussian) noise ("encode"); integrating back maps noise to an image ("decode"). With two models,
    T = decode_ν ∘ encode_μ
carries a μ-sample to a ν-sample (Dual Diffusion Implicit Bridges, Su et al. 2022).
"""
import torch


@torch.no_grad()
def pf_ode(score_fn, sde, x, s_start, s_end, steps=200):
    """Heun (2nd-order) integration of the probability-flow ODE from s_start to s_end."""
    s_grid = torch.linspace(s_start, s_end, steps + 1, dtype=torch.float64)

    def velocity(x, s):
        s_vec = torch.full((x.shape[0],), float(s), device=x.device)
        drift, diffusion = sde.sde(x, s_vec)
        return drift - 0.5 * (diffusion ** 2)[:, None, None, None] * score_fn(x, s_vec)

    for k in range(steps):
        s0, s1 = s_grid[k].item(), s_grid[k + 1].item()
        h = s1 - s0
        v0 = velocity(x, s0)
        x_pred = x + h * v0
        x = x + 0.5 * h * (v0 + velocity(x_pred, s1))
    return x


def encode(score_fn, sde, x, eps=1e-3, steps=200):
    return pf_ode(score_fn, sde, x, eps, sde.T, steps)


def decode(score_fn, sde, z, eps=1e-3, steps=200):
    return pf_ode(score_fn, sde, z, sde.T, eps, steps)


def transport(score_from, score_to, sde, x, eps=1e-3, steps=200):
    """T(x) = decode_to(encode_from(x)); also returns the shared latent."""
    z = encode(score_from, sde, x, eps, steps)
    return decode(score_to, sde, z, eps, steps), z
