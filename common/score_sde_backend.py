"""Turn a pretrained score_sde_pytorch model into a Föllmer denoiser x̂_1(x, t).

Change of variables. score_sde's forward SDE has marginals  y_s = m(s) x_0 + σ(s) z,
while the Föllmer process has  X_t = t X_1 + sqrt(t(1-t)) z. They carry the same
information when the signal-to-noise ratios match:

    m(s)² / σ(s)² = t / (1 - t),   and then   y_s = (m(s) / t) · X_t.

(For VP: ᾱ(s) = t, y = X_t / sqrt(t).) Tweedie then gives
    E[X_1 | X_t = x] = E[x_0 | y_s = y] = (y + σ(s)² ∇log p_s(y)) / m(s).

Class conditioning (cat / dog) uses classifier guidance with a time-dependent
classifier p_s(c | y):   ∇log p_s(y | c) = ∇log p_s(y) + w ∇log p_s(c | y).
w = 1 is the exact Bayes conditional score; w > 1 sharpens (and is no longer exact).
"""
import importlib
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F


def _native_upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """Pure-PyTorch upfirdn2d (replaces the CUDA extension that score_sde compiles at import)."""
    up_x = up_y = up
    down_x = down_y = down
    pad_x0, pad_x1, pad_y0, pad_y1 = pad[0], pad[1], pad[0], pad[1]
    _, channel, in_h, in_w = input.shape
    x = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = x.shape
    kernel_h, kernel_w = kernel.shape
    out = x.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w).to(out)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    return out.reshape(-1, channel, out_h, out_w)


def _stub_cuda_ops():
    """score_sde's `op` package JIT-compiles CUDA kernels on import; replace it (Mac/CPU friendly)."""
    op = types.ModuleType("op")
    op.upfirdn2d = _native_upfirdn2d

    def fused_leaky_relu(x, bias, negative_slope=0.2, scale=2 ** 0.5):
        return F.leaky_relu(x + bias.view(1, -1, *[1] * (x.dim() - 2)), negative_slope) * scale

    op.fused_leaky_relu = fused_leaky_relu
    op.FusedLeakyReLU = None
    sys.modules["op"] = op


def import_score_sde(repo):
    repo = os.path.abspath(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    _stub_cuda_ops()
    sde_lib = importlib.import_module("sde_lib")
    mutils = importlib.import_module("models.utils")
    for name in ("ncsnpp", "ddpm", "ncsnv2"):  # registers architectures
        try:
            importlib.import_module(f"models.{name}")
        except Exception as e:  # noqa: BLE001
            print(f"[score_sde] could not import models.{name}: {e}")
    return sde_lib, mutils


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_sde(sde_lib, config):
    t = config.training.sde.lower()
    m = config.model
    if t == "vpsde":
        return sde_lib.VPSDE(beta_min=m.beta_min, beta_max=m.beta_max, N=m.num_scales), 1e-5
    if t == "subvpsde":
        return sde_lib.subVPSDE(beta_min=m.beta_min, beta_max=m.beta_max, N=m.num_scales), 1e-5
    if t == "vesde":
        return sde_lib.VESDE(sigma_min=m.sigma_min, sigma_max=m.sigma_max, N=m.num_scales), 1e-5
    raise ValueError(t)


def load_score_model(repo, config_name, ckpt_path, device):
    """config_name like 'vp/cifar10_ddpmpp_continuous'. Loads EMA weights. Returns a score function."""
    config, sde, eps, model = load_model(repo, config_name, ckpt_path, device)
    _, mutils = import_score_sde(repo)
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=config.training.continuous)
    return config, sde, eps, score_fn


def load_model(repo, config_name, ckpt_path, device):
    """Same, but returns the network itself (EMA weights loaded), e.g. for fine-tuning."""
    sde_lib, mutils = import_score_sde(repo)
    config = importlib.import_module("configs." + config_name.replace("/", ".")).get_config()
    config.device = device
    # no DataParallel wrapper; .float() because MPS has no float64 (the `sigmas` buffer is float64)
    model = mutils.get_model(config.model.name)(config).float().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[score_sde] missing={missing} unexpected={unexpected}")
    # EMA weights are what the paper reports samples with.
    for p, s in zip(model.parameters(), ckpt["ema"]["shadow_params"]):
        p.data.copy_(s.to(device))
    model.eval()
    sde, eps = build_sde(sde_lib, config)
    return config, sde, eps, model


class SNRMap:
    """Maps Föllmer time t ∈ (0,1) to model time s with m(s)²/σ(s)² = t/(1-t)."""

    def __init__(self, sde, eps, n=20001):
        s = torch.linspace(eps, sde.T, n, dtype=torch.float64)
        ones = torch.ones(n, 1, 1, 1, dtype=torch.float64)  # sde_lib indexes t[:, None, None, None]
        mean, std = sde.marginal_prob(ones, s)
        m = mean.flatten()
        self.s, self.m, self.std = s.numpy(), m.numpy(), std.numpy()
        self.log_snr = np.log(m.numpy() ** 2) - np.log(std.numpy() ** 2)  # decreasing in s
        snr_lo, snr_hi = np.exp(self.log_snr[-1]), np.exp(self.log_snr[0])
        self.t_lo, self.t_hi = snr_lo / (1 + snr_lo), snr_hi / (1 + snr_hi)

    def __call__(self, t):
        t = float(np.clip(t, self.t_lo, self.t_hi))
        target = np.log(t) - np.log1p(-t)
        # np.interp needs increasing x: flip.
        s = float(np.interp(target, self.log_snr[::-1], self.s[::-1]))
        m = float(np.interp(target, self.log_snr[::-1], self.m[::-1]))
        sd = float(np.interp(target, self.log_snr[::-1], self.std[::-1]))
        return t, s, m, sd


class ScoreSDEDenoiser:
    """x̂_1(x, t, labels) for a score_sde model, optionally class-conditioned by a noisy classifier.

    labels: LongTensor of class ids per sample, or -1 for unconditional.
    """

    def __init__(self, sde, eps, score_fn, classifier=None, guidance=1.0):
        self.sde, self.score_fn = sde, score_fn
        self.snr = SNRMap(sde, eps)
        self.classifier, self.guidance = classifier, guidance

    def __call__(self, x, t, labels=None):
        t_eval, s, m, sd = self.snr(t)
        y = x * (m / t_eval)
        s_vec = torch.full((x.shape[0],), s, device=x.device)
        score = self.score_fn(y, s_vec)
        if self.classifier is not None and labels is not None and self.guidance != 0:
            score = score + self.guidance * self.class_grad(y, s_vec, labels.to(x.device))
        return (y + sd ** 2 * score) / m

    def class_grad(self, y, s_vec, labels):
        if labels.is_floating_point():  # (B, n_classes) weights: ∇ Σ_c W_bc log p(c|y)
            with torch.enable_grad():
                yi = y.detach().requires_grad_(True)
                logp = F.log_softmax(self.classifier(yi, s_vec), -1)
                return torch.autograd.grad((logp * labels).sum(), yi)[0]
        cond = labels >= 0
        g = torch.zeros_like(y)
        if cond.any():
            with torch.enable_grad():
                yi = y[cond].detach().requires_grad_(True)
                logp = F.log_softmax(self.classifier(yi, s_vec[cond]), -1)
                sel = logp.gather(1, labels[cond][:, None]).sum()
                g[cond] = torch.autograd.grad(sel, yi)[0]
        return g
