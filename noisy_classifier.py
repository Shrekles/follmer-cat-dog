"""Time-dependent CIFAR-10 classifier p_s(c | y_s) for classifier guidance.

It must be trained on all 10 classes with the *same* forward SDE and data scaling as
the score model, so that  ∇log p_s(y) + ∇log p_s(c|y) = ∇log p_s(y|c)  exactly.
(A cat-vs-dog-only classifier would give the wrong Bayes posterior.)

    python noisy_classifier.py --repo score_sde_pytorch --config vp/cifar10_ddpmpp_continuous \
        --out checkpoints/cifar10_noisy_classifier.pt --epochs 30
"""
import argparse
import importlib
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(s, dim):
    """Sinusoidal embedding of s*999 (same convention as score_sde continuous models)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=s.device) / half)
    a = (s * 999)[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Block(nn.Module):
    def __init__(self, cin, cout, temb, down, dropout=0.1):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(min(32, cin // 4), cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(temb, 2 * cout)
        self.n2, self.c2 = nn.GroupNorm(min(32, cout // 4), cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.down = down

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.t(temb)[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + scale) + shift  # FiLM on noise level
        h = self.c2(self.drop(F.silu(h)))
        h = h + self.skip(x)
        return F.avg_pool2d(h, 2) if self.down else h


class NoisyClassifier(nn.Module):
    def __init__(self, nf=96, ch_mult=(1, 2, 2), blocks=2, n_classes=10, temb=256):
        super().__init__()
        self.temb_dim = temb
        self.temb = nn.Sequential(nn.Linear(temb, temb), nn.SiLU(), nn.Linear(temb, temb))
        self.inp = nn.Conv2d(3, nf, 3, padding=1)
        layers, c = [], nf
        for i, mult in enumerate(ch_mult):
            for b in range(blocks):
                down = (b == blocks - 1) and (i < len(ch_mult) - 1)
                layers.append(Block(c, nf * mult, temb, down))
                c = nf * mult
        self.blocks = nn.ModuleList(layers)
        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, n_classes))

    def forward(self, y, s):
        temb = self.temb(timestep_embedding(s, self.temb_dim))
        h = self.inp(y)
        for blk in self.blocks:
            h = blk(h, temb)
        return self.out(h)


def load_cifar(root, split, device):
    """CIFAR-10 as uint8 tensors: data/cifar10_{split}.pt (converted from the HF mirror), else torchvision."""
    path = os.path.join(root, f"cifar10_{split}.pt")
    if os.path.exists(path):
        d = torch.load(path)
        x, y = d["x"], d["y"]
    else:
        import torchvision
        ds = torchvision.datasets.CIFAR10(root, train=(split == "train"), download=True)
        x, y = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(ds.targets)
    return x.to(device), y.to(device)


def load_classifier(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = NoisyClassifier(**ck["kwargs"])
    net.load_state_dict(ck["model"])
    return net.to(device).eval()


def main():
    from score_sde_backend import build_sde, get_device, import_score_sde
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="score_sde_pytorch")
    p.add_argument("--config", default="vp/cifar10_ddpmpp_continuous")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="checkpoints/cifar10_noisy_classifier.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    sde_lib, _ = import_score_sde(args.repo)
    config = importlib.import_module("configs." + args.config.replace("/", ".")).get_config()
    sde, eps = build_sde(sde_lib, config)
    centered = config.data.centered
    device = get_device()
    print(f"device={device}  sde={type(sde).__name__}  centered={centered}")

    xtr, ytr = load_cifar(args.data, "train", device)
    xte, yte = load_cifar(args.data, "test", device)
    steps_per_epoch = len(xtr) // args.bs

    def train_batches():
        perm = torch.randperm(len(xtr), device=device)
        for i in range(steps_per_epoch):
            idx = perm[i * args.bs:(i + 1) * args.bs]
            x = xtr[idx].float() / 255
            flip = torch.rand(len(idx), device=device) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
            yield x, ytr[idx]

    def test_batches():
        for i in range(0, len(xte), 500):
            yield xte[i:i + 500].float() / 255, yte[i:i + 500]

    kwargs = dict(nf=96, ch_mult=(1, 2, 2), blocks=2, n_classes=10)
    net = NoisyClassifier(**kwargs).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch)

    def noisy(x, s):
        x = x * 2 - 1 if centered else x
        mean, std = sde.marginal_prob(x, s)
        return mean + std[:, None, None, None] * torch.randn_like(x)

    for ep in range(args.epochs):
        net.train()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for x, c in train_batches():
            x, c = x.to(device), c.to(device)
            s = torch.rand(x.shape[0], device=device) * (sde.T - eps) + eps
            logits = net(noisy(x, s), s)
            loss = F.cross_entropy(logits, c)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(c)
            correct += (logits.argmax(1) == c).sum().item()
            n += len(c)
        # test accuracy at a few fixed noise levels
        net.eval()
        accs = {}
        with torch.no_grad():
            for sv in (0.01, 0.2, 0.4, 0.6):
                ok = m = 0
                for x, c in test_batches():
                    x, c = x.to(device), c.to(device)
                    s = torch.full((x.shape[0],), sv, device=device)
                    ok += (net(noisy(x, s), s).argmax(1) == c).sum().item()
                    m += len(c)
                accs[sv] = ok / m
        print(f"epoch {ep + 1}/{args.epochs}  loss={tot / n:.3f}  train_acc={correct / n:.3f}  "
              f"test_acc@s={ {k: round(v, 3) for k, v in accs.items()} }  ({time.time() - t0:.0f}s)")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"model": net.state_dict(), "kwargs": kwargs, "config": args.config}, args.out)


if __name__ == "__main__":
    main()
