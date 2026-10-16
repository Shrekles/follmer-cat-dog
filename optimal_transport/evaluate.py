"""Standard evaluation instrument for Experiment 3 (plain functions, importable on Modal or locally).

- `Judge`: a ResNet-18 (CIFAR variant) 10-class classifier taking images in [-1, 1].
- `judge_probs`: softmax probabilities of the judge, batched.
- `cost_stats`, `rate_stats`: mean +- standard error of E|x - T(x)|^2 and of judge rates.
- `evaluate_pair`: all standard numbers for one transport direction.
- `grid_png`: rows of 16 images, upscaled x4 with NEAREST (the `*_big.png` figures).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CAT, DOG = 3, 5
# CIFAR-10 train per-channel mean/std in [0,1] units; converted to [-1,1] units inside the net.
_MEAN01 = (0.4914, 0.4822, 0.4465)
_STD01 = (0.2470, 0.2435, 0.2616)


class _Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + self.sc(x))


class Judge(nn.Module):
    """ResNet-18 for 32x32 inputs (3x3 stem, no max-pool). Input: images in [-1, 1]."""

    def __init__(self, n_classes=10):
        super().__init__()
        mean = torch.tensor(_MEAN01) * 2 - 1
        std = torch.tensor(_STD01) * 2
        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        layers, cin = [], 64
        for cout, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers += [_Block(cin, cout, stride), _Block(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def load_judge(path, device):
    net = Judge().to(device)
    net.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return net.eval()


@torch.no_grad()
def judge_probs(judge, x, device, bs=500):
    """Softmax probabilities (N, 10) for images x in [-1, 1] (clamped to that range first)."""
    out = []
    for i in range(0, len(x), bs):
        b = x[i:i + bs].to(device).float().clamp(-1, 1)
        out.append(F.softmax(judge(b).float(), -1).cpu())
    return torch.cat(out)


def mean_se(v):
    v = torch.as_tensor(v, dtype=torch.float64)
    return {"mean": float(v.mean()), "stderr": float(v.std(unbiased=True) / math.sqrt(len(v))) if len(v) > 1 else 0.0,
            "n": len(v)}


def sq_cost(x, y):
    """Per-image |x - y|^2 (sum over the 3,072 coordinates)."""
    return (x.float() - y.float()).flatten(1).pow(2).sum(1)


def rate_stats(probs, cls):
    hit = (probs.argmax(1) == cls).double()
    p = float(hit.mean())
    return {"rate": p, "stderr": math.sqrt(p * (1 - p) / len(hit)), "mean_prob": float(probs[:, cls].mean()),
            "n": len(hit)}


def evaluate_pair(judge, x, y, src, tgt, device):
    """All standard numbers for T: x (class src) -> y (meant to be class tgt)."""
    px, py = judge_probs(judge, x, device), judge_probs(judge, y, device)
    c = sq_cost(x, y)
    return {
        "cost": mean_se(c),
        "judge_target_on_outputs": rate_stats(py, tgt),
        "judge_source_on_outputs": rate_stats(py, src),
        "judge_source_on_inputs": rate_stats(px, src),
        "judge_target_on_inputs": rate_stats(px, tgt),
        "output_class_histogram": torch.bincount(py.argmax(1), minlength=10).tolist(),
        "output_range": [float(y.min()), float(y.max())],
        "frac_output_pixels_outside_[-1,1]": float(((y < -1) | (y > 1)).float().mean()),
    }, {"cost": c, "p_out": py, "p_in": px}


def grid_png(rows, path, scale=4, ncol=16):
    """rows: list of tensors (N, 3, 32, 32) in [-1, 1]; one image row each, first ncol columns."""
    from PIL import Image
    import numpy as np
    h = len(rows)
    canvas = torch.ones(3, h * 34 - 2, ncol * 34 - 2)
    for r, t in enumerate(rows):
        for c in range(min(ncol, len(t))):
            canvas[:, r * 34:r * 34 + 32, c * 34:c * 34 + 32] = (t[c].float().clamp(-1, 1) + 1) / 2
    arr = (canvas.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    im = Image.fromarray(arr)
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)
