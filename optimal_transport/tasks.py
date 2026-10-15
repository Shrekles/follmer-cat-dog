"""The two GPU jobs of Experiment 3. They run inside Modal containers (see modal_ot.py).

ddib():     T = decode_dog ∘ encode_cat with the two separately trained models (no training).
ot_flow():  exact optimal pairing between the 5,000 training cats and 5,000 training dogs, then
            a flow network trained on those pairs, which maps any cat to its (approximately)
            optimal dog: a learned estimate of the Brenier map.
Both read inputs from /vol and write a results .pt file to /vol/ot/.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)

import torch

from pf_ode import decode, encode
from score_sde_backend import get_device, load_model, load_score_model

REPO, CONFIG = "/root/proj/score_sde_pytorch", "vp/cifar10_ddpmpp_continuous"
PRETRAINED = "/vol/checkpoints/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth"
CAT, DOG = 3, 5


def cifar(split, cls, n=None):
    d = torch.load(f"/vol/data/cifar10_{split}.pt")
    x = d["x"][d["y"] == cls][:n].float() / 255 * 2 - 1
    return x


def chunked(fn, x, bs=256):
    return torch.cat([fn(x[i:i + bs]) for i in range(0, len(x), bs)])


def ddib(n_real=512, n_model=128, steps=200):
    dev = get_device()
    _, sde, _, s_cat = load_score_model(REPO, CONFIG, "/vol/out/cat.pth", dev)
    _, _, _, s_dog = load_score_model(REPO, CONFIG, "/vol/out/dog.pth", dev)
    out = {}
    t0 = time.time()
    for name, x, s_from, s_to in (("cat_to_dog", cifar("test", CAT, n_real), s_cat, s_dog),
                                  ("dog_to_cat", cifar("test", DOG, n_real), s_dog, s_cat)):
        z = chunked(lambda b: encode(s_from, sde, b.to(dev), steps=steps).cpu(), x)
        y = chunked(lambda b: decode(s_to, sde, b.to(dev), steps=steps).cpu(), z)
        rec = chunked(lambda b: decode(s_from, sde, b.to(dev), steps=steps).cpu(), z)   # cycle check
        out[name] = {"x": x, "y": y, "z": z, "recon": rec}
        print(f"{name}: cost {(x - y).flatten(1).pow(2).sum(1).mean():.0f}  "
              f"cycle error {(x - rec).flatten(1).pow(2).sum(1).mean():.1f}  ({time.time() - t0:.0f}s)", flush=True)
    # model samples: the same noise z decoded by both models (deterministic analogue of the same-path coupling)
    z = torch.randn(n_model, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    xc = chunked(lambda b: decode(s_cat, sde, b.to(dev), steps=steps).cpu(), z)
    xd = chunked(lambda b: decode(s_dog, sde, b.to(dev), steps=steps).cpu(), z)
    out["model_pairs"] = {"cat": xc, "dog": xd, "z": z}
    print(f"model cat/dog from the same z: cost {(xc - xd).flatten(1).pow(2).sum(1).mean():.0f}", flush=True)
    os.makedirs("/vol/ot", exist_ok=True)
    torch.save(out, "/vol/ot/ddib.pt")
    return "ddib"


def exact_ot(a, b):
    """Optimal assignment between two equal-size point clouds under squared distance."""
    from scipy.optimize import linear_sum_assignment
    C = torch.cdist(a.flatten(1).double().cuda(), b.flatten(1).double().cuda()).pow(2).cpu().numpy()
    r, c = linear_sum_assignment(C)
    return torch.as_tensor(c), float(C[r, c].mean())


def ot_flow(steps=15000, bs=64, lr=1e-4, n_eval=512, ode_steps=100, n_train=None):
    dev = get_device()
    cats, dogs = cifar("train", CAT, n_train), cifar("train", DOG, n_train)
    t0 = time.time()
    perm, cost_train = exact_ot(cats, dogs)
    print(f"exact OT on {len(cats)} train cats vs {len(dogs)} train dogs: mean cost {cost_train:.0f}  ({time.time() - t0:.0f}s)", flush=True)
    tc, td = cifar("test", CAT, n_train), cifar("test", DOG, n_train)
    perm_test, cost_test = exact_ot(tc, td)
    print(f"exact OT on {len(tc)} test cats vs {len(td)} test dogs: mean cost {cost_test:.0f}", flush=True)
    pairs_c, pairs_d = cats.to(dev), dogs[perm].to(dev)

    # velocity network: same U-Net as the diffusion models, started from the pretrained weights
    config, _, _, net = load_model(REPO, CONFIG, PRETRAINED, dev)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ema = [p.detach().clone() for p in net.parameters()]
    for step in range(1, steps + 1):
        idx = torch.randint(len(pairs_c), (bs,), device=dev)
        c, d = pairs_c[idx], pairs_d[idx]
        flip = torch.rand(bs, device=dev) < 0.5
        c = torch.where(flip[:, None, None, None], c.flip(3), c)      # flip both images of a pair together
        d = torch.where(flip[:, None, None, None], d.flip(3), d)
        t = torch.rand(bs, device=dev)
        xt = (1 - t)[:, None, None, None] * c + t[:, None, None, None] * d
        loss = (net(xt, t * 999) - (d - c)).pow(2).mean()
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / 500)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for e, p in zip(ema, net.parameters()):
                e.mul_(0.999).add_(p.detach(), alpha=0.001)
        if step % 250 == 0:
            print(f"step {step}/{steps}  loss={loss.item():.4f}  {(time.time() - t0) / 60:.0f} min", flush=True)
    for p, e in zip(net.parameters(), ema):
        p.data.copy_(e)
    net.eval()

    @torch.no_grad()
    def flow(x, t_from, t_to):
        ts = torch.linspace(t_from, t_to, ode_steps + 1)
        x = x.to(dev)
        for k in range(ode_steps):
            a, b = ts[k].item(), ts[k + 1].item()
            va = net(x, torch.full((len(x),), a * 999, device=dev))
            xp = x + (b - a) * va
            vb = net(xp, torch.full((len(x),), b * 999, device=dev))
            x = x + 0.5 * (b - a) * (va + vb)
        return x.cpu()

    out = {"cost_train_ot": cost_train, "cost_test_ot": cost_test, "perm_test": perm_test}
    xc, xd = tc[:n_eval], td[:n_eval]
    out["cat_to_dog"] = {"x": xc, "y": chunked(lambda b: flow(b, 0.0, 1.0), xc)}
    out["dog_to_cat"] = {"x": xd, "y": chunked(lambda b: flow(b, 1.0, 0.0), xd)}
    tr = cats[:min(64, n_eval)]
    out["train_fit"] = {"x": tr, "y": chunked(lambda b: flow(b, 0.0, 1.0), tr), "ot_partner": dogs[perm[:len(tr)]]}
    for k in ("cat_to_dog", "dog_to_cat"):
        v = out[k]
        print(f"{k}: cost {(v['x'] - v['y']).flatten(1).pow(2).sum(1).mean():.0f}", flush=True)
    os.makedirs("/vol/ot", exist_ok=True)
    torch.save(out, "/vol/ot/ot_flow.pt")
    torch.save({"model": net.state_dict(), "ema": {"shadow_params": []}}, "/vol/ot/ot_flow_net.pth")
    return "ot_flow"
