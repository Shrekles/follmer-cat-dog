"""GPU tasks for the iteration loop (run inside Modal containers via modal_iter.py).

train_judge():   ResNet-18 judge on the CIFAR-10 TRAIN split -> /vol/judge/cifar10_judge.pth
transport_pf():  composed probability-flow ODE (encode with source model to s*, decode with target
                 model from s*), saving outputs for several s* -> /vol/iter01/<tag>transport_*.pt
judge_eval():    standard numbers (cost +- se, judge rates) -> /vol/iter01/<tag>{stats.json,small.pt}
Nothing existing is overwritten: every task refuses to write to a path that already exists.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

import evaluate as ev

REPO, CONFIG = "/root/proj/score_sde_pytorch", "vp/cifar10_ddpmpp_continuous"
VOL = "/vol"
CAT, DOG = 3, 5
DIRS = {"cat_to_dog": (CAT, DOG), "dog_to_cat": (DOG, CAT)}


def _fresh(path):
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def cifar(split, cls=None, n=None):
    d = torch.load(f"{VOL}/data/cifar10_{split}.pt")
    x, y = d["x"], d["y"].long()
    if cls is not None:
        x, y = x[y == cls], y[y == cls]
    return x[:n].float() / 255 * 2 - 1, y[:n]


# ----------------------------------------------------------------------------------------- judge
def _augment(xb):
    """xb: (B,3,32,32) float in [-1,1] on GPU. Random crop (pad 4, zero = mid-gray) + h-flip."""
    B = xb.shape[0]
    p = F.pad(xb, (4, 4, 4, 4))
    ox = torch.randint(0, 9, (B,), device=xb.device)
    oy = torch.randint(0, 9, (B,), device=xb.device)
    ar = torch.arange(32, device=xb.device)
    rows = (oy[:, None] + ar)[:, :, None].expand(B, 32, 32)
    cols = (ox[:, None] + ar)[:, None, :].expand(B, 32, 32)
    bidx = torch.arange(B, device=xb.device)[:, None, None].expand(B, 32, 32)
    out = p.permute(0, 2, 3, 1)[bidx, rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(B, device=xb.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


def train_judge(epochs=30, bs=512, max_lr=0.2, wd=5e-4, time_cap_min=18.0, n_train=None,
                out="judge/cifar10_judge.pth"):
    t0 = time.time()
    dev = torch.device("cuda")
    out_path = _fresh(f"{VOL}/{out}")
    xtr, ytr = cifar("train", n=n_train)
    xtr, ytr = xtr.to(dev), ytr.to(dev)
    xte, yte = cifar("test")
    N = len(xtr)
    torch.manual_seed(0)
    net = ev.Judge().to(dev).to(memory_format=torch.channels_last)
    decay, no_decay = [], []
    for n_, p in net.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0}],
                          lr=max_lr, momentum=0.9, nesterov=True)
    spe = math.ceil(N / bs)
    total = epochs * spe

    def lr_at(step):  # one-cycle: linear warm-up 15%, then cosine to ~0
        f = step / total
        if f < 0.15:
            return max_lr * (0.1 + 0.9 * f / 0.15)
        return max_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (f - 0.15) / 0.85)))

    step, ep = 0, 0
    while ep < epochs:
        net.train()
        perm = torch.randperm(N, device=dev)
        tl, tc = 0.0, 0
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            xb = _augment(xtr[idx]).contiguous(memory_format=torch.channels_last)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(xb)
                loss = F.cross_entropy(logits.float(), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            tl += loss.item() * len(idx)
            tc += (logits.argmax(1) == ytr[idx]).sum().item()
        ep += 1
        el = (time.time() - t0) / 60
        if ep == 1:  # shrink the schedule if the full run would not fit in the time cap
            fit = int(time_cap_min / max(el, 1e-6))
            if fit < epochs:
                epochs = max(2, fit)
                total = epochs * spe
                print(f"time cap: reducing to {epochs} epochs", flush=True)
        print(f"epoch {ep}/{epochs} loss {tl / N:.3f} train-acc {tc / N:.3f} lr {lr_at(step):.4f} {el:.1f} min",
              flush=True)
    net.eval()
    net = net.to(memory_format=torch.contiguous_format)
    p = ev.judge_probs(net, xte, dev)
    pred = p.argmax(1)
    per_class = {ev.CLASSES[c]: float((pred[yte == c] == c).float().mean()) for c in range(10)}
    cat_row = torch.bincount(pred[yte == CAT], minlength=10).tolist()
    dog_row = torch.bincount(pred[yte == DOG], minlength=10).tolist()
    meta = {
        "arch": "ResNet-18 (CIFAR variant), trained from scratch on CIFAR-10 train split only",
        "epochs": epochs, "batch": bs, "max_lr": max_lr, "wd": wd, "n_train": N,
        "test_acc": float((pred == yte).float().mean()),
        "test_per_class_acc": per_class,
        "test_cat_called_cat": per_class["cat"], "test_dog_called_dog": per_class["dog"],
        "test_cat_pred_hist": cat_row, "test_dog_pred_hist": dog_row,
        "test_mean_prob_cat_on_cats": float(p[yte == CAT, CAT].mean()),
        "test_mean_prob_dog_on_dogs": float(p[yte == DOG, DOG].mean()),
        "train_minutes": (time.time() - t0) / 60,
    }
    torch.save({"model": net.state_dict(), "meta": meta}, out_path)
    print("judge:", json.dumps(meta), flush=True)
    return {"task": "train_judge", "minutes": (time.time() - t0) / 60, "meta": meta}


# ------------------------------------------------------------------------------------- transport
def transport_pf(direction, stops=(0.4, 0.6, 0.8), full=True, n=512, steps=200, bs=256, eps=1e-3,
                 cat_ckpt="out/cat.pth", dog_ckpt="out/dog.pth", tag="", name="A", folder="iter01"):
    """Composed PF-ODE. For each k in ks (grid index on the full 0..steps grid eps->1), encode with the
    source model to s_k = eps + k (1-eps)/steps, then decode s_k -> eps with the target model using k
    Heun steps. k = steps is the full (valid) coupling; k < steps is partial encoding (NOT a coupling)."""
    from pf_ode import pf_ode
    from score_sde_backend import load_score_model
    t0 = time.time()
    dev = torch.device("cuda")
    out_path = _fresh(f"{VOL}/{folder}/{tag}transport_{direction}_{name}.pt")
    src, tgt = DIRS[direction]
    _, sde, _, s_cat = load_score_model(REPO, CONFIG, f"{VOL}/{cat_ckpt}", dev)
    _, _, _, s_dog = load_score_model(REPO, CONFIG, f"{VOL}/{dog_ckpt}", dev)
    s_from, s_to = (s_cat, s_dog) if src == CAT else (s_dog, s_cat)
    x, _ = cifar("test", src, n)
    ks = sorted({round(s * steps) for s in stops} | ({steps} if full else set()))
    s_of = {k: eps + k * (sde.T - eps) / steps for k in ks}
    outs = {k: [] for k in ks}
    for i in range(0, len(x), bs):
        z, k_prev = x[i:i + bs].to(dev), 0
        states = {}
        for k in ks:
            z = pf_ode(s_from, sde, z, eps + k_prev * (sde.T - eps) / steps, s_of[k], k - k_prev)
            states[k], k_prev = z, k
        for k in ks:
            outs[k].append(pf_ode(s_to, sde, states[k], s_of[k], eps, k).cpu())
        print(f"{direction} batch {i // bs + 1}: done {(time.time() - t0) / 60:.1f} min", flush=True)
    res = {"x": x, "outputs": {}, "meta": {}}
    for k in ks:
        y = torch.cat(outs[k])
        key = f"s{k / steps:.1f}"
        ones = torch.ones(1, 1, 1, 1, dtype=torch.float64)
        mean, std = sde.marginal_prob(ones, torch.tensor([s_of[k]], dtype=torch.float64))
        res["outputs"][key] = y
        res["meta"][key] = {"k": k, "s": s_of[k], "encode_steps": k, "decode_steps": k,
                            "signal_coef_alpha": float(mean.flatten()[0]), "noise_std": float(std.flatten()[0]),
                            "valid_coupling": k == steps}
        print(f"{direction} {key}: cost {ev.sq_cost(x, y).mean():.1f}", flush=True)
    res["meta"]["_run"] = {"direction": direction, "n": len(x), "steps": steps, "eps": eps,
                           "cat_ckpt": cat_ckpt, "dog_ckpt": dog_ckpt, "minutes": (time.time() - t0) / 60}
    torch.save(res, out_path)
    return {"task": f"transport_pf {direction} {ks}", "minutes": (time.time() - t0) / 60}


# ------------------------------------------------------------------------------------ judge eval
def judge_eval(tag="", judge="judge/cifar10_judge.pth", n_save=64, folder="iter01"):
    t0 = time.time()
    dev = torch.device("cuda")
    stats_path = _fresh(f"{VOL}/{folder}/{tag}stats.json")
    small_path = _fresh(f"{VOL}/{folder}/{tag}small.pt")
    ck = torch.load(f"{VOL}/{judge}", map_location="cpu")
    net = ev.Judge().to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    stats = {"judge": ck["meta"], "directions": {}}
    # ceilings on all real test cats / dogs (recomputed, sanity check vs. training meta)
    for cls in (CAT, DOG):
        xr, _ = cifar("test", cls)
        stats["judge"][f"recheck_all_test_{ev.CLASSES[cls]}"] = ev.rate_stats(ev.judge_probs(net, xr, dev), cls)
    small = {}
    files = sorted(f for f in os.listdir(f"{VOL}/{folder}") if f.startswith(f"{tag}transport_"))
    if not tag:
        files = [f for f in files if not f.startswith("smoke_")]
    for direction, (src, tgt) in DIRS.items():
        x, outs, meta = None, {}, {}
        for f in files:
            if f"_{direction}_" not in f:
                continue
            r = torch.load(f"{VOL}/{folder}/{f}")
            if x is None:
                x = r["x"]
            assert torch.equal(x, r["x"]), f"inputs differ in {f}"
            for key, y in r["outputs"].items():
                assert key not in outs, f"duplicate {key} in {f}"
                outs[key], meta[key] = y, {**r["meta"][key], "file": f, "run": r["meta"]["_run"]}
        if x is None:
            continue
        stats["directions"][direction] = {}
        small[direction] = {"x": x[:n_save].clone(), "outputs": {}, "per_image": {}}
        for key in sorted(outs):
            s, per = ev.evaluate_pair(net, x, outs[key], src, tgt, dev)
            stats["directions"][direction][key] = {**s, "meta": meta[key]}
            small[direction]["outputs"][key] = outs[key][:n_save].clone()
            small[direction]["per_image"][key] = {"cost": per["cost"], "p_target_out": per["p_out"][:, tgt],
                                                  "pred_out": per["p_out"].argmax(1),
                                                  "p_source_in": per["p_in"][:, src]}
            print(f"{direction} {key}: cost {s['cost']['mean']:.1f}+-{s['cost']['stderr']:.1f}  "
                  f"judge {ev.CLASSES[tgt]}-rate on T(x) {s['judge_target_on_outputs']['rate']:.3f}  "
                  f"(inputs called {ev.CLASSES[src]}: {s['judge_source_on_inputs']['rate']:.3f})", flush=True)
    stats["judge_eval_minutes"] = (time.time() - t0) / 60
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=1)
    torch.save(small, small_path)
    return {"task": "judge_eval", "minutes": (time.time() - t0) / 60}
