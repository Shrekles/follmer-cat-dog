"""GPU tasks for iteration 03 (CIFAR-only refinement of the target models). Run inside Modal via modal_iter03.py.

All new outputs go to /vol/iter03/; nothing existing is ever overwritten (out/*.pth are read-only inits).

finetune_cifar(): fine-tune out/{cls}.pth (EMA weights, fresh Adam) on CIFAR-10 TRAIN images of one class only.
decode_saved(): decode the latents saved in iteration 02 (encoded by the unchanged source model) with a new
    target model -> /vol/iter03/transport_<dir>_<name>.pt   (T = decode_new o encode_old, plain composed ODE).
memo_check(): median over samples of the min-over-train squared L2 distance (memorisation guard).
chain(): container C (cat) or D (dog): finetune -> prior -> decode -> (mid prior) -> memo -> eval.
"""
import json
import os
import time

import torch

from tasks_iter02 import *  # noqa: F401,F403
from tasks_iter02 import _left, eval_runs, prior_samples, prior_z  # noqa: F401
from tasks_iter import CAT, CONFIG, DIRS, DOG, REPO, VOL, _fresh, cifar
import evaluate as ev

CLS = {"cat": CAT, "dog": DOG}


# ------------------------------------------------------------------------------ fine-tune (CIFAR)
def finetune_cifar(cls, init, out, mid, bs=256, lr=1e-4, lr_end=1e-5, warmup=200, ema_rate=0.999, cap_min=40.0,
                   deadline=None, reserve_min=17.0, seed=3, max_steps=10 ** 9):
    from score_sde_backend import load_model
    t0, dev = time.time(), torch.device("cuda")
    cap = min(cap_min, max(1.0, _left(deadline) - reserve_min))
    out_path, mid_path = _fresh(f"{VOL}/{out}"), _fresh(f"{VOL}/{mid}")
    torch.manual_seed(seed)
    config, sde, eps, model = load_model(REPO, CONFIG, f"{VOL}/{init}", dev)
    model.train()
    xa, _ = cifar("train", CLS[cls])                  # [-1,1] floats, CIFAR-10 TRAIN images of this class only
    xa = xa.to(dev)
    print(f"finetune_cifar {cls}: {len(xa)} train images, bs {bs}, cap {cap:.1f} min", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = [p.detach().clone() for p in model.parameters()]
    accum = 1

    def save(path, step):
        backup = [p.detach().clone() for p in model.parameters()]
        for p_, e in zip(model.parameters(), ema):
            p_.data.copy_(e)
        torch.save({"model": model.state_dict(), "ema": {"shadow_params": []}, "step": step,
                    "meta": {"init": init, "bs": bs, "lr": lr, "lr_end": lr_end, "ema": ema_rate, "cap_min": cap,
                             "data": f"CIFAR-10 train {cls} only (5000 images), random hflip"}}, path)
        for p_, b in zip(model.parameters(), backup):
            p_.data.copy_(b)

    def loss_on(x):
        flip = torch.rand(len(x), device=dev) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        s = torch.rand(len(x), device=dev) * (sde.T - eps) + eps
        z = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_ = model(mean + std[:, None, None, None] * z, s * 999)
        return (out_.float() - z).pow(2).mean()

    def lr_at(step, frac):
        w = min(1.0, (step + 1) / warmup)
        if frac < 0.5:
            return lr * w
        return w * (lr + (lr_end - lr) * min(1.0, (frac - 0.5) / 0.5))

    step, running, saved_mid, log, cur_lr = 0, 0.0, False, [], lr
    while step < max_steps and (time.time() - t0) / 60 < cap:
        frac = (time.time() - t0) / 60 / cap
        x = xa[torch.randint(len(xa), (bs,), device=dev)]
        cur_lr = lr_at(step, frac)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        try:
            tot = 0.0
            for xc in x.chunk(accum):
                loss = loss_on(xc) / accum
                loss.backward()
                tot += loss.item()
        except torch.cuda.OutOfMemoryError:
            accum *= 2
            print(f"OOM at batch {bs}: gradient accumulation x{accum}", flush=True)
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            torch._foreach_mul_(ema, ema_rate)
            torch._foreach_add_(ema, [p.detach() for p in model.parameters()], alpha=1 - ema_rate)
        step += 1
        running += tot
        if step % 100 == 0 or (max_steps < 100 and step % 5 == 0):
            k = 100 if step % 100 == 0 else 5
            el = (time.time() - t0) / 60
            log.append((step, running / k, el, cur_lr))
            print(f"step {step} loss {running / k:.4f} lr {cur_lr:.2e} {el:.1f} min ({step / max(el, 1e-6) / 60:.2f} it/s)"
                  f" peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            running = 0.0
        if not saved_mid and (time.time() - t0) / 60 > cap / 2:
            save(mid_path, step)
            saved_mid = True
            print(f"saved mid snapshot at step {step}", flush=True)
    mid_step = None if saved_mid else step
    if not saved_mid:
        save(mid_path, step)
    save(out_path, step)
    print(f"saved {out_path} at step {step} (accum {accum}, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)",
          flush=True)
    return {"task": f"finetune_cifar {cls}", "steps": step, "images_seen": step * bs, "accum": accum, "loss_log": log,
            "cap_min": cap, "mid_saved_at_end": mid_step is not None,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "minutes": (time.time() - t0) / 60}


# ------------------------------------------------------------------------- decode saved latents
def decode_saved(direction, tgt_ckpt, name, lat_path=None, folder="iter03", tag="", n=None, steps=200, bs=256,
                 eps=1e-3, deadline=None, est_min=5.5):
    from pf_ode import pf_ode
    t0, dev = time.time(), torch.device("cuda")
    lat_path = lat_path or f"{VOL}/iter02/latents_{direction}.pt"
    d = torch.load(lat_path)
    x, z = d["x"][:n], d["z"][:n]
    if _left(deadline) < est_min * len(x) / 512:
        print(f"SKIP decode {direction} {name}: deadline", flush=True)
        return {"task": f"decode_saved {direction} {name} SKIPPED (deadline)", "minutes": (time.time() - t0) / 60}
    out_path = _fresh(f"{VOL}/{folder}/{tag}transport_{direction}_{name}.pt")
    from tasks_iter02 import _score
    sde, s = _score(tgt_ckpt, dev)
    y = torch.cat([pf_ode(s, sde, z[i:i + bs].to(dev), sde.T, eps, steps).cpu() for i in range(0, len(x), bs)])
    meta = {"s1.0": {"k": steps, "s": 1.0, "w": 0.0, "valid_coupling": True,
                     "note": "composed PF-ODE: latents encoded (iter02) by the unchanged source model, decoded by the "
                             "iter03 CIFAR-refined target model"},
            "_run": {"direction": direction, "n": len(x), "steps": steps, "eps": eps, "encoder": d["meta"]["encoder"],
                     "latents": lat_path, "tgt_ckpt": tgt_ckpt, "minutes": (time.time() - t0) / 60}}
    torch.save({"x": x, "outputs": {"s1.0": y}, "meta": meta}, out_path)
    print(f"decode {direction} {name}: cost {ev.sq_cost(x, y).mean():.1f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return {"task": f"decode_saved {direction} {name}", "minutes": (time.time() - t0) / 60}


# ----------------------------------------------------------------------------- memorisation guard
def memo_check(cls, sets, out_name, folder="iter03", n_test=512):
    """sets: {name: path to a prior_*.pt}. Median over samples of min-over-train squared L2 (sum over pixels,
    [-1,1] scale) to the 5000 CIFAR-train images of class cls; same for the first n_test real test images."""
    t0, dev = time.time(), torch.device("cuda")
    out_path = _fresh(f"{VOL}/{folder}/{out_name}.json")
    tr = cifar("train", CLS[cls])[0].flatten(1).to(dev)
    tr2 = tr.pow(2).sum(1)

    def nn_d(y):
        y = y.flatten(1).to(dev).float().clamp(-1, 1)
        d = y.pow(2).sum(1)[:, None] + tr2[None, :] - 2 * y @ tr.T
        return d.min(1).values.clamp_min(0).cpu()

    res = {"test_first512": None, "sets": {}}
    dt = nn_d(cifar("test", CLS[cls], n_test)[0])
    res["test_first512"] = {"median": float(dt.median()), "mean": float(dt.mean()), "q05": float(dt.quantile(0.05))}
    for name, p in sets.items():
        if not os.path.exists(p):
            print(f"memo: missing {p}", flush=True)
            continue
        dd = nn_d(torch.load(p)["samples"])
        res["sets"][name] = {"median": float(dd.median()), "mean": float(dd.mean()), "q05": float(dd.quantile(0.05)),
                             "ratio_to_test_median": float(dd.median() / dt.median()),
                             "frac_below_10pct_test_median": float((dd < 0.1 * dt.median()).float().mean()),
                             "memorisation_flag": bool(dd.median() < 0.5 * dt.median())}
        print(f"memo {name}: median {dd.median():.1f} vs test {dt.median():.1f}", flush=True)
    res["minutes"] = (time.time() - t0) / 60
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    return {"task": f"memo_check {cls}", "result": res, "minutes": res["minutes"]}


# ------------------------------------------------------------------------------- container chain
def chain(cls, deadline, vol=None, smoke=False, tag="", folder="iter03", cap_min=40.0, skip_train=False,
          use_mid=False):
    t0, res = time.time(), []
    direction = "dog_to_cat" if cls == "cat" else "cat_to_dog"
    S = dict(steps=10) if smoke else {}
    n = 4 if smoke else 512
    ck, ck_mid = f"{folder}/{tag}{cls}_c3.pth", f"{folder}/{tag}{cls}_c3_mid.pth"

    def commit():
        if vol is not None:
            vol.commit()

    if not skip_train:
        res.append(finetune_cifar(cls, init=f"out/{cls}.pth", out=ck, mid=ck_mid, cap_min=cap_min, deadline=deadline,
                                  max_steps=20 if smoke else 10 ** 9))
        commit()
    use = ck_mid if use_mid else ck
    nm = f"{cls}_c3_mid" if use_mid else f"{cls}_c3"
    res.append(prior_samples({nm: use}, n=n, folder=folder, tag=tag, deadline=deadline, est_min=5.5, **S))
    commit()
    dname = "iter03_c3_mid" if use_mid else "iter03_c3"
    res.append(decode_saved(direction, use, dname, folder=folder, tag=tag, n=n, deadline=deadline, **S))
    commit()
    runs = [{"name": f"prior_{nm}", "kind": "prior", "path": f"{VOL}/{folder}/{tag}prior_{nm}.pt", "cls": CLS[cls]},
            {"name": f"{direction}_{dname}", "kind": "transport", "direction": direction,
             "path": f"{VOL}/{folder}/{tag}transport_{direction}_{dname}.pt"}]
    memo_sets = {f"prior_{nm}": f"{VOL}/{folder}/{tag}prior_{nm}.pt", f"prior_{cls}_old": f"{VOL}/iter02/prior_{cls}.pt"}
    if not use_mid and not skip_train and _left(deadline) > 20:
        res.append(prior_samples({f"{cls}_c3_mid": ck_mid}, n=n, folder=folder, tag=tag, deadline=deadline,
                                 est_min=5.5, **S))
        commit()
        runs.append({"name": f"prior_{cls}_c3_mid", "kind": "prior", "path": f"{VOL}/{folder}/{tag}prior_{cls}_c3_mid.pt",
                     "cls": CLS[cls]})
        memo_sets[f"prior_{cls}_c3_mid"] = f"{VOL}/{folder}/{tag}prior_{cls}_c3_mid.pt"
    suffix = "_mid" if use_mid else ""
    res.append(memo_check(cls, memo_sets, f"{tag}memo_{cls}{suffix}", folder=folder))
    res.append(eval_runs(runs, f"{tag}eval{cls[0].upper()}{suffix}", folder=folder, kid_refs=False,
                         n_boot=10 if smoke else 200))
    commit()
    return {"task": f"chain {cls}", "sub": res, "minutes": (time.time() - t0) / 60}
