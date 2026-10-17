"""GPU tasks for iteration 05 (iter_05_decay): resume the iteration-04 from-scratch nets with a cosine lr decay and two
EMAs, evaluate every snapshot with the transport protocol, then evaluate the composed ODE (with and without a final
Tweedie denoise step), mixed encoder/decoder pairs, and the Gaussianity of every latent set.

Run inside Modal via modal_iter05.py. All outputs go to /vol/iter05/ (smoke: /vol/iter05/smoke/); /vol/iter04/,
/vol/iter02/, /vol/iter01/ and /vol/out/ are only read. Nothing existing is ever overwritten (_fresh).

resume_decay():  raw weights + Adam from iter04/{cls}_last.pth, time-based cosine lr 2e-4 -> 2e-6, EMAs 0.9995 and
                 0.9999 (both initialised from the iter-04 0.9995 EMA); at 50/75/100% of the cap saves both EMAs
                 ({cls}_e9995_s{step}.pth, {cls}_e9999_s{step}.pth) and the resumable state {cls}_last_s{step}.pth.
snap_eval05():   per snapshot: 512 prior samples (prior_z(512), seed 0), 200 Heun steps, eps 1e-3 (the transport
                 protocol); judge, KID vs all 1,000 test images of the class (plain and with the Tweedie step),
                 saturated-sample fraction, memorisation ratio; pick05() rule from the iteration-04 VALIDATION spec.
eval05():        composed ODE both directions (512 + 512), mixed pairs M1-M4, FT / iter04 re-scored with and without
                 Tweedie, latent Gaussianity, required figures (denoised flow grid, steering grid, strict best pairs,
                 same noise, 32-column comparison per direction).
"""
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import evaluate as ev
from tasks_iter import CAT, CONFIG, DIRS, DOG, REPO, VOL, _fresh, cifar
from tasks_iter02 import _Incep, eval_runs, kid, prior_z
from tasks_iter03 import memo_check
from tasks_iter04 import CLS, JUDGE, _commit, _score4, _segments, labeled_grid

FT = {"cat": "out/cat.pth", "dog": "out/dog.pth"}
REF04 = {"cat": "iter04/cat_s17297.pth", "dog": "iter04/dog_s20119.pth"}       # iteration-04 picks
FT_TRANSPORT = {"cat_to_dog": "iter01/transport_cat_to_dog_B.pt", "dog_to_cat": "iter01/transport_dog_to_cat_A.pt"}
IT04_TRANSPORT = {d: f"iter04/transport_{d}_scratch.pt" for d in DIRS}
FT_LATENTS = {d: f"iter02/latents_{d}.pt" for d in DIRS}
IT04_LATENTS = {d: f"iter04/latents_{d}.pt" for d in DIRS}
MEMO_MIN_RATIO, MEMO_WARN_RATIO = 0.5, 0.8


def _etag(r):
    return "e" + f"{r:.4f}"[2:]          # 0.9995 -> e9995


def sat_frac(y):
    """Fraction of samples with more than 50% of their pixels outside [-1, 1]."""
    return float((((y < -1) | (y > 1)).flatten(1).float().mean(1) > 0.5).float().mean())


def _alpha_sigma(sde, s, dev):
    one = torch.ones(1, 1, 1, 1, device=dev)
    m, sd = sde.marginal_prob(one, torch.full((1,), float(s), device=dev))
    return float(m.flatten()[0]), float(sd.flatten()[0])


@torch.no_grad()
def tweedie(score, sde, x, s, dev, bs=256):
    """Denoised estimate x0_hat = (x_s + sigma_s^2 score(x_s, s)) / alpha_s (one network evaluation, deterministic)."""
    a, sd = _alpha_sigma(sde, s, dev)
    out = []
    for i in range(0, len(x), bs):
        b = x[i:i + bs].to(dev)
        out.append(((b + sd ** 2 * score(b, torch.full((len(b),), float(s), device=dev))) / a).cpu())
    return torch.cat(out)


# --------------------------------------------------------------------------------- training
def resume_decay(cls, last=None, folder="iter05", cap_min=45.0, lr_max=2e-4, lr_min=2e-6, bs=256,
                 ema_rates=(0.9995, 0.9999), snap_fracs=(0.5, 0.75, 1.0), seed=5, max_steps=None, p0=0.0, vol=None):
    """Resume raw weights + Adam from `last` (volume-relative). lr = lr_min + (lr_max-lr_min)(1+cos(pi p))/2 with
    p = p0 + (1-p0) * elapsed/cap (time-based, checked every step; p0 > 0 only when continuing a cancelled run from
    an iter05 _last_s*.pth). Snapshots when p reaches each of snap_fracs."""
    from score_sde_backend import build_network
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    last = last or f"iter04/{cls}_last.pth"
    ck = torch.load(f"{VOL}/{last}", map_location="cpu", weights_only=False)
    ov = ck["config_overrides"]
    config, sde, eps, model = build_network(REPO, CONFIG, dev, ov)
    model.load_state_dict({k.removeprefix("module."): v for k, v in ck["model"].items()}, strict=True)
    params = list(model.parameters())
    n_par = sum(p.numel() for p in params)
    step0 = int(ck["step"])
    min0 = (ck.get("meta") or {}).get("train_minutes")
    if min0 is None:  # iter04 _last.pth has no meta; its final EMA snapshot (same step) records train_minutes
        sp = f"{VOL}/iter04/{cls}_s{step0}.pth"
        min0 = torch.load(sp, map_location="cpu", weights_only=False)["meta"]["train_minutes"] if os.path.exists(sp) \
            else float("nan")
    opt = torch.optim.Adam(params, lr=lr_max)
    warmup, adam_note = 0, f"Adam state loaded from {last}"
    try:
        opt.load_state_dict(ck["optimizer"])
        st = opt.state[params[0]]
        assert st["exp_avg"].shape == params[0].shape
        adam_steps = float(st["step"])
    except Exception as e:  # noqa: BLE001
        opt = torch.optim.Adam(params, lr=lr_max)
        warmup, adam_steps = 200, 0.0
        adam_note = f"Adam state FAILED to load ({e!r}); fresh Adam with a 200-step warmup"
    if "emas" in ck:   # an iter05 _last_s*.pth: both EMAs stored
        emas = {r: [s.to(dev).float().clone() for s in ck["emas"][str(r)]] for r in ema_rates}
        ema_note = f"both EMAs restored from {last}"
    else:
        shadow = ck["ema"]["shadow_params"]
        assert len(shadow) == len(params)
        emas = {r: [s.to(dev).float().clone() for s in shadow] for r in ema_rates}
        ema_note = f"both EMAs initialised from the 0.9995 EMA in {last}"
    print(f"resume {cls} from {last}: step {step0}, {min0:.1f} train-min, {n_par} params; {adam_note} "
          f"(adam step {adam_steps:.0f}); {ema_note}", flush=True)
    model.train()
    xa, _ = cifar("train", CLS[cls])                 # CIFAR-10 TRAIN images of this class only
    xa = xa.to(dev)
    snaps, log = [], []

    def lr_at(p):
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * min(max(p, 0.0), 1.0)))

    def meta(elapsed, p, kind):
        return {"init": f"iter04 from-scratch net; resumed from {last} (step {step0})", "cls": cls, "bs": bs,
                "lr_schedule": f"time-based cosine {lr_max} -> {lr_min} over {cap_min} min (p0={p0})",
                "lr_now": lr_at(p), "progress": p, "ema_rates": list(ema_rates), "kind": kind, "grad_clip": 1.0,
                "bf16_autocast": True, "train_minutes": min0 + elapsed, "iter05_minutes": elapsed, "n_params": n_par,
                "adam": adam_note, "ema_init": ema_note,
                "data": f"CIFAR-10 train {cls} only ({len(xa)} images), random hflip"}

    def save(step, elapsed, p, frac):
        g = step0 + step
        names = [n for n, _ in model.named_parameters()]
        raw = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for r, e in emas.items():
            sd = dict(raw)
            for n, t in zip(names, e):
                sd[n] = t.detach().cpu().clone()
            rel = f"{folder}/{cls}_{_etag(r)}_s{g}.pth"
            torch.save({"model": sd, "ema": {"shadow_params": []}, "step": g, "config_name": CONFIG,
                        "config_overrides": ov, "meta": {**meta(elapsed, p, f"EMA {r} weights"), "ema": r}},
                       _fresh(f"{VOL}/{rel}"))
            snaps.append({"step": g, "path": rel, "minutes": min0 + elapsed, "ema": r, "frac": frac})
        rel = f"{folder}/{cls}_last_s{g}.pth"
        torch.save({"model": raw, "ema": {"shadow_params": [t.detach().cpu() for t in emas[ema_rates[0]]]},
                    "emas": {str(r): [t.detach().cpu() for t in e] for r, e in emas.items()},
                    "optimizer": opt.state_dict(), "step": g, "config_name": CONFIG, "config_overrides": ov,
                    "meta": meta(elapsed, p, "resumable state (raw weights, both EMAs, Adam)")},
                   _fresh(f"{VOL}/{rel}"))
        print(f"saved snapshots at step {g} (p={p:.3f}, {elapsed:.1f} min) + {rel}", flush=True)
        _commit(vol)

    t0 = time.time()
    step, running, nrun, nxt = 0, 0.0, 0, 0
    fracs = list(snap_fracs)

    def prog():
        if max_steps:
            return p0 + (1 - p0) * step / max_steps
        return p0 + (1 - p0) * (time.time() - t0) / 60 / cap_min

    while prog() < 1.0:
        p = prog()
        lr = lr_at(p) * (min(1.0, (step + 1) / warmup) if warmup else 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        x = xa[torch.randint(len(xa), (bs,), device=dev)]
        flip = torch.rand(bs, device=dev) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        s = torch.rand(bs, device=dev) * (sde.T - eps) + eps
        z = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(mean + std[:, None, None, None] * z, s * 999)
        loss = (out.float() - z).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        with torch.no_grad():
            det = [q.detach() for q in params]
            for r, e in emas.items():
                torch._foreach_mul_(e, r)
                torch._foreach_add_(e, det, alpha=1 - r)
        step += 1
        if step % 50 == 0 or max_steps:
            running += loss.item()
            nrun += 1
        if step % 500 == 0 or (max_steps and step % 5 == 0):
            el = (time.time() - t0) / 60
            log.append((step0 + step, running / max(nrun, 1), el, lr))
            print(f"step {step0 + step} loss {running / max(nrun, 1):.4f} lr {lr:.2e} {el:.1f} min "
                  f"({step / max(el, 1e-6) / 60:.2f} it/s) peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
                  flush=True)
            running, nrun = 0.0, 0
        if nxt < len(fracs) - 1 and prog() >= fracs[nxt]:
            save(step, (time.time() - t0) / 60, prog(), fracs[nxt])
            nxt += 1
    el = (time.time() - t0) / 60
    save(step, el, 1.0, fracs[-1])
    return {"task": f"resume_decay {cls}", "last": last, "step0": step0, "steps": step, "images_seen": step * bs,
            "adam": adam_note, "ema_init": ema_note, "n_params": n_par, "loss_log": log, "snapshots": snaps,
            "it_per_s": step / max(el * 60, 1e-6), "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
            "minutes": el}


# ---------------------------------------------------------------------------------- snapshot eval
def snap_eval05(cls, snaps, folder="iter05", n=512, steps=200, eps=1e-3, n_boot=100, vol=None, suffix=""):
    """snaps: list of {"name", "step", "path", "minutes", "ema", "reference"}. Transport protocol (n=512, 200 Heun)."""
    from pf_ode import pf_ode
    t0, dev, c = time.time(), torch.device("cuda"), CLS[cls]
    net = ev.load_judge(f"{VOL}/{JUDGE}", dev)
    inc = _Incep(dev)
    real = cifar("test", c)[0]
    freal = inc.feats(real)
    ceil = ev.rate_stats(ev.judge_probs(net, real, dev), c)
    out = {"cls": cls, "n_samples": n, "heun_steps": steps, "eps": eps, "noise": "prior_z(n), seed 0",
           "kid_floor_train512_vs_test": kid(inc.feats(cifar("train", c, 512)[0]), freal, n_boot),
           "ceiling_all_test": ceil, "snapshots": {}}
    z = prior_z(n)
    memo_sets, figs = {}, {}
    for sn in snaps:
        name = sn["name"]
        sde, sc = _score4(sn["path"], dev)
        y = torch.cat([pf_ode(sc, sde, z[i:i + 256].to(dev), sde.T, eps, steps).cpu() for i in range(0, n, 256)])
        ytw = tweedie(sc, sde, y, eps, dev)
        ppath = _fresh(f"{VOL}/{folder}/prior_{name}.pt")
        torch.save({"z": z, "samples": y, "samples_tweedie": ytw,
                    "meta": {"ckpt": sn["path"], "seed": 0, "steps": steps, "eps": eps, "n": n}}, ppath)
        memo_sets[f"prior_{name}"] = ppath
        p, ptw = ev.judge_probs(net, y, dev), ev.judge_probs(net, ytw, dev)
        st = {"step": sn["step"], "ema": sn.get("ema"), "reference": bool(sn.get("reference")),
              "train_minutes": sn["minutes"], "ckpt": sn["path"], "prior_file": ppath,
              "judge_target": ev.rate_stats(p, c), "judge_cat": ev.rate_stats(p, CAT), "judge_dog": ev.rate_stats(p, DOG),
              "class_histogram": torch.bincount(p.argmax(1), minlength=10).tolist(),
              "kid_vs_all_test": kid(inc.feats(y), freal, n_boot),
              "judge_target_tweedie": ev.rate_stats(ptw, c),
              "kid_vs_all_test_tweedie": kid(inc.feats(ytw), freal, n_boot),
              "frac_pixels_outside_[-1,1]": float(((y < -1) | (y > 1)).float().mean()),
              "frac_samples_saturated": sat_frac(y), "frac_samples_saturated_tweedie": sat_frac(ytw)}
        out["snapshots"][name] = st
        worst = p[:, c].argsort()[:16]
        f1, f2 = f"prior_{name}_big.png", f"prior_{name}_lowest_ptarget_big.png"
        ev.grid_png([y[:16], y[16:32]], f"{VOL}/{folder}/{f1}")
        ev.grid_png([y[worst]], f"{VOL}/{folder}/{f2}")
        figs[name] = (f1, f2)
        print(f"{name}: judge {cls} {st['judge_target']['rate']:.3f} KID {st['kid_vs_all_test']['kid_x1e3']:.2f}"
              f" ± {st['kid_vs_all_test']['se_x1e3']:.2f} (tweedie {st['kid_vs_all_test_tweedie']['kid_x1e3']:.2f}) "
              f"sat {st['frac_samples_saturated']:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        _commit(vol)
    m = memo_check(cls, memo_sets, f"memo_{cls}{suffix}", folder=folder)["result"]
    out["memo_test_first512"] = m["test_first512"]
    for name, st in out["snapshots"].items():
        st["memo"] = m["sets"].get(f"prior_{name}")
    out["pick"] = pick05(out)
    out["minutes"] = (time.time() - t0) / 60
    path = _fresh(f"{VOL}/{folder}/snapstats_{cls}{suffix}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    _commit(vol)
    downloads = [f"{folder}/snapstats_{cls}{suffix}.json", f"{folder}/memo_{cls}{suffix}.json"]
    for name, (f1, f2) in figs.items():
        downloads.append(f"{folder}/{f2}")
        if name == out["pick"]["name"] or out["snapshots"][name]["reference"]:
            downloads.append(f"{folder}/{f1}")
    return {"task": f"snap_eval05 {cls}", "pick": out["pick"], "minutes": out["minutes"], "downloads": downloads}


def pick05(stats):
    """Iteration-05 rule (VALIDATION of iter 04): discard memo ratio < 0.5 (warn < 0.8); lowest KID; among snapshots
    within 1 se (of the lowest) of the lowest KID, take the highest judge rate not above ceiling + 2 points; if still
    tied, the latest step (then the longer EMA). The iteration-04 reference snapshot is reported, not eligible."""
    snaps = {k: v for k, v in stats["snapshots"].items() if not v.get("reference")}
    ceil = stats["ceiling_all_test"]["rate"]
    ok = {k: v for k, v in snaps.items() if v.get("memo") and v["memo"]["ratio_to_test_median"] >= MEMO_MIN_RATIO}
    warn = sorted(k for k, v in snaps.items() if v.get("memo") and v["memo"]["ratio_to_test_median"] < MEMO_WARN_RATIO)
    kidv = lambda k: snaps[k]["kid_vs_all_test"]["kid_x1e3"]  # noqa: E731
    if not ok:
        k = max(snaps, key=lambda k: (snaps[k].get("memo") or {}).get("ratio_to_test_median", 0))
        return {"name": k, "ckpt": snaps[k]["ckpt"], "prior_file": snaps[k]["prior_file"],
                "rule": "ALL snapshots flagged (memo < 0.5); took the highest memo ratio", "memo_warn": warn}
    best = min(ok, key=kidv)
    thr = kidv(best) + snaps[best]["kid_vs_all_test"]["se_x1e3"]
    near = [k for k in ok if kidv(k) <= thr]
    jr = lambda k: snaps[k]["judge_target"]["rate"]  # noqa: E731
    elig = [k for k in near if jr(k) <= ceil + 0.02] or near
    top = max(jr(k) for k in elig)
    tied = [k for k in elig if jr(k) == top]
    k = max(tied, key=lambda k: (snaps[k]["step"], snaps[k].get("ema") or 0))
    return {"name": k, "ckpt": snaps[k]["ckpt"], "prior_file": snaps[k]["prior_file"],
            "rule": "memo>=0.5; lowest KID; within 1 se -> highest judge <= ceiling+2pt; tie -> latest step",
            "lowest_kid": best, "kid_threshold": thr, "within_1se": sorted(near), "memo_warn": warn,
            "ceiling": ceil}


def chain_decay(cls, vol=None, smoke=False, cap_min=45.0, last=None, p0=0.0, folder=None, extra_snaps=()):
    """Container A (cat) / B (dog): resume_decay -> snap_eval05 on {50% e9999, 75% both, 100% both} + iter04 pick."""
    t0 = time.time()
    folder = folder or ("iter05/smoke" if smoke else "iter05")
    r1 = resume_decay(cls, last=last, folder=folder, cap_min=cap_min, max_steps=20 if smoke else None, p0=p0, vol=vol)
    sel = [dict(s, name=f"{cls}_{_etag(s['ema'])}_s{s['step']}") for s in r1["snapshots"]
           if not (s["frac"] == 0.5 and s["ema"] == 0.9995)] + list(extra_snaps)
    ref = torch.load(f"{VOL}/{REF04[cls]}", map_location="cpu", weights_only=False)
    sel.append({"name": f"{cls}_ref04_s{ref['step']}", "step": ref["step"], "path": REF04[cls],
                "minutes": ref["meta"]["train_minutes"], "ema": 0.9995, "reference": True})
    r2 = snap_eval05(cls, sel, folder=folder, n=16 if smoke else 512, steps=5 if smoke else 200,
                     n_boot=10 if smoke else 100, vol=vol)
    return {"task": f"chain_decay {cls}", "sub": [r1, r2], "pick": r2["pick"], "downloads": r2["downloads"],
            "minutes": (time.time() - t0) / 60}


def snap_eval_existing(cls, vol=None, folder="iter05", suffix="_recovered"):
    """Recovery: evaluate whatever iter05 snapshots of `cls` exist (after a cancellation)."""
    import glob
    import re
    t0, sel = time.time(), []
    for p in sorted(glob.glob(f"{VOL}/{folder}/{cls}_e*_s*.pth")):
        m = re.fullmatch(rf"{cls}_(e\d+)_s(\d+)\.pth", os.path.basename(p))
        if m:
            meta = torch.load(p, map_location="cpu", weights_only=False)["meta"]
            sel.append({"name": f"{cls}_{m.group(1)}_s{m.group(2)}", "step": int(m.group(2)),
                        "path": f"{folder}/{os.path.basename(p)}", "minutes": meta["train_minutes"], "ema": meta["ema"]})
    ref = torch.load(f"{VOL}/{REF04[cls]}", map_location="cpu", weights_only=False)
    sel.append({"name": f"{cls}_ref04_s{ref['step']}", "step": ref["step"], "path": REF04[cls],
                "minutes": ref["meta"]["train_minutes"], "ema": 0.9995, "reference": True})
    r = snap_eval05(cls, sel, folder=folder, vol=vol, suffix=suffix)
    return {"task": f"snap_eval_existing {cls}", "sub": [r], "downloads": r["downloads"], "minutes": (time.time() - t0) / 60}


# ------------------------------------------------------------------------------ latent Gaussianity
def gauss_stats(z, n_boot=200, seed=0):
    """Gaussianity diagnostics of a latent set z (N, 3, 32, 32) against N(0, I)."""
    from scipy import stats as sst
    z = z.double()
    N = len(z)
    flat = z.flatten(1)
    d = flat.shape[1]
    e0 = flat.pow(2).mean(1)                                    # per-image |z|^2 / d
    pooled = {k: F.avg_pool2d(z, k).flatten(1).pow(2).mean(1) * k * k for k in (4, 8)}   # iid N(0,1): E = 1
    h = (z[..., :, 1:] * z[..., :, :-1]).flatten(1).mean(1)
    v = (z[..., 1:, :] * z[..., :-1, :]).flatten(1).mean(1)
    zc = flat.flatten().numpy()
    g = torch.Generator().manual_seed(seed)
    boots = {k: [] for k in (4, 8)}
    for _ in range(n_boot):
        ix = torch.randint(N, (N,), generator=g)
        for k in (4, 8):
            boots[k].append(float(pooled[k][ix].mean()))
    out = {"n": N, "d": d, "mean": float(flat.mean()), "std": float(flat.std()),
           "norm2_over_d": {"mean": float(e0.mean()), "std": float(e0.std()), "q01": float(e0.quantile(0.01)),
                            "q50": float(e0.median()), "q99": float(e0.quantile(0.99)),
                            "iid_expected_std": math.sqrt(2 / d)},
           "lag1_autocorr_h": float(h.mean() / e0.mean()), "lag1_autocorr_v": float(v.mean() / e0.mean()),
           "ks_vs_N01": float(sst.kstest(zc, "norm").statistic), "ks_n_values": int(zc.size)}
    for k in (4, 8):
        out[f"pooled{k}_energy_ratio"] = float(pooled[k].mean())
        out[f"pooled{k}_energy_ratio_se"] = float(torch.tensor(boots[k]).std())
        out[f"pooled{k}_energy_ratio_normalised"] = float(pooled[k].mean() / e0.mean())
        out[f"pooled{k}_energy_ratio_q99_per_image"] = float(pooled[k].quantile(0.99))
    return out


def gauss_flags(st, ref):
    """Flag a latent set if a pooled energy ratio deviates from the N(0, I) reference by > 3 x its bootstrap se."""
    fl = {}
    for k in (4, 8):
        dev_ = st[f"pooled{k}_energy_ratio"] - ref[f"pooled{k}_energy_ratio"]
        se = max(st[f"pooled{k}_energy_ratio_se"], 1e-12)
        fl[f"pooled{k}_dev_in_se"] = dev_ / se
        fl[f"pooled{k}_flag"] = bool(abs(dev_) > 3 * se)
    fl["flagged"] = fl["pooled4_flag"] or fl["pooled8_flag"]
    return fl


# ------------------------------------------------------------------------------------ eval
def _kgrid(s_list, steps, eps):
    return sorted({min(steps, max(0, round((s - eps) / (1 - eps) * steps))) for s in s_list})


def _best_idx(p_out, cost, src, tgt, k=16):
    """Strict rule: p_target(out) >= 0.9 and p_source(out) <= 0.05, cost ascending. Fill (marked) by the highest
    p_target - p_source if fewer than k qualify."""
    good = ((p_out[:, tgt] >= 0.9) & (p_out[:, src] <= 0.05)).nonzero().flatten()
    idx = good[cost[good].argsort()][:k].tolist()
    n_fill = 0
    if len(idx) < k:
        rest = [i for i in (p_out[:, tgt] - p_out[:, src]).argsort(descending=True).tolist() if i not in idx]
        n_fill = k - len(idx)
        idx += rest[:n_fill]
    return idx, len(good), n_fill


def eval05(vol=None, smoke=False, folder="iter05", n=512, steps=200, eps=1e-3, n_steer=64, n_flow=8,
           ws=(0.25, 0.5, 1.0), n_cmp=32):
    from pf_ode import pf_ode
    t0, dev = time.time(), torch.device("cuda")
    if smoke:
        folder, n, steps, n_steer, n_cmp = "iter05/smoke", 8, 10, 8, 8
    FD = f"{VOL}/{folder}"
    snap = {}
    for c in ("cat", "dog"):
        p = f"{FD}/snapstats_{c}.json"
        if not os.path.exists(p) and os.path.exists(f"{FD}/snapstats_{c}_recovered.json"):
            p = f"{FD}/snapstats_{c}_recovered.json"
        if smoke and not os.path.exists(p):    # smoke: only the cat smoke model exists; use it for both
            p = f"{FD}/snapstats_cat.json"
        snap[c] = json.load(open(p))
    pick = {c: snap[c]["pick"] for c in snap}
    print("picked", {c: pick[c]["name"] for c in pick}, flush=True)
    sde, s5c = _score4(pick["cat"]["ckpt"], dev)
    _, s5d = _score4(pick["dog"]["ckpt"], dev)
    _, s4c = _score4(REF04["cat"], dev)
    _, s4d = _score4(REF04["dog"], dev)
    _, sFc = _score4(FT["cat"], dev)
    _, sFd = _score4(FT["dog"], dev)
    S5, S4, SF = {CAT: s5c, DOG: s5d}, {CAT: s4c, DOG: s4d}, {CAT: sFc, DOG: sFd}
    s_of = lambda k: eps + k * (sde.T - eps) / steps  # noqa: E731
    net = ev.load_judge(f"{VOL}/{JUDGE}", dev)
    res = {"picked": pick, "steps": steps, "eps": eps, "n": n, "best_pairs": {}, "saturated": {}, "runs_meta": {}}
    downloads, runs = [], []
    kq = _kgrid((0.2, 0.4, 0.6, 0.8, 1.0), steps, eps)                       # steering partial encodings
    kf_enc = _kgrid((0.1, 0.2, 0.35, 0.5, 0.75, 1.0), steps, eps)             # flow grid, source leg
    kf_dec = _kgrid((0.75, 0.5, 0.35, 0.2, 0.1), steps, eps)[::-1]           # flow grid, target leg (descending)
    enc_ks = sorted(set(kq) | set(kf_enc) | {steps})
    dec_ks = sorted(set(kf_dec) | {0}, reverse=True)
    keep, outs = {}, {}

    def save_transport(direction, name, x, y, ytw, note, extra):
        path = _fresh(f"{FD}/transport_{direction}_{name}.pt")
        m = {"s1.0": {"k": steps, "s": 1.0, "w": 0.0, "valid_coupling": True, "note": note},
             "s1.0_tw": {"k": steps, "s": 1.0, "w": 0.0, "valid_coupling": True,
                         "note": note + "; plus a final Tweedie denoise step x0=(x+sigma^2 score)/alpha at s=eps "
                                        "with the decoder (deterministic, still a map)"},
             "_run": {"direction": direction, "n": len(x), "steps": steps, "eps": eps, **extra,
                      "minutes": (time.time() - t0) / 60}}
        torch.save({"x": x, "outputs": {"s1.0": y, "s1.0_tw": ytw}, "meta": m}, path)
        runs.append({"name": f"{direction}_{name}", "kind": "transport", "direction": direction, "path": path,
                     "key": "s1.0"})
        runs.append({"name": f"{direction}_{name}_tw", "kind": "transport", "direction": direction, "path": path,
                     "key": "s1.0_tw"})
        outs[f"{direction}_{name}"] = {"x": x, "y": y, "ytw": ytw}
        res["runs_meta"][f"{direction}_{name}"] = m["_run"]
        res["saturated"][f"{direction}_{name}"] = sat_frac(y)
        res["saturated"][f"{direction}_{name}_tw"] = sat_frac(ytw)
        print(f"{direction} {name}: cost {ev.sq_cost(x, y).mean():.1f} (tw {ev.sq_cost(x, ytw).mean():.1f}) "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    # 1. composed ODE with the iteration-05 picks (states kept for the figures)
    for direction, (src, tgt) in DIRS.items():
        x, _ = cifar("test", src, n)
        enc, dec, zs, ys = {k: [] for k in enc_ks}, {k: [] for k in dec_ks}, [], []
        for i in range(0, len(x), 256):
            st = _segments(S5[src], sde, x[i:i + 256].to(dev), enc_ks, s_of, 0)
            zb = st[steps]
            dst = _segments(S5[tgt], sde, zb, [k for k in dec_ks if k != steps], s_of, steps)
            if i == 0:
                for k in enc_ks:
                    enc[k] = st[k][:n_steer].cpu()
                for k in dec_ks:
                    dec[k] = (zb if k == steps else dst[k])[:n_flow].cpu()
            zs.append(zb.cpu())
            ys.append(dst[0].cpu())
        z, y = torch.cat(zs), torch.cat(ys)
        ytw = tweedie(S5[tgt], sde, y, eps, dev)
        torch.save({"x": x, "z": z, "meta": {"direction": direction, "steps": steps, "eps": eps,
                                            "encoder": pick["cat" if src == CAT else "dog"]["ckpt"]}},
                   _fresh(f"{FD}/latents_{direction}.pt"))
        save_transport(direction, "iter05", x, y, ytw, "composed PF-ODE, iter05 decayed from-scratch models",
                       {"cat_ckpt": pick["cat"]["ckpt"], "dog_ckpt": pick["dog"]["ckpt"]})
        keep[direction] = {"x": x, "z": z, "enc": enc, "dec": dec, "y": y}
        _commit(vol)

    # 2. mixed pairs, decode only from saved latents
    mixed = {"M1": ("cat_to_dog", f"{VOL}/{FT_LATENTS['cat_to_dog']}", S5[DOG], pick["dog"]["ckpt"], FT["cat"]),
             "M2": ("dog_to_cat", f"{VOL}/{FT_LATENTS['dog_to_cat']}", S5[CAT], pick["cat"]["ckpt"], FT["dog"]),
             "M3": ("cat_to_dog", f"{FD}/latents_cat_to_dog.pt", SF[DOG], FT["dog"], pick["cat"]["ckpt"]),
             "M4": ("dog_to_cat", f"{FD}/latents_dog_to_cat.pt", SF[CAT], FT["cat"], pick["dog"]["ckpt"])}
    for mname, (direction, lat, sc, dec_ck, enc_ck) in mixed.items():
        d = torch.load(lat)
        x, z = d["x"][:n], d["z"][:n]
        assert torch.equal(x, keep[direction]["x"]), f"{lat}: inputs differ from the first {n} test images"
        y = torch.cat([pf_ode(sc, sde, z[i:i + 256].to(dev), sde.T, eps, steps).cpu() for i in range(0, len(z), 256)])
        ytw = tweedie(sc, sde, y, eps, dev)
        save_transport(direction, mname, x, y, ytw, f"mixed pair {mname}: encoder {enc_ck}, decoder {dec_ck}",
                       {"latents": lat, "encoder": enc_ck, "decoder": dec_ck})
        _commit(vol)

    # 3. FT baseline and iter04 outputs re-scored, plus their Tweedie step (one evaluation of their own decoder)
    for name, files, SS in (("FT", FT_TRANSPORT, SF), ("iter04", IT04_TRANSPORT, S4)):
        for direction, rel in files.items():
            src, tgt = DIRS[direction]
            d = torch.load(f"{VOL}/{rel}")
            x, y = d["x"][:n], d["outputs"]["s1.0"][:n]
            assert torch.equal(x, keep[direction]["x"]), f"{rel}: inputs differ"
            ytw = tweedie(SS[tgt], sde, y, eps, dev)
            path = _fresh(f"{FD}/transport_{direction}_{name}_tw.pt")
            torch.save({"x": x, "outputs": {"s1.0_tw": ytw}, "meta": {
                "s1.0_tw": {"valid_coupling": True, "note": f"{rel} outputs plus a final Tweedie step (own decoder)"},
                "_run": {"direction": direction, "source_file": rel}}}, path)
            runs.append({"name": f"{direction}_{name}", "kind": "transport", "direction": direction,
                         "path": f"{VOL}/{rel}", "key": "s1.0"} if not smoke else
                        {"name": f"{direction}_{name}", "kind": "transport", "direction": direction,
                         "path": _save_slice(f"{FD}/transport_{direction}_{name}_slice.pt", x, y), "key": "s1.0"})
            runs.append({"name": f"{direction}_{name}_tw", "kind": "transport", "direction": direction, "path": path,
                         "key": "s1.0_tw"})
            outs[f"{direction}_{name}"] = {"x": x, "y": y, "ytw": ytw}
            res["saturated"][f"{direction}_{name}"] = sat_frac(y)
            res["saturated"][f"{direction}_{name}_tw"] = sat_frac(ytw)
    _commit(vol)

    # 4. Gaussianity of every latent set
    gz = {"N01_fresh_seed1": torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(1)),
          "N01_prior_z_seed0": prior_z(n)}
    for direction in DIRS:
        enc_name = "cat" if direction == "cat_to_dog" else "dog"
        gz[f"FT_{enc_name}_encoder_iter02"] = torch.load(f"{VOL}/{FT_LATENTS[direction]}")["z"][:n]
        gz[f"iter04_{enc_name}_encoder"] = torch.load(f"{VOL}/{IT04_LATENTS[direction]}")["z"][:n]
        gz[f"iter05_{enc_name}_encoder"] = keep[direction]["z"]
    gres = {k: gauss_stats(v, n_boot=20 if smoke else 200) for k, v in gz.items()}
    ref = gres["N01_fresh_seed1"]
    for k in gres:
        gres[k]["flags_vs_N01_fresh"] = gauss_flags(gres[k], ref)
    res["gaussianity"] = gres
    for k, v in gres.items():
        print(f"gauss {k}: std {v['std']:.4f} pool4 {v['pooled4_energy_ratio']:.3f}±{v['pooled4_energy_ratio_se']:.3f}"
              f" pool8 {v['pooled8_energy_ratio']:.3f} lag1 {v['lag1_autocorr_h']:.4f} KS {v['ks_vs_N01']:.4f} "
              f"flag {v['flags_vs_N01_fresh']['flagged']}", flush=True)

    # 5. figures -------------------------------------------------------------------------------
    # 5a. flow grid of denoised estimates (Tweedie x0_hat(x_s)), and x_s / alpha_s (clamped) as a second row
    for direction, (src, tgt) in DIRS.items():
        K = keep[direction]
        sname, tname = ev.CLASSES[src], ev.CLASSES[tgt]
        nf = min(n_flow, len(K["x"]))
        cols_h, cols_r, labs = [K["x"][:nf]], [K["x"][:nf]], [f"input\n{sname}"]
        for k in kf_enc:
            xs = K["enc"][k][:nf]
            cols_h.append(tweedie(S5[src], sde, xs, s_of(k), dev))
            cols_r.append(xs / _alpha_sigma(sde, s_of(k), dev)[0])
            labs.append(f"{sname} leg\ns={s_of(k):.2f}")
        for k in kf_dec:
            xs = K["dec"][k][:nf]
            cols_h.append(tweedie(S5[tgt], sde, xs, s_of(k), dev))
            cols_r.append(xs / _alpha_sigma(sde, s_of(k), dev)[0])
            labs.append(f"{tname} leg\ns={s_of(k):.2f}")
        cols_h.append(K["y"][:nf])
        cols_r.append(K["y"][:nf])
        labs.append(f"output\n{tname}")
        rows, rl = [], []
        for r in range(nf):
            rows += [torch.stack([c_[r] for c_ in cols_h]), torch.stack([c_[r] for c_ in cols_r])]
            rl += [f"#{r} x0hat", f"#{r} x_s/a_s"]
        fname = "flow_grid_big.png" if direction == "cat_to_dog" else "flow_grid_dog_to_cat_big.png"
        labeled_grid(rows, f"{FD}/{fname}", col_labels=labs, row_labels=rl,
                     title=f"PF-ODE flow {sname} -> noise -> {tname} (iter05 models). Odd rows: denoised estimate "
                           f"x0hat=(x_s+sigma^2 score)/alpha at the nearest Heun grid point; even rows: x_s/alpha_s "
                           f"clamped. {sname} leg uses the {sname} model, {tname} leg the {tname} model.")
        downloads.append(f"{folder}/{fname}")

    # 5b. steering grid (cat -> dog), first n_steer cats; only s*=1, w=0 is a coupling
    K = keep["cat_to_dog"]
    xs = K["x"][:n_steer]
    cols, labels, stats = [], [], []
    for k in kq:
        yk = K["y"][:n_steer] if k == steps else pf_ode(s5d, sde, K["enc"][k][:n_steer].to(dev), s_of(k), eps, k).cpu()
        cols.append(yk)
        labels.append(f"s*={s_of(k):.1f}\nw=0")
    for w in ws:
        def sg(xx, ss, w=w):
            return (1 + w) * s5d(xx, ss) - w * s5c(xx, ss)
        cols.append(pf_ode(sg, sde, K["z"][:n_steer].to(dev), sde.T, eps, steps).cpu())
        labels.append(f"s*=1.0\nw={w}")
    for lab, yk in zip(labels, cols):
        p = ev.judge_probs(net, yk, dev)
        stats.append({"setting": lab.replace("\n", " "), "valid_coupling": lab == "s*=1.0\nw=0",
                      "cost_mean": float(ev.sq_cost(xs, yk).mean()),
                      "cost_clamped_mean": float(ev.sq_cost(xs, yk.clamp(-1, 1)).mean()),
                      "judge_dog_rate": float((p.argmax(1) == DOG).float().mean()),
                      "judge_dog_mean_prob": float(p[:, DOG].mean()), "n": len(xs)})
    allc = [xs[:n_flow]] + [c_[:n_flow] for c_ in cols]
    rows = [torch.stack([c_[r] for c_ in allc]) for r in range(min(n_flow, len(xs)))]
    foot = [f"per column over the first {len(xs)} test cats: mean cost (clamped) / judge dog-rate",
            "  ".join(f"[{s_['setting']}] {s_['cost_clamped_mean']:.0f} / {s_['judge_dog_rate']:.2f}" for s_ in stats[:5]),
            "  ".join(f"[{s_['setting']}] {s_['cost_clamped_mean']:.0f} / {s_['judge_dog_rate']:.2f}" for s_ in stats[5:]),
            "GREEN frame = s*=1, w=0: the only valid coupling. Partial encodings (s*<1) and guidance (w>0) are "
            "diagnostics, NOT couplings."]
    labeled_grid(rows, f"{FD}/steering_grid_big.png", col_labels=["input cat"] + labels,
                 title="steering strength, cat -> dog (iter05 models): partial encoding s*, then density-ratio "
                       "guidance w at s*=1", footer=foot, highlight=1 + len(kq) - 1)
    res["steering"] = {"columns": stats, "n_images": len(xs),
                       "guidance": "decode with (1+w) s_dog - w s_cat (tilted law, NOT nu)",
                       "partial": "encode with cat model to s*, decode with dog model from s* (NOT a coupling for s*<1)"}
    downloads.append(f"{folder}/steering_grid_big.png")
    _commit(vol)

    # 5c. best pairs (strict rule) for the iter05 pair, the mixed pairs and FT
    for key, o in outs.items():
        direction = "cat_to_dog" if key.startswith("cat_to_dog") else "dog_to_cat"
        name = key[len(direction) + 1:]
        if name == "iter04":
            continue
        src, tgt = DIRS[direction]
        x, y = o["x"], o["y"]
        p_out = ev.judge_probs(net, y, dev)
        cost = ev.sq_cost(x, y.clamp(-1, 1))
        idx, n_good, n_fill = _best_idx(p_out, cost, src, tgt)
        res["best_pairs"][key] = {"rule": "p_target(out)>=0.9 and p_source(out)<=0.05, clamped cost ascending; "
                                          "filled by highest p_target-p_source if fewer than 16",
                                  "n_qualifying": n_good, "n_filled": n_fill, "idx": idx,
                                  "cost": cost[idx].tolist(), "p_target": p_out[idx, tgt].tolist(),
                                  "p_source": p_out[idx, src].tolist()}
        if name == "iter05":
            bname = "best_pairs_big.png" if direction == "cat_to_dog" else "best_pairs_dog_to_cat_big.png"
        else:
            bname = f"best_pairs_{key}_big.png"
        sn, tn = ev.CLASSES[src], ev.CLASSES[tgt]
        labeled_grid([x[idx], y[idx]], f"{FD}/{bname}",
                     col_labels=[f"#{i} c={cost[i]:.0f}\np={p_out[i, tgt]:.2f}" for i in idx],
                     title=f"best {sn}->{tn} pairs, {key} (top input, bottom output); strict rule P({tn})>=0.9 and "
                           f"P({sn})<=0.05; {n_good}/{len(x)} qualify" + (f"; last {n_fill} are fill" if n_fill else ""))
        downloads.append(f"{folder}/{bname}")

    # 5d. same noise -> new models, iter04 models, FT models
    pc, pd = torch.load(pick["cat"]["prior_file"]), torch.load(pick["dog"]["prior_file"])
    assert torch.equal(pc["z"], pd["z"])
    same = {}

    def same_scene(a, b, name, a_cls=CAT, b_cls=DOG):
        m = min(len(a), len(b))
        a, b = a[:m].clamp(-1, 1), b[:m].clamp(-1, 1)
        pa, pb = ev.judge_probs(net, a, dev), ev.judge_probs(net, b, dev)
        same[name] = {"n": m, "matched_cost_mean": float(ev.sq_cost(a, b).mean()),
                      "mismatched_cost_mean": float(ev.sq_cost(a, b.roll(1, 0)).mean()),
                      "frac_both_correct_class": float(((pa.argmax(1) == a_cls) & (pb.argmax(1) == b_cls)).float().mean())}
        same[name]["matched_over_mismatched"] = same[name]["matched_cost_mean"] / same[name]["mismatched_cost_mean"]

    same_scene(pc["samples"], pd["samples"], "iter05_cat_vs_iter05_dog")
    rows, rl = [pc["samples"][:16], pd["samples"][:16]], ["iter05 cat", "iter05 dog"]
    o4c, o4d = f"{VOL}/iter04/prior_cat_s17297.pt", f"{VOL}/iter04/prior_dog_s20119.pt"
    oFc, oFd = f"{VOL}/iter02/prior_cat.pt", f"{VOL}/iter02/prior_dog.pt"
    for lab, (pa_, pb_) in (("iter04", (o4c, o4d)), ("FT", (oFc, oFd))):
        if os.path.exists(pa_) and os.path.exists(pb_):
            A, B = torch.load(pa_), torch.load(pb_)
            m = min(len(A["z"]), len(pc["z"]))
            if torch.equal(A["z"][:m], pc["z"][:m]) and torch.equal(B["z"][:m], pc["z"][:m]):
                same_scene(A["samples"], B["samples"], f"{lab}_cat_vs_{lab}_dog")
                same_scene(pc["samples"], A["samples"], f"iter05_cat_vs_{lab}_cat", CAT, CAT)
                same_scene(pd["samples"], B["samples"], f"iter05_dog_vs_{lab}_dog", DOG, DOG)
                rows += [A["samples"][:16], B["samples"][:16]]
                rl += [f"{lab} cat", f"{lab} dog"]
    labeled_grid(rows, f"{FD}/same_noise_both_models_big.png", row_labels=rl,
                 title=f"same seed-0 noise decoded by iter05 cat ({pick['cat']['name']}) / dog ({pick['dog']['name']}), "
                       f"iter04 picks (100-step decodes) and fine-tuned models (200-step decodes)")
    res["same_noise"] = same
    downloads.append(f"{folder}/same_noise_both_models_big.png")
    _commit(vol)

    # 5e. 32-column comparison per direction: input / FT / iter04 / iter05 / mixed (FT enc -> new dec) /
    #     mixed (new enc -> FT dec)
    for direction, mA, mB in (("cat_to_dog", "M1", "M3"), ("dog_to_cat", "M2", "M4")):
        x = keep[direction]["x"][:n_cmp]
        rows = [x] + [outs[f"{direction}_{k}"]["y"][:n_cmp] for k in ("FT", "iter04", "iter05", mA, mB)]
        rl = ["input", "FT (iter01)", "iter04", "iter05", f"{mA} FTenc>new", f"{mB} newenc>FT"]
        labeled_grid(rows, f"{FD}/compare_{direction}_big.png", row_labels=rl,
                     col_labels=[f"#{i}" for i in range(len(x))],
                     title=f"{direction}, test images 0-{len(x) - 1}: rows input / fine-tuned pair / iter04 scratch / "
                           f"iter05 decayed / {mA} (FT encoder -> iter05 decoder) / {mB} (iter05 encoder -> FT decoder)")
        downloads.append(f"{folder}/compare_{direction}_big.png")

    # 6. standard evaluation (judge, cost clamped/unclamped, KID vs floor) + the chosen priors
    for c in ("cat", "dog"):
        runs.append({"name": f"prior_{pick[c]['name']}", "kind": "prior", "path": pick[c]["prior_file"], "cls": CLS[c]})
    eval_runs(runs, "evalT", folder=folder, kid_refs=True, n_boot=10 if smoke else 200)
    small = torch.load(f"{FD}/evalT_small.pt")
    for key in [f"{d}_{m}" for d in DIRS for m in ("iter05", "M1", "M2", "M3", "M4")]:
        if key not in small:
            continue
        dd = small[key]
        fn = f"{key}_lowest_ptarget_big.png"
        ev.grid_png([dd["worst_x"], dd["worst_y"]], f"{FD}/{fn}")
        downloads.append(f"{folder}/{fn}")
    res["minutes"] = (time.time() - t0) / 60
    with open(_fresh(f"{FD}/eval05.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    _commit(vol)
    downloads += [f"{folder}/eval05.json", f"{folder}/evalT_stats.json"]
    return {"task": "eval05", "minutes": res["minutes"], "downloads": downloads}


def _save_slice(path, x, y):
    torch.save({"x": x, "outputs": {"s1.0": y}, "meta": {"s1.0": {"note": "smoke slice"}, "_run": {}}}, _fresh(path))
    return path


def chain_smoke(vol=None):
    """smoke: 20 resume steps (cat) + snap_eval n=16/5 steps, then eval05 at n=8/10 steps (every figure, mixed pairs,
    Gaussianity)."""
    t0 = time.time()
    r1 = chain_decay("cat", vol=vol, smoke=True)
    r2 = eval05(vol=vol, smoke=True)
    return {"task": "chain_smoke", "sub": [r1, r2], "downloads": r1["downloads"] + r2["downloads"],
            "minutes": (time.time() - t0) / 60}
