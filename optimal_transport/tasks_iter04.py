"""GPU tasks for iteration 04: cat-only and dog-only models trained FROM SCRATCH (random init).

Run inside Modal via modal_iter04.py. All outputs go to /vol/iter04/; nothing existing is overwritten.

Architecture: score_sde_pytorch DDPM++ continuous VP (config vp/cifar10_ddpmpp_continuous) with the smaller
OVERRIDES below (9.86M parameters instead of 61.8M). Every saved checkpoint carries "config_overrides", and
common/score_sde_backend.load_score_model_auto rebuilds the matching architecture from it.

train():          one class, CIFAR-10 TRAIN images of that class only, EMA snapshots /vol/iter04/{cls}_s{step}.pth
snap_eval():      per snapshot: 256 prior samples (seed-0 noise, PF-ODE Heun), judge rate, KID vs real test
                  images of the class, memorisation check (tasks_iter03.memo_check) -> snapstats_{cls}.json
chain_train():    train -> snap_eval (one A10G container per class)
eval_composed():  pick one snapshot per class (rule in pick_snapshot), composed PF-ODE both directions on the
                  first 512 test cats / dogs, standard eval (tasks_iter02.eval_runs), flow grids, steering grid,
                  best pairs, same-noise figure; all figures rendered in the container (4x NEAREST).
"""
import json
import os
import time

import torch

import evaluate as ev
from tasks_iter import CAT, CONFIG, DIRS, DOG, REPO, VOL, _fresh, cifar
from tasks_iter02 import _Incep, eval_runs, kid, prior_z
from tasks_iter03 import memo_check

CLS = {"cat": CAT, "dog": DOG}
OVERRIDES = {"nf": 64, "ch_mult": [1, 2, 2, 2], "num_res_blocks": 2, "attn_resolutions": [16], "dropout": 0.1}
JUDGE = "judge/cifar10_judge.pth"
MEMO_MIN_RATIO = 0.5


def _score4(ckpt, dev):
    from score_sde_backend import load_score_model_auto
    _, sde, _, s = load_score_model_auto(REPO, CONFIG, f"{VOL}/{ckpt}", dev)
    return sde, s


def _commit(vol):
    if vol is not None:
        vol.commit()


# ----------------------------------------------------------------------------------------- train
def train(cls, folder="iter04", tag="", bs=256, lr=2e-4, warmup=1000, ema_max=0.9995, cap_min=95.0,
          snap_fracs=(0.125, 0.25, 0.5, 0.75, 1.0), max_steps=None, seed=0, vol=None, init=None):
    """From-scratch training. Snapshots at wall-clock fractions of cap_min (or step fractions of max_steps).
    init: optional iter-04 EMA snapshot (volume-relative) to resume from (fresh Adam; step count, EMA schedule and
    train_minutes continue from the snapshot)."""
    from score_sde_backend import build_network, load_model_auto
    dev = torch.device("cuda")
    torch.manual_seed(seed + (0 if init is None else 1))
    step0, min0, init_note = 0, 0.0, "random (score_sde default init), no pretrained weights"
    if init is None:
        config, sde, eps, model = build_network(REPO, CONFIG, dev, OVERRIDES)
    else:
        config, sde, eps, model = load_model_auto(REPO, CONFIG, f"{VOL}/{init}", dev)
        ck = torch.load(f"{VOL}/{init}", map_location="cpu", weights_only=False)
        step0, min0 = ck["step"], ck["meta"]["train_minutes"]
        init_note = (f"random init, trained {step0} steps ({min0:.1f} min); resumed from EMA snapshot {init} "
                     f"with fresh Adam (warmup {warmup}) after the first containers were cancelled")
        print(f"resume from {init}: step {step0}, {min0:.1f} train-min", flush=True)
    n_par = sum(p.numel() for p in model.parameters())
    model.train()
    xa, _ = cifar("train", CLS[cls])                 # [-1,1] floats, CIFAR-10 TRAIN images of this class only
    xa = xa.to(dev)
    print(f"train {cls}: {len(xa)} train images, {n_par} params, bs {bs}, cap {cap_min} min, "
          f"max_steps {max_steps}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = [p.detach().clone() for p in model.parameters()]
    snaps = []

    def save(step, elapsed):
        step = step0 + step
        elapsed = min0 + elapsed
        path = _fresh(f"{VOL}/{folder}/{tag}{cls}_s{step}.pth")
        backup = [p.detach().clone() for p in model.parameters()]
        for p_, e in zip(model.parameters(), ema):
            p_.data.copy_(e)
        torch.save({"model": model.state_dict(), "ema": {"shadow_params": []}, "step": step,
                    "config_name": CONFIG, "config_overrides": OVERRIDES,
                    "meta": {"init": init_note, "cls": cls, "bs": bs,
                             "lr": lr, "warmup": warmup, "ema": f"min({ema_max}, (1+t)/(10+t))", "grad_clip": 1.0,
                             "bf16_autocast": True, "train_minutes": elapsed, "n_params": n_par,
                             "data": f"CIFAR-10 train {cls} only ({len(xa)} images), random hflip"}}, path)
        for p_, b in zip(model.parameters(), backup):
            p_.data.copy_(b)
        snaps.append({"step": step, "path": f"{folder}/{tag}{cls}_s{step}.pth", "minutes": elapsed})
        print(f"saved snapshot {path} ({elapsed:.1f} min)", flush=True)
        _commit(vol)

    t0 = time.time()
    step, running, log, next_snap = 0, 0.0, [], 0
    fracs = list(snap_fracs)

    def progress():
        return step / max_steps if max_steps else (time.time() - t0) / 60 / cap_min

    while progress() < 1.0:
        x = xa[torch.randint(len(xa), (bs,), device=dev)]
        flip = torch.rand(bs, device=dev) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        s = torch.rand(bs, device=dev) * (sde.T - eps) + eps
        z = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(mean + std[:, None, None, None] * z, s * 999)   # network predicts the added noise
        loss = (out.float() - z).pow(2).mean()
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warmup)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        decay = min(ema_max, (1 + step0 + step) / (10 + step0 + step))
        with torch.no_grad():
            torch._foreach_mul_(ema, decay)
            torch._foreach_add_(ema, [p.detach() for p in model.parameters()], alpha=1 - decay)
        step += 1
        running += loss.item() if (step % 50 == 0 or max_steps) else 0.0
        if step % 500 == 0 or (max_steps and step % 5 == 0):
            el = (time.time() - t0) / 60
            k = 5 if max_steps else 10
            log.append((step, running / k, el))
            print(f"step {step} loss {running / k:.4f} {el:.1f} min ({step / max(el, 1e-6) / 60:.2f} it/s) "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            running = 0.0
        if next_snap < len(fracs) - 1 and progress() >= fracs[next_snap]:
            save(step, (time.time() - t0) / 60)
            next_snap += 1
    el = (time.time() - t0) / 60
    save(step, el)                                  # final snapshot (fraction 1.0)
    last = _fresh(f"{VOL}/{folder}/{tag}{cls}_last.pth")
    torch.save({"model": model.state_dict(), "ema": {"shadow_params": [e.cpu() for e in ema]},
                "optimizer": opt.state_dict(), "step": step0 + step, "config_name": CONFIG, "config_overrides": OVERRIDES},
               last)
    _commit(vol)
    return {"task": f"train {cls}", "init": init, "step0": step0, "steps": step, "images_seen": step * bs, "epochs": step * bs / len(xa),
            "n_params": n_par, "loss_log": log, "snapshots": snaps, "it_per_s": step / max(el * 60, 1e-6),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "minutes": el}


# ------------------------------------------------------------------------------------ snapshot eval
def snap_eval(cls, snaps, folder="iter04", tag="", n=256, steps=100, eps=1e-3, n_boot=100, vol=None, suffix=""):
    from pf_ode import pf_ode
    t0, dev, c = time.time(), torch.device("cuda"), CLS[cls]
    net = ev.load_judge(f"{VOL}/{JUDGE}", dev)
    inc = _Incep(dev)
    real = cifar("test", c)[0]
    freal = inc.feats(real)
    out = {"cls": cls, "n_samples": n, "heun_steps": steps, "eps": eps, "noise": "prior_z(n), seed 0",
           "kid_floor_train_vs_test": kid(inc.feats(cifar("train", c, n)[0]), freal, n_boot),
           "ceiling_all_test": ev.rate_stats(ev.judge_probs(net, real, dev), c), "snapshots": {}}
    z = prior_z(n)
    downloads, memo_sets = [], {}
    for sn in snaps:
        name = f"{cls}_s{sn['step']}"
        sde, sc = _score4(sn["path"], dev)
        y = torch.cat([pf_ode(sc, sde, z[i:i + 256].to(dev), sde.T, eps, steps).cpu() for i in range(0, n, 256)])
        ppath = _fresh(f"{VOL}/{folder}/{tag}prior_{name}.pt")
        torch.save({"z": z, "samples": y, "meta": {"ckpt": sn["path"], "seed": 0, "steps": steps, "eps": eps, "n": n}},
                   ppath)
        memo_sets[f"prior_{name}"] = ppath
        p = ev.judge_probs(net, y, dev)
        worst = p[:, c].argsort()[:16]
        st = {"step": sn["step"], "train_minutes": sn["minutes"], "ckpt": sn["path"], "prior_file": ppath,
              "judge_target": ev.rate_stats(p, c), "judge_cat": ev.rate_stats(p, CAT), "judge_dog": ev.rate_stats(p, DOG),
              "class_histogram": torch.bincount(p.argmax(1), minlength=10).tolist(),
              "kid_vs_all_test": kid(inc.feats(y), freal, n_boot),
              "frac_pixels_outside_[-1,1]": float(((y < -1) | (y > 1)).float().mean())}
        out["snapshots"][name] = st
        f1, f2 = f"{tag}prior_{name}_big.png", f"{tag}prior_{name}_lowest_ptarget_big.png"
        ev.grid_png([y[:16], y[16:32]], f"{VOL}/{folder}/{f1}")
        ev.grid_png([y[worst]], f"{VOL}/{folder}/{f2}")
        downloads += [f"{folder}/{f1}", f"{folder}/{f2}"]
        print(f"{name}: judge {cls} {st['judge_target']['rate']:.3f} KID {st['kid_vs_all_test']['kid_x1e3']:.2f}"
              f" ({(time.time() - t0) / 60:.1f} min)", flush=True)
        _commit(vol)
    m = memo_check(cls, memo_sets, f"{tag}memo_{cls}{suffix}", folder=folder)["result"]
    out["memo_test_first512"] = m["test_first512"]
    for name, st in out["snapshots"].items():
        st["memo"] = m["sets"].get(f"prior_{name}")
    out["pick"] = pick_snapshot(out)
    out["minutes"] = (time.time() - t0) / 60
    path = _fresh(f"{VOL}/{folder}/{tag}snapstats_{cls}{suffix}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    _commit(vol)
    downloads += [f"{folder}/{tag}snapstats_{cls}{suffix}.json", f"{folder}/{tag}memo_{cls}{suffix}.json"]
    return {"task": f"snap_eval {cls}", "pick": out["pick"], "minutes": out["minutes"], "downloads": downloads}


def pick_snapshot(stats):
    """Rule: among snapshots whose memorisation ratio (median NN-train distance of samples / same for real
    test images) is >= 0.5, take the one with the lowest KID; ties impossible in practice. If every snapshot
    is flagged, take the one with the highest ratio (and say so)."""
    snaps = stats["snapshots"]
    ok = {k: v for k, v in snaps.items() if v.get("memo") and v["memo"]["ratio_to_test_median"] >= MEMO_MIN_RATIO}
    if ok:
        k = min(ok, key=lambda k: ok[k]["kid_vs_all_test"]["kid_x1e3"])
        return {"name": k, "ckpt": snaps[k]["ckpt"], "prior_file": snaps[k]["prior_file"],
                "rule": f"lowest KID among snapshots with memo ratio >= {MEMO_MIN_RATIO}", "candidates": sorted(ok)}
    k = max(snaps, key=lambda k: (snaps[k].get("memo") or {}).get("ratio_to_test_median", 0))
    return {"name": k, "ckpt": snaps[k]["ckpt"], "prior_file": snaps[k]["prior_file"],
            "rule": "ALL snapshots flagged; took the highest memo ratio", "candidates": []}


def chain_train(cls, vol=None, smoke=False, tag="", folder="iter04", cap_min=95.0):
    t0 = time.time()
    r1 = train(cls, folder=folder, tag=tag, cap_min=cap_min, max_steps=30 if smoke else None, vol=vol)
    r2 = snap_eval(cls, r1["snapshots"], folder=folder, tag=tag, n=16 if smoke else 256, steps=5 if smoke else 100,
                   n_boot=10 if smoke else 100, vol=vol)
    return {"task": f"chain_train {cls}", "sub": [r1, r2], "downloads": r2["downloads"],
            "minutes": (time.time() - t0) / 60}


def chain_resume(cls, init, cap_min=40.0, vol=None, folder="iter04"):
    """Resume from an EMA snapshot for cap_min more minutes, evaluate the new snapshots, and merge them with the
    earlier snapstats_{cls}.json into snapstats_{cls}_all.json (with the pick over all snapshots)."""
    t0 = time.time()
    r1 = train(cls, folder=folder, cap_min=cap_min, warmup=200, snap_fracs=(0.25, 0.5, 0.75, 1.0), vol=vol, init=init)
    r2 = snap_eval(cls, r1["snapshots"], folder=folder, vol=vol, suffix="_resume")
    old = json.load(open(f"{VOL}/{folder}/snapstats_{cls}.json"))
    new = json.load(open(f"{VOL}/{folder}/snapstats_{cls}_resume.json"))
    allst = {**new, "snapshots": {**old["snapshots"], **new["snapshots"]}, "resumed_from": init}
    allst["pick"] = pick_snapshot(allst)
    with open(_fresh(f"{VOL}/{folder}/snapstats_{cls}_all.json"), "w") as fh:
        json.dump(allst, fh, indent=1)
    _commit(vol)
    return {"task": f"chain_resume {cls}", "sub": [r1, r2], "pick": allst["pick"],
            "downloads": r2["downloads"] + [f"{folder}/snapstats_{cls}_all.json"], "minutes": (time.time() - t0) / 60}


def snap_eval_from_volume(classes=("cat", "dog"), folder="iter04", vol=None):
    """Recovery path (training containers were cancelled at ~67 min): evaluate every non-smoke EMA snapshot
    /vol/iter04/{cls}_s*.pth that exists on the volume, per class, with snap_eval."""
    import glob
    import re
    t0, res, downloads = time.time(), [], []
    for cls in classes:
        snaps = []
        for p in glob.glob(f"{VOL}/{folder}/{cls}_s*.pth"):
            m = re.fullmatch(rf"{cls}_s(\d+)\.pth", os.path.basename(p))
            if not m:
                continue
            meta = torch.load(p, map_location="cpu", weights_only=False)["meta"]
            snaps.append({"step": int(m.group(1)), "path": f"{folder}/{os.path.basename(p)}",
                          "minutes": meta["train_minutes"]})
        snaps.sort(key=lambda d: d["step"])
        print(f"{cls}: snapshots {[d['step'] for d in snaps]}", flush=True)
        r = snap_eval(cls, snaps, folder=folder, vol=vol)
        res.append(r)
        downloads += r["downloads"]
    return {"task": "snap_eval_from_volume", "sub": res, "downloads": downloads, "minutes": (time.time() - t0) / 60}


# ------------------------------------------------------------------------------------- figures
def labeled_grid(rows, path, col_labels=None, title=None, footer=None, row_labels=None, scale=4, highlight=None):
    """rows: list of (N,3,32,32) tensors in [-1,1] (one image row each). 4x NEAREST, text header/footer.
    highlight: column index drawn with a green frame (the valid-coupling column)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    ncol = max(len(r) for r in rows)
    canvas = torch.ones(3, len(rows) * 34 - 2, ncol * 34 - 2)
    for r, t in enumerate(rows):
        for c in range(len(t)):
            canvas[:, r * 34:r * 34 + 32, c * 34:c * 34 + 32] = (t[c].float().clamp(-1, 1) + 1) / 2
    im = Image.fromarray((canvas.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    left = 90 if row_labels else 0
    top = (22 if title else 0) + (40 if col_labels else 0)
    nfoot = len(footer or [])
    W, H = im.width + left, im.height + top + 20 * nfoot + (6 if nfoot else 0)
    out = Image.new("RGB", (W, H), "white")
    out.paste(im, (left, top))
    d = ImageDraw.Draw(out)
    cw = 34 * scale
    if title:
        d.text((4, 3), title, fill="black", font=font)
    if col_labels:
        for c, lab in enumerate(col_labels):
            for j, line in enumerate(str(lab).split("\n")[:2]):
                d.text((left + c * cw + 3, top - 40 + j * 18), line, fill="black", font=font)
    if row_labels:
        for r, lab in enumerate(row_labels):
            d.text((3, top + r * cw + 50), str(lab), fill="black", font=font)
    if highlight is not None:
        x0 = left + highlight * cw - 3
        d.rectangle([x0, top - 3, x0 + 32 * scale + 5, top + im.height + 2], outline=(0, 160, 0), width=4)
    for j, line in enumerate(footer or []):
        d.text((4, top + im.height + 4 + 20 * j), line, fill="black", font=font)
    out.save(path)


# ------------------------------------------------------------------------------- composed ODE eval
def _segments(score, sde, x, ks, s_of, k_from):
    """Integrate the PF-ODE through the grid indices ks (from k_from), returning the state at each k."""
    from pf_ode import pf_ode
    states, k_prev = {}, k_from
    for k in ks:
        x = pf_ode(score, sde, x, s_of(k_prev), s_of(k), abs(k - k_prev))
        states[k], k_prev = x, k
    return states


def eval_composed(vol=None, smoke=False, tag="", folder="iter04", n=512, steps=200, eps=1e-3, n_steer=64,
                  n_flow=8, ws=(0.25, 0.5, 1.0)):
    from pf_ode import pf_ode
    t0, dev = time.time(), torch.device("cuda")
    if smoke:
        n, steps, n_steer = 8, 10, 8
    F = f"{VOL}/{folder}"
    snap = {}
    for c in ("cat", "dog"):
        p = f"{F}/{tag}snapstats_{c}_all.json"
        if not os.path.exists(p):
            p = f"{F}/{tag}snapstats_{c}.json"
        if smoke and not os.path.exists(p):  # smoke: only the cat smoke model exists; use it for both
            p = f"{F}/{tag}snapstats_cat.json"
        snap[c] = json.load(open(p))
    pick = {c: snap[c]["pick"] for c in snap}
    print("picked", {c: pick[c]["name"] for c in pick}, flush=True)
    sde, s_cat = _score4(pick["cat"]["ckpt"], dev)
    _, s_dog = _score4(pick["dog"]["ckpt"], dev)
    score = {CAT: s_cat, DOG: s_dog}
    s_of = lambda k: eps + k * (sde.T - eps) / steps  # noqa: E731
    kq = [round(q * steps) for q in (0.2, 0.4, 0.6, 0.8, 1.0)]       # s* ~ 0.2 ... 1.0 on the Heun grid
    net = ev.load_judge(f"{VOL}/{JUDGE}", dev)
    res = {"picked": pick, "steps": steps, "eps": eps, "n": n, "flow": {}, "best_pairs": {}}
    downloads, runs = [], []
    keep = {}
    for direction, (src, tgt) in DIRS.items():
        x, _ = cifar("test", src, n)
        enc, dec, ys = {k: [] for k in kq}, {k: [] for k in kq[:-1]}, []
        for i in range(0, len(x), 256):
            st = _segments(score[src], sde, x[i:i + 256].to(dev), kq, s_of, 0)           # encode eps -> 1
            zb = st[kq[-1]]
            dst = _segments(score[tgt], sde, zb, kq[-2::-1] + [0], s_of, kq[-1])          # decode 1 -> eps
            for k in kq:
                enc[k].append(st[k].cpu())
            for k in kq[:-1]:
                dec[k].append(dst[k].cpu())
            ys.append(dst[0].cpu())
        enc = {k: torch.cat(v) for k, v in enc.items()}
        dec = {k: torch.cat(v) for k, v in dec.items()}
        y, z = torch.cat(ys), enc[kq[-1]]
        torch.save({"x": x, "z": z, "meta": {"direction": direction, "steps": steps, "eps": eps,
                                            "encoder": pick["cat" if src == CAT else "dog"]["ckpt"]}},
                   _fresh(f"{F}/{tag}latents_{direction}.pt"))
        tpath = _fresh(f"{F}/{tag}transport_{direction}_scratch.pt")
        meta = {"s1.0": {"k": steps, "s": 1.0, "w": 0.0, "valid_coupling": True,
                         "note": "composed PF-ODE, from-scratch iter04 models (encode source, decode target)"},
                "_run": {"direction": direction, "n": len(x), "steps": steps, "eps": eps,
                         "cat_ckpt": pick["cat"]["ckpt"], "dog_ckpt": pick["dog"]["ckpt"],
                         "minutes": (time.time() - t0) / 60}}
        torch.save({"x": x, "outputs": {"s1.0": y}, "meta": meta}, tpath)
        runs.append({"name": f"{direction}_scratch", "kind": "transport", "direction": direction, "path": tpath})
        keep[direction] = {"x": x, "y": y, "z": z, "enc": enc}
        print(f"{direction}: cost {ev.sq_cost(x, y).mean():.1f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        # flow grid: input | encode s=.2,.4,.6,.8,1 | decode s=.8,.6,.4,.2 | output
        cols = [x[:n_flow]] + [enc[k][:n_flow] for k in kq] + [dec[k][:n_flow] for k in kq[-2::-1]] + [y[:n_flow]]
        rows = [torch.stack([c_[r] for c_ in cols]) for r in range(min(n_flow, len(x)))]
        sname, tname = ev.CLASSES[src], ev.CLASSES[tgt]
        labs = [f"input\n{sname}"] + [f"enc {sname}\ns={s_of(k):.1f}" for k in kq] + \
               [f"dec {tname}\ns={s_of(k):.1f}" for k in kq[-2::-1]] + [f"output\n{tname}"]
        fname = f"{tag}flow_grid_big.png" if direction == "cat_to_dog" else f"{tag}flow_grid_dog_to_cat_big.png"
        labeled_grid(rows, f"{F}/{fname}", col_labels=labs,
                     title=f"PF-ODE flow {sname} -> noise -> {tname} (from-scratch models; raw ODE states clamped "
                           f"to [-1,1]; {steps} Heun steps)")
        downloads.append(f"{folder}/{fname}")
        # best pairs: judge p_target(out) >= 0.9 and p_source(in) >= 0.5, lowest cost first
        p_out, p_in = ev.judge_probs(net, y, dev), ev.judge_probs(net, x, dev)
        cost = ev.sq_cost(x, y)
        good = ((p_out[:, tgt] >= 0.9) & (p_in[:, src] >= 0.5)).nonzero().flatten()
        idx = good[cost[good].argsort()][:16].tolist()
        if len(idx) < 16:
            rest = [i for i in p_out[:, tgt].argsort(descending=True).tolist() if i not in idx]
            idx += rest[:16 - len(idx)]
        res["best_pairs"][direction] = {"rule": "p_target(out)>=0.9 and p_source(in)>=0.5, lowest cost; filled by "
                                                "highest p_target if fewer than 16", "n_qualifying": len(good),
                                        "idx": idx, "cost": cost[idx].tolist(), "p_target": p_out[idx, tgt].tolist()}
        bname = f"{tag}best_pairs_big.png" if direction == "cat_to_dog" else f"{tag}best_pairs_dog_to_cat_big.png"
        labeled_grid([x[idx], y[idx]], f"{F}/{bname}", col_labels=[f"c={cost[i]:.0f}\np={p_out[i, tgt]:.2f}" for i in idx],
                     title=f"best {sname}->{tname} pairs (top: input, bottom: output); c = |x-T(x)|^2, p = judge "
                           f"P({tname}); {len(good)}/{len(x)} qualify")
        downloads.append(f"{folder}/{bname}")
        _commit(vol)

    # steering-strength grid (cat -> dog), first n_steer cats; s*=1, w=0 is the only valid coupling
    K = keep["cat_to_dog"]
    xs = K["x"][:n_steer]
    cols, labels, stats = [], [], []
    for k in kq:
        if k == kq[-1]:
            yk = K["y"][:n_steer]
        else:
            yk = pf_ode(s_dog, sde, K["enc"][k][:n_steer].to(dev), s_of(k), eps, k).cpu()
        cols.append(yk)
        labels.append(f"s*={s_of(k):.1f}\nw=0")
    for w in ws:
        def sg(xx, ss, w=w):
            return (1 + w) * s_dog(xx, ss) - w * s_cat(xx, ss)
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
    foot = ["per column over the first %d test cats: mean cost (clamped) / judge dog-rate" % len(xs),
            "  ".join(f"[{s_['setting']}] {s_['cost_clamped_mean']:.0f} / {s_['judge_dog_rate']:.2f}" for s_ in stats[:5]),
            "  ".join(f"[{s_['setting']}] {s_['cost_clamped_mean']:.0f} / {s_['judge_dog_rate']:.2f}" for s_ in stats[5:]),
            "GREEN frame = s*=1, w=0: the only valid coupling. Partial encodings (s*<1) and guidance (w>0) are "
            "diagnostics, NOT couplings."]
    labeled_grid(rows, f"{F}/{tag}steering_grid_big.png", col_labels=["input cat"] + labels,
                 title="steering strength, cat -> dog (from-scratch models): partial encoding s*, then density-ratio "
                       "guidance w at s*=1", footer=foot, highlight=1 + kq.index(kq[-1]))
    sj = {"columns": stats, "n_images": len(xs), "guidance": "decode with (1+w) s_dog - w s_cat (tilted law, NOT nu)",
          "partial": "encode with cat model to s*, decode with dog model from s* (NOT a coupling for s*<1)"}
    with open(_fresh(f"{F}/{tag}steering_grid.json"), "w") as fh:
        json.dump(sj, fh, indent=1)
    res["steering"] = sj
    downloads += [f"{folder}/{tag}steering_grid_big.png", f"{folder}/{tag}steering_grid.json"]
    _commit(vol)

    # same noise -> both new models (and the fine-tuned iter02 models for reference)
    pc, pd = torch.load(pick["cat"]["prior_file"]), torch.load(pick["dog"]["prior_file"])
    assert torch.equal(pc["z"], pd["z"])
    same = {}

    def same_scene(a, b, name, a_cls=CAT, b_cls=DOG):
        m = min(len(a), len(b))
        a, b = a[:m], b[:m]
        pa, pb = ev.judge_probs(net, a, dev), ev.judge_probs(net, b, dev)
        same[name] = {"n": m, "matched_cost_mean": float(ev.sq_cost(a, b.clamp(-1, 1)).mean()),
                      "mismatched_cost_mean": float(ev.sq_cost(a, b.roll(1, 0).clamp(-1, 1)).mean()),
                      "frac_both_correct_class": float(((pa.argmax(1) == a_cls) & (pb.argmax(1) == b_cls)).float().mean())}
        same[name]["matched_over_mismatched"] = same[name]["matched_cost_mean"] / same[name]["mismatched_cost_mean"]

    same_scene(pc["samples"].clamp(-1, 1), pd["samples"], "scratch_cat_vs_scratch_dog")
    rows = [pc["samples"][:16], pd["samples"][:16]]
    rl = ["new cat", "new dog"]
    oc_p, od_p = f"{VOL}/iter02/prior_cat.pt", f"{VOL}/iter02/prior_dog.pt"
    if os.path.exists(oc_p) and os.path.exists(od_p) and not smoke:
        oc, od = torch.load(oc_p), torch.load(od_p)
        if torch.equal(oc["z"][:len(pc["z"])], pc["z"]):
            same_scene(oc["samples"].clamp(-1, 1), od["samples"], "finetuned_cat_vs_finetuned_dog")
            same_scene(pc["samples"].clamp(-1, 1), oc["samples"], "scratch_cat_vs_finetuned_cat", CAT, CAT)
            same_scene(pd["samples"].clamp(-1, 1), od["samples"], "scratch_dog_vs_finetuned_dog", DOG, DOG)
            rows += [oc["samples"][:16], od["samples"][:16]]
            rl += ["old cat", "old dog"]
        else:
            same["note"] = "iter02 prior noise differs; old-model rows omitted"
    labeled_grid(rows, f"{F}/{tag}same_noise_both_models_big.png", row_labels=rl,
                 title=f"same seed-0 noise decoded by the new cat model ({pick['cat']['name']}) and new dog model "
                       f"({pick['dog']['name']}); rows 3-4: fine-tuned iter-01 models (200-step decodes)")
    res["same_noise"] = same
    downloads.append(f"{folder}/{tag}same_noise_both_models_big.png")
    _commit(vol)

    # standard evaluation (judge, cost clamped/unclamped, KID vs floor) + chosen priors
    for c in ("cat", "dog"):
        runs.append({"name": f"prior_{pick[c]['name']}", "kind": "prior", "path": pick[c]["prior_file"], "cls": CLS[c]})
    eval_runs(runs, f"{tag}evalT", folder=folder, kid_refs=True, n_boot=10 if smoke else 200)
    small = torch.load(f"{F}/{tag}evalT_small.pt")
    for direction in DIRS:
        d = small[f"{direction}_scratch"]
        for nm, sl in (("", slice(0, 16)), ("_b", slice(16, 32))):
            fn = f"{tag}{direction}_scratch{nm}_big.png"
            if len(d["x"][sl]):
                ev.grid_png([d["x"][sl], d["y"][sl]], f"{F}/{fn}")
                downloads.append(f"{folder}/{fn}")
        fn = f"{tag}{direction}_scratch_lowest_ptarget_big.png"
        ev.grid_png([d["worst_x"], d["worst_y"]], f"{F}/{fn}")
        downloads.append(f"{folder}/{fn}")
    res["minutes"] = (time.time() - t0) / 60
    with open(_fresh(f"{F}/{tag}eval04.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    _commit(vol)
    downloads += [f"{folder}/{tag}eval04.json", f"{folder}/{tag}evalT_stats.json"]
    return {"task": "eval_composed", "minutes": res["minutes"], "downloads": downloads}
