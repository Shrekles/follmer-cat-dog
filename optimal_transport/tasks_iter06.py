"""GPU tasks for iteration 06 (iter_06_repick_full). No training.

ft_full():  fine-tuned (FT) composed PF-ODE (out/cat.pth, out/dog.pth; Heun 200, eps 1e-3, no Tweedie) on test images
            512-999 of each class, combined with the iteration-01 outputs for 0-511 -> numbers over all 1,000.
repick():   encoder-aware re-pick of the from-scratch snapshot pairs (iterations 04/05) on a 64-image composed-transport
            probe (stage 1: every encoder with a fixed decoder; stage 2: top-3 encoders x every decoder), then the
            selected pair per direction on the first 512 test images with the standard eval and figures (stage 3).

Run inside Modal via modal_iter06.py. All outputs go to /vol/iter06/ (smoke: /vol/iter06/smoke/); everything else on
the volume is only read. Nothing existing is ever overwritten (_fresh).
"""
import json
import os
import time

import torch

import evaluate as ev
from tasks_iter import CAT, CONFIG, DIRS, DOG, REPO, VOL, _fresh, cifar
from tasks_iter02 import _Incep, eval_runs, kid
from tasks_iter03 import memo_check
from tasks_iter04 import JUDGE, _commit, _score4, labeled_grid
from tasks_iter05 import FT, FT_TRANSPORT, IT04_TRANSPORT, _best_idx, gauss_stats, sat_frac

NAME = {CAT: "cat", DOG: "dog"}
CANDS = {
    "cat": ["iter04/cat_s12129.pth", "iter04/cat_s14714.pth", "iter04/cat_s17297.pth", "iter04/cat_s19874.pth",
            "iter04/cat_s22457.pth"] + [f"iter05/cat_{e}_s{s}.pth" for s in (28347, 31247, 34123)
                                        for e in ("e9995", "e9999")],
    "dog": ["iter04/dog_s12386.pth", "iter04/dog_s14961.pth", "iter04/dog_s17541.pth", "iter04/dog_s20119.pth",
            "iter04/dog_s22697.pth"] + [f"iter05/dog_{e}_s{s}.pth" for s in (28586, 31505, 34420)
                                        for e in ("e9995", "e9999")],
}
SMOKE_CANDS = {"cat": ["iter04/cat_s17297.pth", "iter04/cat_s22457.pth", "iter05/cat_e9995_s31247.pth"],
               "dog": ["iter04/dog_s20119.pth", "iter04/dog_s22697.pth", "iter05/dog_e9999_s28586.pth"]}
REF_ENC = {"cat_to_dog": "iter04/cat_s17297.pth", "dog_to_cat": "iter04/dog_s20119.pth"}
FIXED_DEC = {"cat_to_dog": "iter04/dog_s20119.pth", "dog_to_cat": "iter04/cat_s17297.pth"}


def _short(p):
    return p.replace("iter0", "").replace(".pth", "").replace("/", "_")   # iter04/cat_s17297.pth -> 4_cat_s17297


def _ode(score, sde, x, s0, s1, steps, dev, bs=256):
    from pf_ode import pf_ode
    return torch.cat([pf_ode(score, sde, x[i:i + bs].to(dev), s0, s1, steps).cpu() for i in range(0, len(x), bs)])


def pair_stats(net, inc, freal_tgt, x, y, src, tgt, n_boot=200, with_kid=True):
    c, cc = ev.sq_cost(x, y), ev.sq_cost(x, y.clamp(-1, 1))
    p = ev.judge_probs(net, y, torch.device("cuda"))
    out = {"n": len(x), "cost": {**ev.mean_se(c), "median": float(c.median())},
           "cost_clamped": {**ev.mean_se(cc), "median": float(cc.median())},
           "judge_target": ev.rate_stats(p, tgt), "judge_source": ev.rate_stats(p, src),
           "class_histogram": torch.bincount(p.argmax(1), minlength=10).tolist(),
           "frac_samples_saturated": sat_frac(y),
           "n_over_10pct_clipped": int((((y < -1) | (y > 1)).flatten(1).float().mean(1) > 0.1).sum())}
    if with_kid:
        out["kid_vs_all_test_target"] = kid(inc.feats(y), freal_tgt, n_boot)
    return out, {"cost": c, "cost_clamped": cc, "p": p}


def _fmt(s):
    k = s.get("kid_vs_all_test_target")
    return (f"cost {s['cost']['mean']:.1f} (clamped {s['cost_clamped']['mean']:.1f} ± {s['cost_clamped']['stderr']:.1f},"
            f" med {s['cost_clamped']['median']:.1f}) judge {s['judge_target']['rate']:.3f} src "
            f"{s['judge_source']['rate']:.3f}" + (f" KID {k['kid_x1e3']:.2f} ± {k['se_x1e3']:.2f}" if k else ""))


# ------------------------------------------------------------------------------------------ container A
def ft_full(vol=None, smoke=False, smoke_dir="smoke", lo=512, bs=256, eps=1e-3, n_det=16):
    """FT composed ODE on test indices lo..999 per direction; combined with iteration-01 outputs for 0..511."""
    from score_sde_backend import load_score_model
    t0, dev = time.time(), torch.device("cuda")
    folder = f"iter06/{smoke_dir}" if smoke else "iter06"
    FD = f"{VOL}/{folder}"
    steps, n_new, n_boot = (5, 8, 10) if smoke else (200, None, 200)
    _, sde, _, sc = load_score_model(REPO, CONFIG, f"{VOL}/{FT['cat']}", dev)
    _, _, _, sd = load_score_model(REPO, CONFIG, f"{VOL}/{FT['dog']}", dev)
    S = {CAT: sc, DOG: sd}
    net = ev.load_judge(f"{VOL}/{JUDGE}", dev)
    inc = _Incep(dev)
    real = {c: cifar("test", c)[0] for c in (CAT, DOG)}
    freal = {c: inc.feats(real[c]) for c in (CAT, DOG)}
    res = {"settings": {"cat_ckpt": FT["cat"], "dog_ckpt": FT["dog"], "steps": steps, "eps": eps, "solver": "Heun",
                        "tweedie": False, "batch": bs, "new_indices": f"{lo}-999", "old_file": FT_TRANSPORT},
           "ceilings_all_test": {}, "kid_floor": {}, "directions": {}}
    for c in (CAT, DOG):
        res["ceilings_all_test"][NAME[c]] = ev.rate_stats(ev.judge_probs(net, real[c], dev), c)
        res["kid_floor"][f"train1000_{NAME[c]}_vs_test1000"] = kid(inc.feats(cifar("train", c, 1000)[0]), freal[c], n_boot)
        res["kid_floor"][f"train512_{NAME[c]}_vs_test1000"] = kid(inc.feats(cifar("train", c, 512)[0]), freal[c], n_boot)
    downloads, per_image = [], {}
    for direction, (src, tgt) in DIRS.items():
        xall = real[src]
        old = torch.load(f"{VOL}/{FT_TRANSPORT[direction]}")
        x0, y0 = old["x"], old["outputs"]["s1.0"]
        assert len(x0) == 512 and torch.equal(x0, xall[:512]), "iteration-01 inputs are not test images 0-511"
        # determinism check: re-transport the first n_det images with the same code
        zc = _ode(S[src], sde, x0[:n_det], eps, sde.T, steps, dev, bs)
        yc = _ode(S[tgt], sde, zc, sde.T, eps, steps, dev, bs)
        det = {"n": n_det, "max_abs_diff": float((yc - y0[:n_det]).abs().max()),
               "mean_abs_diff": float((yc - y0[:n_det]).abs().mean()),
               "cost_old": float(ev.sq_cost(x0[:n_det], y0[:n_det]).mean()),
               "cost_new": float(ev.sq_cost(x0[:n_det], yc).mean()), "steps": steps}
        print(f"{direction} determinism: {det}", flush=True)
        xn = xall[lo:] if n_new is None else xall[lo:lo + n_new]
        zn, yn = [], []
        for i in range(0, len(xn), bs):
            zb = _ode(S[src], sde, xn[i:i + bs], eps, sde.T, steps, dev, bs)
            yn.append(_ode(S[tgt], sde, zb, sde.T, eps, steps, dev, bs))
            zn.append(zb)
            print(f"{direction} batch {i // bs + 1}: {(time.time() - t0) / 60:.1f} min", flush=True)
        zn, yn = torch.cat(zn), torch.cat(yn)
        path = _fresh(f"{FD}/transport_{direction}_FT_{lo}_999.pt")
        torch.save({"x": xn, "z": zn, "outputs": {"s1.0": yn}, "meta": {
            "s1.0": {"k": steps, "s": 1.0, "w": 0.0, "valid_coupling": True,
                     "note": f"FT composed PF-ODE on test {NAME[src]}s {lo}..{lo + len(xn) - 1}"},
            "_run": {**res["settings"], "direction": direction, "n": len(xn), "minutes": (time.time() - t0) / 60}}},
            path)
        _commit(vol)
        X, Y = torch.cat([x0, xn]), torch.cat([y0, yn])
        d = {"determinism_first16": det, "new_file": path, "n_total": len(X)}
        for sub, sl in (("all", slice(None)), ("0_511", slice(0, 512)), (f"{lo}_999", slice(512, None))):
            st, per = pair_stats(net, inc, freal[tgt], X[sl], Y[sl], src, tgt, n_boot)
            d[sub] = st
            print(f"FT {direction} [{sub}] n={len(X[sl])}: {_fmt(st)}", flush=True)
            if sub == "all":
                P = per
        res["directions"][direction] = d
        per_image[direction] = {"cost": P["cost"].tolist(), "cost_clamped": P["cost_clamped"].tolist(),
                                "p_target": P["p"][:, tgt].tolist(), "p_source": P["p"][:, src].tolist(),
                                "pred": P["p"].argmax(1).tolist()}
        # figures
        sn, tn = NAME[src], NAME[tgt]
        for tag, a in (("", 0), ("_b", 16)):
            if len(xn) <= a:
                continue
            f = f"ft_{direction}_{lo}{tag}_big.png"
            labeled_grid([xn[a:a + 16], yn[a:a + 16]], f"{FD}/{f}", row_labels=["input", "FT T(x)"],
                         col_labels=[f"#{lo + a + i}" for i in range(min(16, len(xn) - a))],
                         title=f"FT composed ODE {sn}->{tn}, test images {lo + a}-{lo + a + 15} (top input, bottom T(x))")
            downloads.append(f"{folder}/{f}")
        pt = P["p"][:, tgt]
        w = pt.argsort()[:16].tolist()
        f = f"ft_{direction}_1000_lowest_ptarget_big.png"
        labeled_grid([X[w], Y[w]], f"{FD}/{f}", col_labels=[f"#{i}\np={pt[i]:.2f}" for i in w],
                     title=f"FT {sn}->{tn}: the 16 lowest P({tn}) outputs over all {len(X)} test {sn}s (top input)")
        downloads.append(f"{folder}/{f}")
        idx, n_good, n_fill = _best_idx(P["p"], P["cost_clamped"], src, tgt)
        d["best_pairs"] = {"n_qualifying": n_good, "n_filled": n_fill, "idx": idx,
                           "cost_clamped": P["cost_clamped"][idx].tolist(), "p_target": pt[idx].tolist()}
        f = f"ft_best_pairs_{direction}_1000_big.png"
        labeled_grid([X[idx], Y[idx]], f"{FD}/{f}",
                     col_labels=[f"#{i} c={P['cost_clamped'][i]:.0f}\np={pt[i]:.2f}" for i in idx],
                     title=f"FT best {sn}->{tn} pairs over all {len(X)} (strict P({tn})>=0.9, P({sn})<=0.05, "
                           f"clamped cost ascending); {n_good}/{len(X)} qualify")
        downloads.append(f"{folder}/{f}")
    res["minutes"] = (time.time() - t0) / 60
    for name, obj in (("ft1000.json", res), ("ft1000_per_image.json", per_image)):
        with open(_fresh(f"{FD}/{name}"), "w") as fh:
            json.dump(obj, fh, indent=1 if name == "ft1000.json" else None)
        downloads.append(f"{folder}/{name}")
    _commit(vol)
    return {"task": "ft_full", "minutes": res["minutes"], "downloads": downloads}


# ------------------------------------------------------------------------------------------ container B
def _rank_encoders(st1, ref_key, encs, dec):
    cap = st1[ref_key]["cost_clamped"]["mean"]
    cc = lambda e: st1[(e, dec)]["cost_clamped"]["mean"]  # noqa: E731
    feas = sorted([e for e in encs if cc(e) <= cap], key=lambda e: (-st1[(e, dec)]["judge_target"]["rate"], cc(e)))
    rest = sorted([e for e in encs if cc(e) > cap], key=cc)
    return feas + rest, cap


def _select(stats, cap):
    feas = [k for k, s in stats.items() if s["cost_clamped"]["mean"] <= cap]
    return min(feas, key=lambda k: (-stats[k]["judge_target"]["rate"], stats[k]["cost_clamped"]["mean"]))


def repick(vol=None, smoke=False, smoke_dir="smoke", eps=1e-3, n_probe=64, n3=512, n_cmp=32, bs=256):
    t0, dev = time.time(), torch.device("cuda")
    folder = f"iter06/{smoke_dir}" if smoke else "iter06"
    FD = f"{VOL}/{folder}"
    steps, n_boot, cands = 200, 200, CANDS
    if smoke:
        steps, n_boot, cands, n_probe, n3, n_cmp = 5, 10, SMOKE_CANDS, 8, 16, 16
    net = ev.load_judge(f"{VOL}/{JUDGE}", dev)
    inc = _Incep(dev)
    real = {c: cifar("test", c)[0] for c in (CAT, DOG)}
    freal = {c: inc.feats(real[c]) for c in (CAT, DOG)}
    res = {"settings": {"steps": steps, "eps": eps, "n_probe": n_probe, "n_stage3": n3, "candidates": cands,
                        "fixed_decoder": FIXED_DEC, "reference_pair": {d: [REF_ENC[d], FIXED_DEC[d]] for d in DIRS},
                        "rule": "max judge target-rate on the probe s.t. clamped cost <= reference (iter04) pair's "
                                "clamped cost on the same images; tie -> lower clamped cost"},
           "directions": {}, "stage_minutes": {}}
    downloads, selected = [], {}
    for direction, (src, tgt) in DIRS.items():
        sname, tname = NAME[src], NAME[tgt]
        x = real[src][:n_probe]
        lat, gz = {}, {}
        for e in cands[sname]:
            sde, s = _score4(e, dev)
            lat[e] = _ode(s, sde, x, eps, sde.T, steps, dev, bs)
            gz[e] = gauss_stats(lat[e], n_boot=n_boot)
            del s
        outs, st = {}, {}

        def decode_many(dec, encs):
            sde_, s_ = _score4(dec, dev)
            Y = _ode(s_, sde_, torch.cat([lat[e] for e in encs]), sde_.T, eps, steps, dev, bs)
            for j, e in enumerate(encs):
                y = Y[j * len(x):(j + 1) * len(x)]
                outs[(e, dec)] = y
                st[(e, dec)] = pair_stats(net, inc, freal[tgt], x, y, src, tgt, n_boot)[0]
                print(f"[{direction}] enc {_short(e)} dec {_short(dec)}: {_fmt(st[(e, dec)])} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)

        decode_many(FIXED_DEC[direction], cands[sname])
        ref_key = (REF_ENC[direction], FIXED_DEC[direction])
        order, cap = _rank_encoders(st, ref_key, cands[sname], FIXED_DEC[direction])
        top3 = order[:3]
        print(f"[{direction}] cost cap {cap:.2f}; encoder ranking {[_short(e) for e in order]}", flush=True)
        stage1 = {e: st[(e, FIXED_DEC[direction])] for e in cands[sname]}
        for dec in cands[tname]:
            if dec != FIXED_DEC[direction]:
                decode_many(dec, top3)
        sel = _select(st, cap)
        selected[direction] = sel
        print(f"[{direction}] SELECTED enc {_short(sel[0])} dec {_short(sel[1])}: {_fmt(st[sel])}", flush=True)
        res["directions"][direction] = {
            "cost_cap_ref_clamped": cap, "reference": list(ref_key), "reference_stats": st[ref_key],
            "stage1": {_short(e): {**v, "encoder": e, "decoder": FIXED_DEC[direction]} for e, v in stage1.items()},
            "encoder_gaussianity": {_short(e): v for e, v in gz.items()},
            "encoder_ranking": [_short(e) for e in order], "top3": top3,
            "stage2": {f"{_short(e)}->{_short(d)}": {**v, "encoder": e, "decoder": d} for (e, d), v in st.items()
                       if e in top3},
            "selected": {"encoder": sel[0], "decoder": sel[1], "is_reference": sel == ref_key, "probe_stats": st[sel]}}
        torch.save({"x": x, "latents": {_short(e): z for e, z in lat.items()},
                    "outputs": {f"{_short(e)}->{_short(d)}": y for (e, d), y in outs.items()}},
                   _fresh(f"{FD}/probe_{direction}.pt"))
        # stage-1 probe figure: first 16 probe images, each encoder with the fixed decoder
        rows = [x[:16]] + [outs[(e, FIXED_DEC[direction])][:16] for e in order]
        f = f"probe_stage1_{direction}_big.png"
        labeled_grid(rows, f"{FD}/{f}", row_labels=["input"] + [_short(e) for e in order],
                     title=f"stage 1 {sname}->{tname}: every {sname} encoder (rows, ranked) with decoder "
                           f"{_short(FIXED_DEC[direction])}; probe images 0-15")
        downloads.append(f"{folder}/{f}")
        _commit(vol)
    res["stage_minutes"]["stages_1_2"] = (time.time() - t0) / 60

    # stage 3: selected pairs on the first n3 test images
    runs, keep = [], {}
    for direction, (src, tgt) in DIRS.items():
        enc, dec = selected[direction]
        x = real[src][:n3]
        sde, s = _score4(enc, dev)
        z = _ode(s, sde, x, eps, sde.T, steps, dev, bs)
        sde, s = _score4(dec, dev)
        y = _ode(s, sde, z, sde.T, eps, steps, dev, bs)
        path = _fresh(f"{FD}/transport_{direction}_repick.pt")
        torch.save({"x": x, "z": z, "outputs": {"s1.0": y}, "meta": {
            "s1.0": {"k": steps, "s": 1.0, "w": 0.0, "valid_coupling": True,
                     "note": f"composed PF-ODE, re-picked from-scratch pair: encoder {enc}, decoder {dec}"},
            "_run": {"direction": direction, "n": len(x), "steps": steps, "eps": eps, "encoder": enc, "decoder": dec,
                     "minutes": (time.time() - t0) / 60}}}, path)
        keep[direction] = {"x": x, "repick": y}
        runs.append({"name": f"{direction}_repick", "kind": "transport", "direction": direction, "path": path})
        for nm, files in (("FT", FT_TRANSPORT), ("iter04", IT04_TRANSPORT)):
            d = torch.load(f"{VOL}/{files[direction]}")
            assert torch.equal(d["x"][:len(x)], x)
            keep[direction][nm] = d["outputs"]["s1.0"][:len(x)]
            runs.append({"name": f"{direction}_{nm}", "kind": "transport", "direction": direction,
                         "path": f"{VOL}/{files[direction]}"})
        print(f"stage 3 {direction}: cost {ev.sq_cost(x, y).mean():.1f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        _commit(vol)
    eval_runs(runs, "evalT", folder=folder, kid_refs=True, n_boot=n_boot)
    small = torch.load(f"{FD}/evalT_small.pt")
    # held-out probe complement (images n_probe..n3-1) and memorisation on T(x)
    res["heldout"], memo_sets = {}, {}
    for direction, (src, tgt) in DIRS.items():
        K = keep[direction]
        res["heldout"][direction] = {}
        for nm in ("repick", "FT", "iter04"):
            sl = slice(n_probe, n3)
            res["heldout"][direction][nm] = pair_stats(net, inc, freal[tgt], K["x"][sl], K[nm][sl], src, tgt, n_boot)[0]
            print(f"heldout {direction} {nm}: {_fmt(res['heldout'][direction][nm])}", flush=True)
            p = _fresh(f"{FD}/samples_{direction}_{nm}.pt")
            torch.save({"samples": K[nm]}, p)
            memo_sets.setdefault(NAME[tgt], {})[f"T_{direction}_{nm}"] = p
    res["memo"] = {c: memo_check(c, sets, f"memo_T_{c}", folder=folder)["result"] for c, sets in memo_sets.items()}
    # figures
    for direction, (src, tgt) in DIRS.items():
        K, sn, tn = keep[direction], NAME[src], NAME[tgt]
        enc, dec = selected[direction]
        lab = f"repick {_short(enc)}->{_short(dec)}"
        for tag, a in (("", 0), ("_b", 16)):
            if len(K["x"]) <= a:
                continue
            f = f"{direction}_repick{tag}_big.png"
            ev.grid_png([K["x"][a:a + 16], K["repick"][a:a + 16]], f"{FD}/{f}")
            downloads.append(f"{folder}/{f}")
        dd = small[f"{direction}_repick"]
        f = f"{direction}_repick_lowest_ptarget_big.png"
        labeled_grid([dd["worst_x"], dd["worst_y"]], f"{FD}/{f}",
                     col_labels=[f"#{i}\np={q:.2f}" for i, q in zip(dd["worst_idx"].tolist(), dd["worst_p_target"].tolist())],
                     title=f"{lab}: 16 lowest P({tn}) outputs of {len(K['x'])} (top input)")
        downloads.append(f"{folder}/{f}")
        y = K["repick"]
        p_out = ev.judge_probs(net, y, dev)
        cost = ev.sq_cost(K["x"], y.clamp(-1, 1))
        idx, n_good, n_fill = _best_idx(p_out, cost, src, tgt)
        res.setdefault("best_pairs", {})[direction] = {"n_qualifying": n_good, "n_filled": n_fill, "idx": idx,
                                                       "cost_clamped": cost[idx].tolist(),
                                                       "p_target": p_out[idx, tgt].tolist(),
                                                       "p_source": p_out[idx, src].tolist()}
        f = f"best_pairs_{direction}_repick_big.png"
        labeled_grid([K["x"][idx], y[idx]], f"{FD}/{f}",
                     col_labels=[f"#{i} c={cost[i]:.0f}\np={p_out[i, tgt]:.2f}" for i in idx],
                     title=f"best {sn}->{tn} pairs, {lab} (top input); strict P({tn})>=0.9 and P({sn})<=0.05, "
                           f"clamped cost ascending; {n_good}/{len(y)} qualify" + (f"; last {n_fill} fill" if n_fill else ""))
        downloads.append(f"{folder}/{f}")
        m = min(n_cmp, len(K["x"]))
        f = f"compare_{direction}_big.png"
        labeled_grid([K["x"][:m], K["FT"][:m], K["iter04"][:m], K["repick"][:m]], f"{FD}/{f}",
                     row_labels=["input", "FT (iter01)", "iter04 pick", "re-pick"],
                     col_labels=[f"#{i}" for i in range(m)],
                     title=f"{direction}, test images 0-{m - 1}: rows input / fine-tuned pair / iter04 pick "
                           f"(4_cat_s17297, 4_dog_s20119) / {lab}")
        downloads.append(f"{folder}/{f}")
    res["minutes"] = (time.time() - t0) / 60
    with open(_fresh(f"{FD}/repick06.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    _commit(vol)
    downloads += [f"{folder}/repick06.json", f"{folder}/evalT_stats.json"] + \
                 [f"{folder}/memo_T_{c}.json" for c in memo_sets]
    return {"task": "repick", "selected": {d: [_short(e) for e in v] for d, v in selected.items()},
            "minutes": res["minutes"], "downloads": downloads}


def chain_smoke(vol=None, smoke_dir="smoke"):
    t0 = time.time()
    r1 = ft_full(vol=vol, smoke=True, smoke_dir=smoke_dir)
    r2 = repick(vol=vol, smoke=True, smoke_dir=smoke_dir)
    return {"task": "chain_smoke", "sub": [r1, r2], "downloads": r1["downloads"] + r2["downloads"],
            "minutes": (time.time() - t0) / 60}
