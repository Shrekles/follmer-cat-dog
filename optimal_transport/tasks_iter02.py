"""GPU tasks for iteration 02 (purity of the target marginal). Run inside Modal via modal_iter02.py.

All outputs go to /vol/<folder>/ (default iter02); nothing existing is ever overwritten.

prior_samples(): decode the SAME 512 noise vectors (seed 0) with several models -> prior_<name>.pt
encode_latents(): encode the first n test images of the source class with the source model -> latents_<dir>.pt
guided_decode(): decode saved latents with s_g = (1+w) s_tgt - w s_src   -> transport_<dir>_w<w>.pt
    (w = 0 is the plain composed ODE.)  w > 0 samples a TILTED law ~ p_tgt^(1+w) / p_src^w, i.e. NOT nu,
    so w > 0 runs are diagnostics, never valid couplings of mu and nu.
build_cinic(): download CINIC-10 into /vol/data/cinic10/ and build uint8 tensors of ImageNet-derived dogs/cats.
finetune_dog_cinic(): fine-tune out/dog.pth on 50% CIFAR-train dogs + 50% CINIC-ImageNet dogs.
eval_runs(): judge + cost (unclamped and clamped) + KID (Inception-v3 pool features) for a list of runs.
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
from tasks_iter import CAT, CONFIG, DIRS, DOG, REPO, VOL, _fresh, cifar

CINIC_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"


def _left(deadline):
    return (deadline - time.time()) / 60 if deadline else 1e9


def _score(ckpt, dev):
    from score_sde_backend import load_score_model
    _, sde, _, s = load_score_model(REPO, CONFIG, f"{VOL}/{ckpt}", dev)
    return sde, s


def prior_z(n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, 32, 32, generator=g)


# ------------------------------------------------------------------------------- prior samples
def prior_samples(models, n=512, steps=200, bs=256, eps=1e-3, folder="iter02", tag="", deadline=None,
                  est_min=5.0):
    """models: {name: ckpt}. Decodes the same z (seed 0) with each model (Heun, steps)."""
    from pf_ode import pf_ode
    t0, dev, done = time.time(), torch.device("cuda"), []
    z = prior_z(n)
    for name, ckpt in models.items():
        if _left(deadline) < est_min * n / 512:
            print(f"SKIP prior {name}: deadline", flush=True)
            continue
        out_path = _fresh(f"{VOL}/{folder}/{tag}prior_{name}.pt")
        sde, s = _score(ckpt, dev)
        ys = [pf_ode(s, sde, z[i:i + bs].to(dev), sde.T, eps, steps).cpu() for i in range(0, n, bs)]
        torch.save({"z": z, "samples": torch.cat(ys),
                    "meta": {"ckpt": ckpt, "seed": 0, "steps": steps, "eps": eps, "n": n}}, out_path)
        done.append(name)
        print(f"prior {name}: {(time.time() - t0) / 60:.1f} min", flush=True)
    return {"task": f"prior_samples {done}", "minutes": (time.time() - t0) / 60}


# ------------------------------------------------------------------------------------- latents
def encode_latents(direction, n=512, steps=200, bs=256, eps=1e-3, cat_ckpt="out/cat.pth", dog_ckpt="out/dog.pth",
                   folder="iter02", tag=""):
    from pf_ode import pf_ode
    t0, dev = time.time(), torch.device("cuda")
    out_path = _fresh(f"{VOL}/{folder}/{tag}latents_{direction}.pt")
    src, _ = DIRS[direction]
    sde, s = _score(cat_ckpt if src == CAT else dog_ckpt, dev)
    x, _ = cifar("test", src, n)
    zs = [pf_ode(s, sde, x[i:i + bs].to(dev), eps, sde.T, steps).cpu() for i in range(0, len(x), bs)]
    torch.save({"x": x, "z": torch.cat(zs), "meta": {"direction": direction, "steps": steps, "eps": eps,
                                                     "encoder": cat_ckpt if src == CAT else dog_ckpt}}, out_path)
    print(f"latents {direction}: {(time.time() - t0) / 60:.1f} min", flush=True)
    return {"task": f"encode_latents {direction}", "minutes": (time.time() - t0) / 60}


def _wait_for(path, max_wait_min, vol=None):
    t0 = time.time()
    while not os.path.exists(path):
        if (time.time() - t0) / 60 > max_wait_min:
            return False
        time.sleep(20)
        if vol is not None:
            vol.reload()
    return True


# ------------------------------------------------------------------------------- guided decode
def guided_decode(direction, w, steps=200, bs=256, eps=1e-3, cat_ckpt="out/cat.pth", dog_ckpt="out/dog.pth",
                  tgt_ckpt=None, name=None, folder="iter02", tag="", deadline=None, est_min=None, vol=None,
                  wait_min=0.0):
    """Decode saved latents with s_g = (1+w) s_tgt - w s_src. tgt_ckpt overrides the target model."""
    from pf_ode import pf_ode
    t0, dev = time.time(), torch.device("cuda")
    lat = f"{VOL}/{folder}/{tag}latents_{direction}.pt"
    if wait_min and not _wait_for(lat, wait_min, vol):
        print(f"SKIP {direction} w{w}: latents never appeared", flush=True)
        return {"task": f"guided_decode {direction} w{w} SKIPPED (no latents)", "minutes": (time.time() - t0) / 60}
    d = torch.load(lat)
    n = len(d["x"])
    est = est_min if est_min is not None else (5.0 if w == 0 else 10.0) * n / 512
    if _left(deadline) < est:
        print(f"SKIP {direction} w{w}: deadline ({_left(deadline):.1f} min left)", flush=True)
        return {"task": f"guided_decode {direction} w{w} SKIPPED (deadline)", "minutes": (time.time() - t0) / 60}
    name = name or f"w{w}"
    out_path = _fresh(f"{VOL}/{folder}/{tag}transport_{direction}_{name}.pt")
    src, tgt = DIRS[direction]
    src_ckpt = cat_ckpt if src == CAT else dog_ckpt
    tgt_ckpt = tgt_ckpt or (dog_ckpt if tgt == DOG else cat_ckpt)
    sde, s_tgt = _score(tgt_ckpt, dev)
    if w != 0:
        _, s_src = _score(src_ckpt, dev)

        def score(x, s):
            return (1 + w) * s_tgt(x, s) - w * s_src(x, s)
    else:
        score = s_tgt
    ys = [pf_ode(score, sde, d["z"][i:i + bs].to(dev), sde.T, eps, steps).cpu() for i in range(0, n, bs)]
    y = torch.cat(ys)
    valid = (w == 0)
    note = ("composed PF-ODE (encode source model, decode target model): valid coupling up to model/ODE error"
            if valid else
            f"density-ratio guided decode, score (1+w)s_tgt - w s_src with w={w}: samples a tilted law "
            "~ p_tgt^(1+w)/p_src^w, NOT the target marginal; diagnostic only, NOT a coupling of mu and nu")
    meta = {"s1.0": {"k": steps, "s": 1.0, "w": w, "valid_coupling": valid, "note": note},
            "_run": {"direction": direction, "n": n, "steps": steps, "eps": eps, "w": w, "encoder": d["meta"]["encoder"],
                     "src_ckpt": src_ckpt, "tgt_ckpt": tgt_ckpt, "minutes": (time.time() - t0) / 60}}
    torch.save({"x": d["x"], "outputs": {"s1.0": y}, "meta": meta}, out_path)
    print(f"{direction} {name}: cost {ev.sq_cost(d['x'], y).mean():.1f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return {"task": f"guided_decode {direction} {name}", "minutes": (time.time() - t0) / 60}


# ------------------------------------------------------------------------------------- CINIC-10
class _Tee:
    """File-like wrapper: reading from `src` also writes the bytes to `dst` (save tarball while parsing)."""

    def __init__(self, src, dst, t0, cap_min):
        self.src, self.dst, self.t0, self.cap, self.n = src, dst, t0, cap_min, 0

    def read(self, k=-1):
        if (time.time() - self.t0) / 60 > self.cap:
            raise TimeoutError(f"CINIC download exceeded {self.cap} min cap after {self.n / 1e6:.0f} MB")
        b = self.src.read(k)
        self.dst.write(b)
        self.n += len(b)
        return b


def build_cinic(cap_min=12.0, smoke=False):
    import io
    import tarfile
    import urllib.request
    import numpy as np
    from PIL import Image
    t0 = time.time()
    if smoke:
        req = urllib.request.Request(CINIC_URL, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            info = {"status": r.status, "content_length": r.headers.get("Content-Length"), "url": r.geturl()}
        print("CINIC HEAD", info, flush=True)
        return {"task": "build_cinic smoke", "ok": True, "info": info, "minutes": (time.time() - t0) / 60}
    out_dir = f"{VOL}/data/cinic10"
    os.makedirs(out_dir, exist_ok=True)
    dog_path, cat_path = f"{VOL}/data/cinic10_dog_imagenet.pt", f"{VOL}/data/cinic10_cat_imagenet.pt"
    if os.path.exists(dog_path):
        print("CINIC dog tensor already on the volume; reusing", flush=True)
        return {"task": "build_cinic (reused)", "ok": True, "minutes": 0.0}
    tar_path = _fresh(f"{out_dir}/CINIC-10.tar.gz")
    imgs = {"dog": [], "cat": []}
    counts = {"skipped_cifar_dog": 0, "skipped_cifar_cat": 0, "members": 0}
    try:
        with urllib.request.urlopen(CINIC_URL, timeout=60) as r, open(tar_path + ".part", "wb") as fh:
            tee = _Tee(r, fh, t0, cap_min)
            with tarfile.open(fileobj=tee, mode="r|gz") as tf:
                for m in tf:
                    counts["members"] += 1
                    parts = m.name.split("/")
                    if not m.isfile() or len(parts) < 3 or parts[-2] not in imgs:
                        continue
                    cls, base = parts[-2], parts[-1]
                    if base.startswith("cifar10-"):
                        counts[f"skipped_cifar_{cls}"] += 1
                        continue
                    im = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
                    if im.size != (32, 32):
                        im = im.resize((32, 32), Image.BICUBIC)
                    imgs[cls].append(np.asarray(im).transpose(2, 0, 1).copy())
            while tee.read(1 << 20):  # drain the rest of the stream so the saved tarball is complete
                pass
        os.replace(tar_path + ".part", tar_path)
    except Exception as e:  # noqa: BLE001
        print(f"CINIC FAILED: {e!r}", flush=True)
        return {"task": "build_cinic", "ok": False, "error": repr(e), "minutes": (time.time() - t0) / 60}
    for cls, path in (("dog", dog_path), ("cat", cat_path)):
        x = torch.from_numpy(np.stack(imgs[cls]))
        y = torch.full((len(x),), DOG if cls == "dog" else CAT, dtype=torch.long)
        torch.save({"x": x, "y": y, "source": CINIC_URL, "filter": "ImageNet-derived only (no cifar10-* files)"},
                   _fresh(path))
        counts[f"imagenet_{cls}"] = len(x)
    print("CINIC counts", counts, flush=True)
    return {"task": "build_cinic", "ok": True, "counts": counts, "minutes": (time.time() - t0) / 60}


# -------------------------------------------------------------------------------- fine-tune dog
def finetune_dog_cinic(init="out/dog.pth", out="iter02/dog_cinic.pth", mid="iter02/dog_cinic_mid.pth",
                       bs=128, lr=1e-4, warmup=200, ema_rate=0.999, cap_min=32.0, max_steps=10 ** 9,
                       n_sub=None, deadline=None, reserve_min=0.0, seed=0, cinic_path=None):
    from score_sde_backend import load_model
    t0, dev = time.time(), torch.device("cuda")
    cap = min(cap_min, max(1.0, _left(deadline) - reserve_min))
    out_path, mid_path = _fresh(f"{VOL}/{out}"), _fresh(f"{VOL}/{mid}")
    torch.manual_seed(seed)
    config, sde, eps, model = load_model(REPO, CONFIG, f"{VOL}/{init}", dev)
    model.train()
    xa, _ = cifar("train", DOG)                       # [-1,1] floats, CIFAR-10 TRAIN dogs only
    xb = torch.load(f"{VOL}/{cinic_path or 'data/cinic10_dog_imagenet.pt'}")["x"]
    if n_sub:
        xa, xb = xa[:n_sub], xb[:n_sub]
    xa = xa.to(dev)
    xb = (xb.float() / 255 * 2 - 1).to(dev)
    print(f"finetune: cifar-train dogs {len(xa)}, cinic-imagenet dogs {len(xb)}, cap {cap:.1f} min", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = [p.detach().clone() for p in model.parameters()]
    accum = 1

    def save(path, step):
        backup = [p.detach().clone() for p in model.parameters()]
        for p_, e in zip(model.parameters(), ema):
            p_.data.copy_(e)
        torch.save({"model": model.state_dict(), "ema": {"shadow_params": []}, "step": step,
                    "meta": {"init": init, "bs": bs, "lr": lr, "ema": ema_rate, "data": "50% CIFAR-train dogs + 50% CINIC ImageNet dogs"}},
                   path)
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

    step, running, saved_mid, log = 0, 0.0, False, []
    while step < max_steps and (time.time() - t0) / 60 < cap:
        half = bs // 2
        x = torch.cat([xa[torch.randint(len(xa), (half,), device=dev)], xb[torch.randint(len(xb), (bs - half,), device=dev)]])
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warmup)
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
        if step % 100 == 0:
            el = (time.time() - t0) / 60
            log.append((step, running / 100, el))
            print(f"step {step} loss {running / 100:.4f} {el:.1f} min ({step / max(el, 1e-6) / 60:.2f} it/s)", flush=True)
            running = 0.0
        if not saved_mid and (time.time() - t0) / 60 > cap / 2:
            save(mid_path, step)
            saved_mid = True
            print(f"saved mid snapshot at step {step}", flush=True)
    if not saved_mid:
        save(mid_path, step)
    save(out_path, step)
    print(f"saved {out_path} at step {step} (accum {accum}, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)",
          flush=True)
    return {"task": "finetune_dog_cinic", "steps": step, "images_seen": step * bs, "accum": accum, "loss_log": log,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "minutes": (time.time() - t0) / 60}


# ---------------------------------------------------------------------------------------- KID
class _Incep:
    def __init__(self, dev):
        import torchvision
        m = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        self.m, self.dev = m.eval().to(dev), dev
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def feats(self, x, bs=256):
        out = []
        for i in range(0, len(x), bs):
            b = (x[i:i + bs].to(self.dev).float().clamp(-1, 1) + 1) / 2
            b = F.interpolate(b, size=(299, 299), mode="bilinear", align_corners=False)
            out.append(self.m((b - self.mean) / self.std).double())
        return torch.cat(out)


def _mmd_unbiased(kxx, kyy, kxy, ix, iy):
    mx = (ix[:, None] != ix[None, :])
    my = (iy[:, None] != iy[None, :])
    a = kxx[ix][:, ix][mx].mean()
    b = kyy[iy][:, iy][my].mean()
    return float(a + b - 2 * kxy[ix][:, iy].mean())


def kid(fx, fy, n_boot=200, seed=0):
    """Unbiased MMD^2 with kernel (x.y/d + 1)^3 (KID), x1e3; bootstrap se over both sets (pairs of
    duplicated indices are excluded from the within-set sums)."""
    d = fx.shape[1]
    kxx, kyy, kxy = (fx @ fx.T / d + 1) ** 3, (fy @ fy.T / d + 1) ** 3, (fx @ fy.T / d + 1) ** 3
    ix, iy = torch.arange(len(fx), device=fx.device), torch.arange(len(fy), device=fy.device)
    est = _mmd_unbiased(kxx, kyy, kxy, ix, iy)
    g = torch.Generator(device="cpu").manual_seed(seed)
    bs = [_mmd_unbiased(kxx, kyy, kxy, torch.randint(len(fx), (len(fx),), generator=g).to(fx.device),
                        torch.randint(len(fy), (len(fy),), generator=g).to(fy.device)) for _ in range(n_boot)]
    bs = torch.tensor(bs)
    return {"kid_x1e3": est * 1e3, "se_x1e3": float(bs.std()) * 1e3, "n_x": len(fx), "n_y": len(fy)}


# ----------------------------------------------------------------------------------------- eval
def eval_runs(runs, out_name, judge="judge/cifar10_judge.pth", folder="iter02", n_first=32, n_worst=16,
              kid_refs=True, n_boot=200):
    """runs: list of dicts
         {"name", "kind": "transport", "path", "key" (default s1.0), "direction"}
         {"name", "kind": "prior", "path", "cls"}           (cls = class the model is meant to produce)
       Writes /vol/<folder>/<out_name>_stats.json and <out_name>_small.pt (cloned slices only)."""
    t0, dev = time.time(), torch.device("cuda")
    stats_path = _fresh(f"{VOL}/{folder}/{out_name}_stats.json")
    small_path = _fresh(f"{VOL}/{folder}/{out_name}_small.pt")
    net = ev.load_judge(f"{VOL}/{judge}", dev)
    inc = _Incep(dev)
    real = {c: cifar("test", c)[0] for c in (CAT, DOG)}
    freal = {c: inc.feats(real[c]) for c in (CAT, DOG)}
    stats = {"runs": {}, "kid_refs": {}, "ceilings": {}}
    small = {}
    for c in (CAT, DOG):
        stats["ceilings"][f"all_test_{ev.CLASSES[c]}"] = ev.rate_stats(ev.judge_probs(net, real[c], dev), c)
        stats["ceilings"][f"first512_test_{ev.CLASSES[c]}"] = ev.rate_stats(ev.judge_probs(net, real[c][:512], dev), c)
    if kid_refs:
        for c, o in ((DOG, CAT), (CAT, DOG)):
            tr = cifar("train", c, 512)[0]
            stats["kid_refs"][f"train512_{ev.CLASSES[c]}_vs_test_{ev.CLASSES[c]}"] = kid(inc.feats(tr), freal[c], n_boot)
            stats["kid_refs"][f"test512_{ev.CLASSES[o]}_vs_test_{ev.CLASSES[c]}"] = kid(inc.feats(real[o][:512]), freal[c], n_boot)
    for r in runs:
        if not os.path.exists(r["path"]):
            print(f"eval: missing {r['path']}, skipped", flush=True)
            continue
        d = torch.load(r["path"])
        if r["kind"] == "transport":
            src, tgt = DIRS[r["direction"]]
            key = r.get("key", "s1.0")
            x, y = d["x"], d["outputs"][key]
            s, per = ev.evaluate_pair(net, x, y, src, tgt, dev)
            yc = y.clamp(-1, 1)
            s["cost_clamped"] = ev.mean_se(ev.sq_cost(x, yc))
            s["kid_vs_all_test_target"] = kid(inc.feats(y), freal[tgt], n_boot)
            s["meta"] = {**d["meta"].get(key, {}), "run": d["meta"].get("_run", {}), "file": r["path"]}
            pt = per["p_out"][:, tgt]
            worst = pt.argsort()[:n_worst]
            small[r["name"]] = {"x": x[:n_first].clone(), "y": y[:n_first].clone(),
                                "worst_idx": worst.clone(), "worst_x": x[worst].clone(), "worst_y": y[worst].clone(),
                                "worst_p_target": pt[worst].clone(), "p_target": pt.clone(),
                                "pred": per["p_out"].argmax(1).clone(), "cost": per["cost"].clone(),
                                "direction": r["direction"]}
            print(f"{r['name']}: cost {s['cost']['mean']:.1f} (clamped {s['cost_clamped']['mean']:.1f}) "
                  f"judge {ev.CLASSES[tgt]} {s['judge_target_on_outputs']['rate']:.3f} "
                  f"KID {s['kid_vs_all_test_target']['kid_x1e3']:.2f}", flush=True)
        else:
            c = r["cls"]
            y = d["samples"]
            p = ev.judge_probs(net, y, dev)
            s = {"judge_cat": ev.rate_stats(p, CAT), "judge_dog": ev.rate_stats(p, DOG),
                 "judge_target": ev.rate_stats(p, c), "class_histogram": torch.bincount(p.argmax(1), minlength=10).tolist(),
                 "kid_vs_all_test_target": kid(inc.feats(y), freal[c], n_boot), "meta": d["meta"],
                 "target": ev.CLASSES[c]}
            worst = p[:, c].argsort()[:n_worst]
            small[r["name"]] = {"y": y[:n_first].clone(), "worst_idx": worst.clone(), "worst_y": y[worst].clone(),
                                "worst_p_target": p[worst, c].clone(), "p_target": p[:, c].clone(),
                                "pred": p.argmax(1).clone()}
            print(f"{r['name']}: judge {ev.CLASSES[c]} {s['judge_target']['rate']:.3f} cat {s['judge_cat']['rate']:.3f} "
                  f"KID {s['kid_vs_all_test_target']['kid_x1e3']:.2f}", flush=True)
        stats["runs"][r["name"]] = s
    stats["minutes"] = (time.time() - t0) / 60
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=1)
    torch.save(small, small_path)
    return {"task": f"eval_runs {out_name}", "minutes": (time.time() - t0) / 60}


# ------------------------------------------------------------------------------ container chains
def chain_b(deadline, vol=None, smoke=False, tag="", folder="iter02", cap_min=32.0):
    """Container B: CINIC-10 download -> dog fine-tune -> prior samples + cat->dog decode -> eval.
    Fallback (download fails or exceeds its cap): guided cat->dog w=1.0 and dog->cat w=0.5 from A's latents."""
    t0, res = time.time(), []
    S = dict(steps=10) if smoke else {}
    n = 4 if smoke else 512
    r = build_cinic(smoke=smoke)
    res.append(r)
    if smoke:  # no CINIC tensor in the smoke run: use CIFAR-train dogs for both halves
        os.makedirs(f"{VOL}/{folder}", exist_ok=True)
        fake = f"{VOL}/{folder}/{tag}fake_cinic.pt"
        if not os.path.exists(fake):
            d = torch.load(f"{VOL}/data/cifar10_train.pt")
            torch.save({"x": d["x"][d["y"] == DOG][:512].clone()}, fake)
    if r.get("ok"):
        if vol is not None:
            vol.commit()
        res.append(finetune_dog_cinic(out=f"{folder}/{tag}dog_cinic.pth", mid=f"{folder}/{tag}dog_cinic_mid.pth",
                                      cap_min=cap_min, deadline=deadline, reserve_min=15.0,
                                      max_steps=20 if smoke else 10 ** 9, n_sub=512 if smoke else None,
                                      cinic_path=(f"{folder}/{tag}fake_cinic.pt" if smoke else None)))
        if vol is not None:
            vol.commit()
        res.append(prior_samples({"dog_cinic": f"{folder}/{tag}dog_cinic.pth"}, n=n, folder=folder, tag=tag,
                                 deadline=deadline, est_min=5.5, **S))
        if vol is not None:
            vol.reload()
        lat_tag = tag
        if not _wait_for(f"{VOL}/{folder}/{tag}latents_cat_to_dog.pt", 3.0, vol):
            lat_tag = tag + "b_"
            res.append(encode_latents("cat_to_dog", n=n, folder=folder, tag=lat_tag, **S))
        res.append(guided_decode("cat_to_dog", 0.0, tgt_ckpt=f"{folder}/{tag}dog_cinic.pth", name="cinic_w0",
                                 folder=folder, tag=lat_tag, deadline=deadline, est_min=5.5 * n / 512, **S))
        runs = [{"name": "prior_dog_cinic", "kind": "prior", "path": f"{VOL}/{folder}/{tag}prior_dog_cinic.pt", "cls": DOG},
                {"name": "cat_to_dog_cinic_w0", "kind": "transport", "direction": "cat_to_dog",
                 "path": f"{VOL}/{folder}/{lat_tag}transport_cat_to_dog_cinic_w0.pt"}]
    else:
        print("B FALLBACK: guidance-only runs", flush=True)
        res.append(guided_decode("cat_to_dog", 1.0, folder=folder, tag=tag, deadline=deadline, vol=vol,
                                 wait_min=20.0, **S))
        res.append(guided_decode("dog_to_cat", 0.5, folder=folder, tag=tag, deadline=deadline, vol=vol,
                                 wait_min=40.0, **S))
        runs = [{"name": "cat_to_dog_w1.0", "kind": "transport", "direction": "cat_to_dog",
                 "path": f"{VOL}/{folder}/{tag}transport_cat_to_dog_w1.0.pt"},
                {"name": "dog_to_cat_w0.5", "kind": "transport", "direction": "dog_to_cat",
                 "path": f"{VOL}/{folder}/{tag}transport_dog_to_cat_w0.5.pt"}]
    if vol is not None:
        vol.commit()
    res.append(eval_runs(runs, f"{tag}evalB", folder=folder, kid_refs=False, n_boot=10 if smoke else 200))
    return {"task": "chain_b", "sub": res, "minutes": (time.time() - t0) / 60}
