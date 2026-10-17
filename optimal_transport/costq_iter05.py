"""CPU-only helper (no GPU): per-image cost quantiles of the iteration-05 transports, with saturated outputs counted
separately. Run: modal run optimal_transport/costq_iter05.py  (prints JSON; writes nothing to the volume)."""
import json
import os

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.7.1", "numpy<2")
vol = modal.Volume.from_name("follmer-cat-dog")

app = modal.App("follmer-cat-dog-iter05-costq")


@app.function(image=image, volumes={"/vol": vol}, cpu=4, timeout=900)
def costq():
    import torch
    files = {}
    for d in ("cat_to_dog", "dog_to_cat"):
        for m in ("iter05", "M1", "M2", "M3", "M4"):
            p = f"/vol/iter05/transport_{d}_{m}.pt"
            if os.path.exists(p):
                files[f"{d}_{m}"] = (p, "s1.0")
    files.update({"cat_to_dog_FT": ("/vol/iter01/transport_cat_to_dog_B.pt", "s1.0"),
                  "dog_to_cat_FT": ("/vol/iter01/transport_dog_to_cat_A.pt", "s1.0"),
                  "cat_to_dog_iter04": ("/vol/iter04/transport_cat_to_dog_scratch.pt", "s1.0"),
                  "dog_to_cat_iter04": ("/vol/iter04/transport_dog_to_cat_scratch.pt", "s1.0")})
    out = {}
    for k, (p, key) in files.items():
        d = torch.load(p)
        x, y = d["x"][:512].float(), d["outputs"][key][:512].float()
        c = (x - y.clamp(-1, 1)).flatten(1).pow(2).sum(1)
        sat = ((y < -1) | (y > 1)).flatten(1).float().mean(1) > 0.5
        out_frac = ((y < -1) | (y > 1)).flatten(1).float().mean(1)
        q = lambda v, a: float(v.quantile(a))  # noqa: E731
        out[k] = {"mean_clamped": float(c.mean()), "median_clamped": q(c, 0.5), "q90": q(c, 0.9),
                  "n_saturated": int(sat.sum()), "mean_clamped_excl_saturated": float(c[~sat].mean()),
                  "n_over_10pct_pixels_clipped": int((out_frac > 0.1).sum()),
                  "mean_clamped_excl_over10pct_clipped": float(c[out_frac <= 0.1].mean())}
    return out


@app.local_entrypoint()
def main():
    print("COSTQ", json.dumps(costq.remote()))
