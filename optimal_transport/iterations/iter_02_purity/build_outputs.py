"""Local post-processing for iteration 02: figures (*_big.png, x4 NEAREST) + metrics.json.
Usage: python3 build_outputs.py [smoke]"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "recovered_src"))  # evaluate.py (byte-identical copy; original is iCloud-evicted)
import torch  # noqa: E402

from evaluate import grid_png  # noqa: E402

smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
SRC = os.path.join(HERE, "smoke") if smoke else HERE
DST = SRC
pre = "smoke_" if smoke else ""

stats, small = {}, {}
for nm in ("evalA", "evalB"):
    p = os.path.join(SRC, f"{pre}{nm}_stats.json")
    if os.path.exists(p):
        stats[nm] = json.load(open(p))
        small.update(torch.load(os.path.join(SRC, f"{pre}{nm}_small.pt")))

# ---------------------------------------------------------------- figures
for name, d in small.items():
    if "x" in d:  # transport
        grid_png([d["x"][:16], d["y"][:16]], os.path.join(DST, f"{name}_big.png"))
        if len(d["x"]) > 16:
            grid_png([d["x"][16:32], d["y"][16:32]], os.path.join(DST, f"{name}_b_big.png"))
        grid_png([d["worst_x"], d["worst_y"]], os.path.join(DST, f"{name}_lowest_ptarget_big.png"))
    else:  # prior samples
        grid_png([d["y"][:16], d["y"][16:32]], os.path.join(DST, f"{name}_big.png"))
        grid_png([d["worst_y"]], os.path.join(DST, f"{name}_lowest_ptarget_big.png"))
# same-z prior comparison: rows cat model / dog model / CINIC dog model
rows = [small[k]["y"][:16] for k in ("prior_cat", "prior_dog", "prior_dog_cinic") if k in small]
if rows:
    grid_png(rows, os.path.join(DST, "prior_same_z_big.png"))
# cat->dog comparison on the same 16 inputs: input / baseline / CINIC / w0.5 / w1.5
cmp = [k for k in ("baseline_cat_to_dog_iter01", "cat_to_dog_cinic_w0", "cat_to_dog_w0.5", "cat_to_dog_w1.5") if k in small]
if cmp:
    grid_png([small[cmp[0]]["x"][:16]] + [small[k]["y"][:16] for k in cmp], os.path.join(DST, "cat_to_dog_compare_big.png"))
cmp = [k for k in ("baseline_dog_to_cat_iter01", "dog_to_cat_w1.0", "dog_to_cat_w0.5") if k in small]
if cmp:
    grid_png([small[cmp[0]]["x"][:16]] + [small[k]["y"][:16] for k in cmp], os.path.join(DST, "dog_to_cat_compare_big.png"))

# ---------------------------------------------------------------- metrics.json
runs, refs, ceil = {}, {}, {}
for nm, s in stats.items():
    runs.update({k: {**v, "_from": nm} for k, v in s["runs"].items()})
    refs.update(s.get("kid_refs", {}))
    ceil.update(s.get("ceilings", {}))
table = []
for k, v in runs.items():
    row = {"run": k}
    if "cost" in v:
        row.update(cost=v["cost"]["mean"], cost_se=v["cost"]["stderr"], cost_clamped=v["cost_clamped"]["mean"],
                   cost_clamped_se=v["cost_clamped"]["stderr"], judge_target=v["judge_target_on_outputs"]["rate"],
                   judge_source=v["judge_source_on_outputs"]["rate"],
                   valid_coupling=v["meta"].get("valid_coupling", None), w=v["meta"].get("w", 0.0))
    else:
        row.update(judge_target=v["judge_target"]["rate"], judge_cat=v["judge_cat"]["rate"], judge_dog=v["judge_dog"]["rate"])
    row.update(kid_x1e3=v["kid_vs_all_test_target"]["kid_x1e3"], kid_se_x1e3=v["kid_vs_all_test_target"]["se_x1e3"])
    table.append(row)
gpu = []
for f in sorted(glob.glob(os.path.join(HERE, "gpu_log", "*.json"))):
    g = json.load(open(f))
    gpu.append({"file": os.path.basename(f), "container_minutes": g.get("container_minutes")})
out = {"summary_table": table, "kid_refs": refs, "judge_ceilings": ceil, "gpu": gpu, "runs": runs}
json.dump(out, open(os.path.join(DST, "metrics.json"), "w"), indent=1)
for r in table:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print("refs", {k: round(v["kid_x1e3"], 3) for k, v in refs.items()})
print("ceilings", {k: round(v["rate"], 3) for k, v in ceil.items()})
print("gpu", gpu)
