"""Local post-processing for iteration 03: figures (*_big.png, x4 NEAREST) + metrics.json.
Usage: python3 build_outputs.py [smoke]"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IT2 = os.path.join(HERE, "..", "iter_02_purity")
sys.path.insert(0, os.path.join(IT2, "recovered_src"))  # evaluate.py byte-identical copy (original may be evicted)
import torch  # noqa: E402

from evaluate import grid_png as _grid_png  # noqa: E402


def grid_png(rows, path):  # skip figures already written (an iCloud-evicted file stalls on rewrite)
    if os.path.exists(path):
        return
    _grid_png(rows, path)

smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
SRC = os.path.join(HERE, "smoke") if smoke else HERE
DST = SRC
pre = "smoke_" if smoke else ""

# ---- new results
stats, small, memo = {}, {}, {}
for nm in ("evalD", "evalC", "evalD_mid", "evalC_mid"):
    p = os.path.join(SRC, f"{pre}{nm}_stats.json")
    if os.path.exists(p):
        stats[nm] = json.load(open(p))
        small.update(torch.load(os.path.join(SRC, f"{pre}{nm}_small.pt")))
for nm in ("memo_dog", "memo_cat", "memo_dog_mid", "memo_cat_mid"):
    p = os.path.join(SRC, f"{pre}{nm}.json")
    if os.path.exists(p):
        memo[nm] = json.load(open(p))
# ---- iteration-02 reference (baseline iter-01 transport + old priors), evaluated with the same code/refs
old_stats = json.load(open(os.path.join(IT2, "evalA_stats.json")))
old_small = torch.load(os.path.join(IT2, "evalA_small.pt"))
BASE = {"cat_to_dog": "baseline_cat_to_dog_iter01", "dog_to_cat": "baseline_dog_to_cat_iter01"}
OLDP = {"cat_to_dog": "prior_dog", "dog_to_cat": "prior_cat"}
NEW = {"cat_to_dog": "cat_to_dog_iter03_c3", "dog_to_cat": "dog_to_cat_iter03_c3"}
NEWP = {"cat_to_dog": "prior_dog_c3", "dog_to_cat": "prior_cat_c3"}
CEIL = {"cat_to_dog": "all_test_dog", "dog_to_cat": "all_test_cat"}

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
for dr in ("cat_to_dog", "dog_to_cat"):
    if NEW[dr] in small and not smoke:
        b, n = old_small[BASE[dr]], small[NEW[dr]]
        assert torch.equal(b["x"][:32], n["x"][:32]), "inputs differ"
        grid_png([b["x"][:16], b["y"][:16], n["y"][:16]], os.path.join(DST, f"{dr}_compare_big.png"))
        grid_png([b["x"][16:32], b["y"][16:32], n["y"][16:32]], os.path.join(DST, f"{dr}_compare_b_big.png"))
# same z: old cat / new cat / old dog / new dog
rows = []
for o, n in (("prior_cat", "prior_cat_c3"), ("prior_dog", "prior_dog_c3")):
    if n in small and not smoke:
        rows += [old_small[o]["y"][:16], small[n]["y"][:16]]
if rows:
    grid_png(rows, os.path.join(DST, "prior_same_z_cat_dog_old_new_big.png"))

# ---------------------------------------------------------------- metrics.json
runs = {}
for nm, s in stats.items():
    runs.update({k: {**v, "_from": nm} for k, v in s["runs"].items()})
ceil = old_stats["ceilings"]


def row(v):
    r = {}
    if "cost" in v:
        r.update(cost=v["cost"]["mean"], cost_se=v["cost"]["stderr"], cost_clamped=v["cost_clamped"]["mean"],
                 cost_clamped_se=v["cost_clamped"]["stderr"], judge_target=v["judge_target_on_outputs"]["rate"],
                 judge_target_se=v["judge_target_on_outputs"]["stderr"], judge_source=v["judge_source_on_outputs"]["rate"])
    else:
        r.update(judge_target=v["judge_target"]["rate"], judge_target_se=v["judge_target"]["stderr"],
                 judge_cat=v["judge_cat"]["rate"], judge_dog=v["judge_dog"]["rate"],
                 class_histogram=v.get("class_histogram"))
    r.update(kid_x1e3=v["kid_vs_all_test_target"]["kid_x1e3"], kid_se_x1e3=v["kid_vs_all_test_target"]["se_x1e3"])
    return r


table = {k: row(v) for k, v in runs.items()}
ref = {k: row(old_stats["runs"][k]) for k in list(BASE.values()) + list(OLDP.values())}
crit = {}
for dr, cls in (("cat_to_dog", "dog"), ("dog_to_cat", "cat")):
    if NEW[dr] not in table or NEWP[dr] not in table:
        continue
    t, b, p, op = table[NEW[dr]], ref[BASE[dr]], table[NEWP[dr]], ref[OLDP[dr]]
    c = ceil[CEIL[dr]]["rate"]
    m = memo.get(f"memo_{cls}", {}).get("sets", {}).get(f"prior_{cls}_c3", {})
    kid_se = (t["kid_se_x1e3"] ** 2 + b["kid_se_x1e3"] ** 2) ** 0.5
    crit[dr] = {
        "1_judge_ge_0.79_and_le_ceiling_plus_0.02": bool(0.79 <= t["judge_target"] <= c + 0.02),
        "2_kid_le_baseline_within_1se": bool(t["kid_x1e3"] <= b["kid_x1e3"] + t["kid_se_x1e3"]),
        "2_strict_kid_le_baseline": bool(t["kid_x1e3"] <= b["kid_x1e3"]),
        "3_clamped_cost_le_110": bool(t["cost_clamped"] <= 110),
        "4_prior_judge_up": bool(p["judge_target"] > op["judge_target"]),
        "4_prior_kid_le_old": bool(p["kid_x1e3"] <= op["kid_x1e3"]),
        "4_no_memorisation_flag": (not m.get("memorisation_flag", True)) if m else None,
        "values": {"judge": t["judge_target"], "judge_base": b["judge_target"], "ceiling": c,
                   "kid": t["kid_x1e3"], "kid_se": t["kid_se_x1e3"], "kid_base": b["kid_x1e3"],
                   "kid_base_se": b["kid_se_x1e3"], "kid_diff_combined_se": kid_se,
                   "cost_clamped": t["cost_clamped"], "cost_base_clamped": b["cost_clamped"],
                   "prior_judge": p["judge_target"], "prior_judge_old": op["judge_target"],
                   "prior_kid": p["kid_x1e3"], "prior_kid_old": op["kid_x1e3"],
                   "memo_ratio": m.get("ratio_to_test_median")},
        "5_by_eye": "see REPORT.md",
    }
gpu = []
# gpu_log/*.json copies are iCloud-evicted on this Mac (disk full); read the identical RESULT line of the client logs
for f in sorted(glob.glob(os.path.join(HERE, "..", "..", "..", "logs", "iter03_*.log"))):
    if "try1" in f or smoke:
        continue
    lines = [ln for ln in open(f, errors="replace") if ln.startswith("RESULT ")]
    if not lines:
        continue
    g = json.loads(lines[-1][7:])
    gpu.append({"file": os.path.basename(f), "container_minutes": g.get("container_minutes")})
    for r in g.get("results", []):
        for s in r.get("sub", []):
            if s.get("task", "").startswith("finetune_cifar"):
                gpu[-1]["finetune"] = {k: s.get(k) for k in ("steps", "images_seen", "accum", "cap_min", "peak_mem_gb",
                                                             "minutes")}
                gpu[-1]["loss_log_last"] = s.get("loss_log", [])[-3:]
out = {"summary_table": table, "reference_iter02_eval": ref, "criteria": crit, "memorisation": memo,
       "judge_ceilings": ceil, "gpu": gpu, "runs": runs}
json.dump(out, open(os.path.join(DST, "metrics.json"), "w"), indent=1)
for k, r in {**ref, **table}.items():
    print(k, {a: (round(v, 4) if isinstance(v, float) else v) for a, v in r.items() if a != "class_histogram"})
print(json.dumps(crit, indent=1))
print("memo", {k: {s: round(v["ratio_to_test_median"], 3) for s, v in m["sets"].items()} for k, m in memo.items()})
print("gpu", gpu)
