"""Local post-processing for iteration 04: metrics.json + printed tables (figures are rendered on Modal).
Usage: python3 build_outputs.py"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda n: json.load(open(os.path.join(HERE, n)))  # noqa: E731

snap = {c: J(f"snapstats_{c}_all.json") for c in ("cat", "dog")}
ev4 = J("eval04.json") if os.path.exists(os.path.join(HERE, "eval04.json")) else None
evT = J("evalT_stats.json") if os.path.exists(os.path.join(HERE, "evalT_stats.json")) else None
old = J("../iter_02_purity/evalA_stats.json")

per_snap = {}
for c, s in snap.items():
    rows = []
    for name, v in s["snapshots"].items():
        rows.append({"snapshot": name, "step": v["step"], "train_min": round(v["train_minutes"], 1),
                     "judge_target": v["judge_target"]["rate"], "judge_target_se": v["judge_target"]["stderr"],
                     "kid_x1e3": v["kid_vs_all_test"]["kid_x1e3"], "kid_se_x1e3": v["kid_vs_all_test"]["se_x1e3"],
                     "memo_ratio": v["memo"]["ratio_to_test_median"], "memo_median": v["memo"]["median"],
                     "memo_frac_below_10pct": v["memo"]["frac_below_10pct_test_median"],
                     "memo_flag": v["memo"]["memorisation_flag"], "class_histogram": v["class_histogram"]})
    per_snap[c] = {"rows": sorted(rows, key=lambda r: r["step"]), "pick": s["pick"],
                   "kid_floor_train256": s["kid_floor_train_vs_test"], "ceiling": s["ceiling_all_test"]["rate"],
                   "memo_test_median": s["memo_test_first512"]["median"], "heun_steps": s["heun_steps"],
                   "n_samples": s["n_samples"]}

out = {"per_snapshot_prior": per_snap}
if evT:
    comp = {}
    for d, base in (("cat_to_dog", "baseline_cat_to_dog_iter01"), ("dog_to_cat", "baseline_dog_to_cat_iter01")):
        v, b = evT["runs"][f"{d}_scratch"], old["runs"][base]
        f = lambda v: {"cost": v["cost"]["mean"], "cost_se": v["cost"]["stderr"],  # noqa: E731
                       "cost_clamped": v["cost_clamped"]["mean"], "cost_clamped_se": v["cost_clamped"]["stderr"],
                       "judge_target": v["judge_target_on_outputs"]["rate"],
                       "judge_target_se": v["judge_target_on_outputs"]["stderr"],
                       "judge_source_on_outputs": v["judge_source_on_outputs"]["rate"],
                       "kid_x1e3": v["kid_vs_all_test_target"]["kid_x1e3"],
                       "kid_se_x1e3": v["kid_vs_all_test_target"]["se_x1e3"],
                       "output_class_histogram": v["output_class_histogram"],
                       "frac_pixels_outside": v["frac_output_pixels_outside_[-1,1]"]}
        comp[d] = {"scratch_iter04": f(v), "finetuned_best_iter01": f(b)}
    out["composed_ode"] = comp
    out["judge_ceilings"] = evT["ceilings"]
    out["kid_floors"] = evT["kid_refs"]
    out["chosen_priors_eval512refs"] = {k: {"judge_target": v["judge_target"]["rate"],
                                            "kid_x1e3": v["kid_vs_all_test_target"]["kid_x1e3"]}
                                        for k, v in evT["runs"].items() if k.startswith("prior_")}
if ev4:
    out["steering_grid"] = ev4["steering"]
    out["same_noise"] = ev4["same_noise"]
    out["best_pairs"] = ev4["best_pairs"]
    out["picked"] = ev4["picked"]
gpu = []
for f in sorted(glob.glob(os.path.join(HERE, "gpu_log", "*.json"))):
    g = json.load(open(f))
    e = {"file": os.path.basename(f), "container_minutes": g["container_minutes"]}
    for r in g["results"]:
        for s in r.get("sub", []):
            if s.get("task", "").startswith("train"):
                e["train"] = {k: s.get(k) for k in ("steps", "images_seen", "epochs", "it_per_s", "n_params",
                                                    "peak_mem_gb", "minutes")}
                e["loss_log_last"] = s.get("loss_log", [])[-3:]
    gpu.append(e)
out["gpu"] = gpu
json.dump(out, open(os.path.join(HERE, "metrics.json"), "w"), indent=1)

for c, p in per_snap.items():
    print(f"\n{c}: ceiling {p['ceiling']:.3f}, KID floor (256 train) {p['kid_floor_train256']['kid_x1e3']:.2f}, "
          f"pick {p['pick']['name']} ({p['pick']['rule']})")
    for r in p["rows"]:
        print(f"  {r['snapshot']:>10} {r['train_min']:5.1f} min judge {r['judge_target']:.3f} KID {r['kid_x1e3']:.2f}"
              f"+-{r['kid_se_x1e3']:.2f} memo {r['memo_ratio']:.3f} (<10%: {r['memo_frac_below_10pct']:.3f})")
if evT:
    for d, v in out["composed_ode"].items():
        for k, r in v.items():
            print(d, k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()})
if ev4:
    for s in ev4["steering"]["columns"]:
        print(s)
    print(json.dumps(ev4["same_noise"], indent=1))
    print({d: v["n_qualifying"] for d, v in ev4["best_pairs"].items()})
print(gpu)
