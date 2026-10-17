"""Local post-processing for iteration 05: metrics.json + printed tables (figures are rendered on Modal).
Usage: python3 build_outputs.py [smoke]"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "smoke") if len(sys.argv) > 1 and sys.argv[1] == "smoke" else HERE
J = lambda n: json.load(open(os.path.join(D, n)))  # noqa: E731
FT = {"cat_to_dog": {"kid": 2.92, "kid_se": 0.92, "judge": 0.748}, "dog_to_cat": {"kid": 2.16, "kid_se": 0.64, "judge": 0.746}}

out = {"per_snapshot_prior": {}}
for c in ("cat", "dog"):
    p = os.path.join(D, f"snapstats_{c}.json")
    if not os.path.exists(p):
        continue
    s = json.load(open(p))
    rows = []
    for name, v in s["snapshots"].items():
        rows.append({"snapshot": name, "step": v["step"], "ema": v["ema"], "reference_iter04": v["reference"],
                     "train_min": round(v["train_minutes"], 1),
                     "judge_target": v["judge_target"]["rate"], "judge_target_se": v["judge_target"]["stderr"],
                     "kid_x1e3": v["kid_vs_all_test"]["kid_x1e3"], "kid_se_x1e3": v["kid_vs_all_test"]["se_x1e3"],
                     "kid_tweedie_x1e3": v["kid_vs_all_test_tweedie"]["kid_x1e3"],
                     "kid_tweedie_se_x1e3": v["kid_vs_all_test_tweedie"]["se_x1e3"],
                     "judge_target_tweedie": v["judge_target_tweedie"]["rate"],
                     "memo_ratio": v["memo"]["ratio_to_test_median"], "memo_flag": v["memo"]["memorisation_flag"],
                     "memo_frac_below_10pct": v["memo"]["frac_below_10pct_test_median"],
                     "frac_samples_saturated": v["frac_samples_saturated"],
                     "frac_pixels_outside": v["frac_pixels_outside_[-1,1]"], "class_histogram": v["class_histogram"]})
    out["per_snapshot_prior"][c] = {"rows": sorted(rows, key=lambda r: (r["step"], r["ema"])), "pick": s["pick"],
                                    "kid_floor_train512": s["kid_floor_train512_vs_test"],
                                    "ceiling": s["ceiling_all_test"]["rate"], "n_samples": s["n_samples"],
                                    "heun_steps": s["heun_steps"]}
    print(f"\n{c}: snapshot | EMA | train-min | judge | KID (se) | KID tw | memo | sat")
    for r in out["per_snapshot_prior"][c]["rows"]:
        print(f"  {r['snapshot']} | {r['ema']} | {r['train_min']} | {r['judge_target']:.3f} | {r['kid_x1e3']:.2f} "
              f"({r['kid_se_x1e3']:.2f}) | {r['kid_tweedie_x1e3']:.2f} | {r['memo_ratio']:.3f} | "
              f"{r['frac_samples_saturated']:.4f}")
    print("  pick:", s["pick"]["name"], s["pick"].get("within_1se"))

if os.path.exists(os.path.join(D, "evalT_stats.json")):
    evT, e5 = J("evalT_stats.json"), J("eval05.json")
    comp = {}
    print("\nrun | cost | cost clamped | judge target | source on out | KID (se) | sat | outside")
    for name, v in evT["runs"].items():
        if name.startswith("prior_"):
            continue
        direction = "cat_to_dog" if name.startswith("cat_to_dog") else "dog_to_cat"
        r = {"cost": v["cost"]["mean"], "cost_se": v["cost"]["stderr"],
             "cost_clamped": v["cost_clamped"]["mean"], "cost_clamped_se": v["cost_clamped"]["stderr"],
             "judge_target": v["judge_target_on_outputs"]["rate"], "judge_target_se": v["judge_target_on_outputs"]["stderr"],
             "judge_source_on_outputs": v["judge_source_on_outputs"]["rate"],
             "kid_x1e3": v["kid_vs_all_test_target"]["kid_x1e3"], "kid_se_x1e3": v["kid_vs_all_test_target"]["se_x1e3"],
             "output_class_histogram": v["output_class_histogram"], "output_range": v["output_range"],
             "frac_pixels_outside": v["frac_output_pixels_outside_[-1,1]"],
             "frac_samples_saturated": e5["saturated"].get(name)}
        f = FT[direction]
        r["criterion4_kid_le_FT_plus_1se"] = r["kid_x1e3"] <= f["kid"] + f["kid_se"]
        r["criterion4_judge_ge_FT_plus_2pt"] = r["judge_target"] >= f["judge"] + 0.02
        comp[name] = r
        print(f"  {name} | {r['cost']:.1f} ± {r['cost_se']:.1f} | {r['cost_clamped']:.1f} | {r['judge_target']:.3f} | "
              f"{r['judge_source_on_outputs']:.3f} | {r['kid_x1e3']:.2f} ({r['kid_se_x1e3']:.2f}) | "
              f"{r['frac_samples_saturated']} | {r['frac_pixels_outside']:.4f}")
    out["transports"] = comp
    out["chosen_priors_evalT"] = {k: {"judge_target": v["judge_target"]["rate"],
                                      "kid_x1e3": v["kid_vs_all_test_target"]["kid_x1e3"],
                                      "kid_se_x1e3": v["kid_vs_all_test_target"]["se_x1e3"]}
                                  for k, v in evT["runs"].items() if k.startswith("prior_")}
    out["judge_ceilings"] = evT["ceilings"]
    out["kid_floors"] = evT["kid_refs"]
    out["gaussianity"] = e5["gaussianity"]
    print("\nlatent set | std | |z|^2/d mean (std) | pool4 (se) | pool8 (se) | lag1 h/v | KS | flagged")
    for k, v in e5["gaussianity"].items():
        print(f"  {k} | {v['std']:.4f} | {v['norm2_over_d']['mean']:.4f} ({v['norm2_over_d']['std']:.4f}) | "
              f"{v['pooled4_energy_ratio']:.3f} ({v['pooled4_energy_ratio_se']:.3f}) | {v['pooled8_energy_ratio']:.3f} "
              f"({v['pooled8_energy_ratio_se']:.3f}) | {v['lag1_autocorr_h']:.4f}/{v['lag1_autocorr_v']:.4f} | "
              f"{v['ks_vs_N01']:.4f} | {v['flags_vs_N01_fresh']['flagged']}")
    out["steering_grid"] = e5["steering"]
    out["same_noise"] = e5["same_noise"]
    out["best_pairs"] = e5["best_pairs"]
    out["picked"] = e5["picked"]
    out["mixed_pairs"] = {"M1": "FT cat encoder latents (iter02) -> iter05 dog decoder (cat_to_dog)",
                          "M2": "FT dog encoder latents (iter02) -> iter05 cat decoder (dog_to_cat)",
                          "M3": "iter05 cat encoder latents -> FT dog decoder out/dog.pth (cat_to_dog)",
                          "M4": "iter05 dog encoder latents -> FT cat decoder out/cat.pth (dog_to_cat)"}
    print("\nsteering:", [(c["setting"], round(c["cost_clamped_mean"], 1), c["judge_dog_rate"]) for c in e5["steering"]["columns"]])
    print("same noise:", {k: round(v["matched_over_mismatched"], 3) for k, v in e5["same_noise"].items() if isinstance(v, dict)})
    print("best pairs qualifying:", {k: v["n_qualifying"] for k, v in e5["best_pairs"].items()})

gpu = []
for f in sorted(glob.glob(os.path.join(HERE, "gpu_log", "*.json"))):
    g = json.load(open(f))
    gpu.append({"file": os.path.basename(f), "container_minutes": g["container_minutes"],
                "tasks": [(r.get("task"), r.get("minutes"), r.get("error")) for r in g["results"]]})
out["gpu"] = gpu
if os.path.exists(os.path.join(D, "cost_quantiles.json")):
    out["cost_quantiles_clamped"] = J("cost_quantiles.json")
if D == HERE:
    with open(os.path.join(HERE, "metrics.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote metrics.json")
