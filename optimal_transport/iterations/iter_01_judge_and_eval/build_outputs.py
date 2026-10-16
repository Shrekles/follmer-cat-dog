"""Local post-processing for iteration 01: metrics.json + *_big.png figures from the downloaded
stats.json / small.pt (and gpu_log/*.json). Usage: python build_outputs.py [subdir_prefix]"""
import glob
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import evaluate as ev  # noqa: E402

smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
src_dir = os.path.join(HERE, "smoke") if smoke else HERE
tag = "smoke_" if smoke else ""
stats = json.load(open(os.path.join(src_dir, f"{tag}stats.json")))
small = torch.load(os.path.join(src_dir, f"{tag}small.pt"))

# --- figures
for d, v in small.items():
    keys = sorted(v["outputs"])
    full = keys[-1]
    ev.grid_png([v["x"][:16], v["outputs"][full][:16]], os.path.join(src_dir, f"{tag}{d}_full_big.png"))
    ev.grid_png([v["x"][:16]] + [v["outputs"][k][:16] for k in keys], os.path.join(src_dir, f"{tag}{d}_sweep_big.png"))
    ev.grid_png([v["x"][16:32], v["outputs"][full][16:32]], os.path.join(src_dir, f"{tag}{d}_full_b_big.png"))

# --- metrics
gpu = []
for f in sorted(glob.glob(os.path.join(HERE, "gpu_log", "*.json"))):
    g = json.load(open(f))
    gpu.append({"file": os.path.basename(f), "container_minutes": g["container_minutes"],
                "tasks": [(r["task"], round(r["minutes"], 2)) for r in g["results"]]})
J = stats["judge"]
m = {
    "judge": {k: J[k] for k in ("arch", "epochs", "test_acc", "test_cat_called_cat", "test_dog_called_dog",
                                "test_mean_prob_cat_on_cats", "test_mean_prob_dog_on_dogs", "test_per_class_acc",
                                "test_cat_pred_hist", "test_dog_pred_hist", "train_minutes")},
    "transport": {},
    "gpu_minutes": {"jobs": gpu, "total_container_minutes": sum(g["container_minutes"] for g in gpu)},
}
for d, per in stats["directions"].items():
    m["transport"][d] = {}
    for key, s in per.items():
        m["transport"][d][key] = {
            "valid_coupling": s["meta"]["valid_coupling"], "s": s["meta"]["s"],
            "signal_coef_alpha": s["meta"]["signal_coef_alpha"],
            "steps_encode_decode": [s["meta"]["encode_steps"], s["meta"]["decode_steps"]],
            "cost_mean": s["cost"]["mean"], "cost_stderr": s["cost"]["stderr"], "n": s["cost"]["n"],
            "judge_target_rate_on_outputs": s["judge_target_on_outputs"]["rate"],
            "judge_target_rate_stderr": s["judge_target_on_outputs"]["stderr"],
            "judge_target_mean_prob_on_outputs": s["judge_target_on_outputs"]["mean_prob"],
            "judge_source_rate_on_outputs": s["judge_source_on_outputs"]["rate"],
            "judge_source_rate_on_inputs": s["judge_source_on_inputs"]["rate"],
            "judge_source_mean_prob_on_inputs": s["judge_source_on_inputs"]["mean_prob"],
            "output_class_histogram": s["output_class_histogram"],
            "frac_output_pixels_outside_[-1,1]": s["frac_output_pixels_outside_[-1,1]"],
        }
out = os.path.join(src_dir, f"{tag}metrics.json")
json.dump(m, open(out, "w"), indent=1)
print(json.dumps(m, indent=1))
