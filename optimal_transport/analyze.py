"""Figures and a cost table from the Experiment 3 results (cheap; runs locally).

    python optimal_transport/analyze.py

Reads results/ddib.pt and results/ot_flow.pt (written by modal_ot.py) and writes:
    results/cat_to_dog_big.png   rows: real test cat | composed flows | learned Brenier map
    results/dog_to_cat_big.png   same, starting from real test dogs
    results/model_pairs_big.png  model cat and model dog decoded from the same noise
    results/train_fit_big.png    train cat | its exact-OT partner dog | learned map's output
    results/summary.json
"""
import json
import os

import torch
import torchvision
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def grid(rows, name, k=16, scale=4):
    img = ((torch.cat([r[:k] for r in rows]) + 1) / 2).clamp(0, 1)
    path = os.path.join(R, f"{name}.png")
    torchvision.utils.save_image(img, path, nrow=k, padding=1)
    im = Image.open(path)
    im.resize((im.width * scale, im.height * scale), Image.NEAREST).save(path.replace(".png", "_big.png"))


def cost(a, b):
    c = (a - b).flatten(1).pow(2).sum(1)
    return {"mean": round(c.mean().item(), 1), "stderr": round((c.std() / len(c) ** 0.5).item(), 1)}


def main():
    dd = torch.load(os.path.join(R, "ddib.pt"), weights_only=False)
    of = torch.load(os.path.join(R, "ot_flow.pt"), weights_only=False)
    s = {"exact OT, 5000 train cats vs 5000 train dogs": round(of["cost_train_ot"], 1),
         "exact OT, 1000 test cats vs 1000 test dogs": round(of["cost_test_ot"], 1)}
    for k in ("cat_to_dog", "dog_to_cat"):
        x = dd[k]["x"]
        n = min(len(x), len(of[k]["x"]))
        assert torch.equal(x[:n], of[k]["x"][:n]), "both methods should start from the same test images"
        s[f"{k}: composed flows"] = cost(x[:n], dd[k]["y"][:n])
        s[f"{k}: learned Brenier map"] = cost(x[:n], of[k]["y"][:n])
        s[f"{k}: composed-flow cycle error"] = cost(x[:n], dd[k]["recon"][:n])
        grid([x, dd[k]["y"], of[k]["y"]], k)
    mp = dd["model_pairs"]
    s["model cat vs model dog, same noise"] = cost(mp["cat"], mp["dog"])
    grid([mp["cat"], mp["dog"]], "model_pairs")
    tf = of["train_fit"]
    s["train cats: learned map vs exact-OT partner"] = cost(tf["y"], tf["ot_partner"])
    grid([tf["x"], tf["ot_partner"], tf["y"]], "train_fit")
    with open(os.path.join(R, "summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    for k, v in s.items():
        print(f"{k:50s} {v}")


if __name__ == "__main__":
    main()
