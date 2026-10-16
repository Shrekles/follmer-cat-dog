"""Modal entry point for the iteration loop (pattern of modal_ot.py). Run from the repository root.

    modal run optimal_transport/modal_iter.py --chain smoke     # tiny versions of every task
    modal run optimal_transport/modal_iter.py --chain A         # judge, then dog->cat full + s*=0.4
    modal run optimal_transport/modal_iter.py --chain B         # cat->dog full + sweep, dog->cat s*=0.6,0.8
    modal run optimal_transport/modal_iter.py --chain eval      # judge_eval, downloads stats + small tensors

Each chain runs its tasks sequentially in ONE A10G container. GPU tasks live in tasks_iter.py.
"""
import json
import os
import time

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_SDE_COMMIT = "cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44"
GPU = os.environ.get("MODAL_GPU", "A10G")
ITER_DIR = os.path.join(ROOT, "optimal_transport/iterations/iter_01_judge_and_eval")

app = modal.App("follmer-cat-dog-iter")
vol = modal.Volume.from_name("follmer-cat-dog")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "ml_collections", "numpy<2", "scipy", "pillow")
    .run_commands(
        "git clone https://github.com/yang-song/score_sde_pytorch /root/proj/score_sde_pytorch",
        f"cd /root/proj/score_sde_pytorch && git checkout {SCORE_SDE_COMMIT}",
    )
    .add_local_dir(os.path.join(ROOT, "common"), "/root/proj/common", ignore=["__pycache__"])
    .add_local_dir(os.path.join(ROOT, "optimal_transport"), "/root/proj/optimal_transport",
                   ignore=["__pycache__", "results", "iterations"])
)

S = dict(steps=10, n=4, bs=4)
CHAINS = {
    "smoke": [
        ("train_judge", dict(epochs=1, n_train=2048, out="judge/smoke_cifar10_judge.pth")),
        ("transport_pf", dict(direction="cat_to_dog", stops=(0.4, 0.6, 0.8), full=True, tag="smoke_", name="B", **S)),
        ("transport_pf", dict(direction="dog_to_cat", stops=(0.4,), full=True, tag="smoke_", name="A", **S)),
        ("transport_pf", dict(direction="dog_to_cat", stops=(0.6, 0.8), full=False, tag="smoke_", name="B", **S)),
        ("judge_eval", dict(tag="smoke_", judge="judge/smoke_cifar10_judge.pth")),
    ],
    "A": [
        ("train_judge", dict()),
        ("transport_pf", dict(direction="dog_to_cat", stops=(0.4,), full=True, name="A")),
    ],
    "B": [
        ("transport_pf", dict(direction="cat_to_dog", stops=(0.4, 0.6, 0.8), full=True, name="B")),
        ("transport_pf", dict(direction="dog_to_cat", stops=(0.6, 0.8), full=False, name="B")),
    ],
    "eval": [("judge_eval", dict())],
}


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=2 * 3600)
def run(chain: list):
    import sys
    sys.path.insert(0, "/root/proj/optimal_transport")
    import tasks_iter
    t0 = time.time()
    results = []
    for task, kwargs in chain:
        print(f"== {task} {kwargs}", flush=True)
        vol.reload()
        r = getattr(tasks_iter, task)(**kwargs)
        vol.commit()
        results.append(r)
        print(f"== done {task}: {r.get('minutes', 0):.1f} min", flush=True)
    return {"container_minutes": (time.time() - t0) / 60, "results": results}


@app.local_entrypoint()
def main(chain: str = "smoke"):
    print("chain", chain, "on", GPU, flush=True)
    t0 = time.time()
    out = run.remote(CHAINS[chain])
    out["client_wall_minutes"] = (time.time() - t0) / 60
    print("RESULT", json.dumps(out), flush=True)
    os.makedirs(os.path.join(ITER_DIR, "gpu_log"), exist_ok=True)
    with open(os.path.join(ITER_DIR, "gpu_log", f"{chain}_{int(time.time())}.json"), "w") as f:
        json.dump(out, f, indent=1)
    if chain in ("eval", "smoke"):
        tag = "smoke_" if chain == "smoke" else ""
        dst = os.path.join(ITER_DIR, "smoke") if chain == "smoke" else ITER_DIR
        os.makedirs(dst, exist_ok=True)
        for name in ("stats.json", "small.pt"):
            local = os.path.join(dst, f"{tag}{name}")
            with open(local + ".part", "wb") as f:
                for chunk in vol.read_file(f"iter01/{tag}{name}"):
                    f.write(chunk)
            os.replace(local + ".part", local)
            print("downloaded", local, flush=True)
