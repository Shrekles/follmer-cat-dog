"""Modal entry point for iteration 04 (from-scratch cat-only / dog-only models). Run from the repository root.

    modal run optimal_transport/modal_iter04.py --chain smoke_train   # 30 steps + tiny snapshot eval (cat)
    modal run optimal_transport/modal_iter04.py --chain smoke_eval    # tiny composed-ODE eval on the smoke models
    modal run --detach optimal_transport/modal_iter04.py --chain train_cat   # 95 min train + snapshot eval
    modal run --detach optimal_transport/modal_iter04.py --chain train_dog
    modal run --detach optimal_transport/modal_iter04.py --chain eval        # after both trainings finished

One A10G container per call. GPU tasks live in tasks_iter04.py; outputs in /vol/iter04/.
Small outputs (JSON, PNG figures rendered in the container) are downloaded to iterations/iter_04_from_scratch/.
"""
import inspect
import json
import os
import time

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OT = os.path.join(ROOT, "optimal_transport")
SCORE_SDE_COMMIT = "cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44"
GPU = "A10G"
ITER_DIR = os.path.join(OT, "iterations/iter_04_from_scratch")

app = modal.App("follmer-cat-dog-iter04")
vol = modal.Volume.from_name("follmer-cat-dog")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "ml_collections", "numpy<2", "scipy", "pillow")
    .run_commands(
        "git clone https://github.com/yang-song/score_sde_pytorch /root/proj/score_sde_pytorch",
        f"cd /root/proj/score_sde_pytorch && git checkout {SCORE_SDE_COMMIT}",
    )
    .add_local_file(os.path.join(ROOT, "common", "score_sde_backend.py"), "/root/proj/common/score_sde_backend.py")
)
for _f in ("evaluate.py", "pf_ode.py", "tasks_iter.py", "tasks_iter02.py", "tasks_iter03.py", "tasks_iter04.py"):
    image = image.add_local_file(os.path.join(OT, _f), f"/root/proj/optimal_transport/{_f}")

CHAINS = {
    "smoke_train": [("chain_train", dict(cls="cat", smoke=True, tag="smoke_"))],
    "smoke_eval": [("eval_composed", dict(smoke=True, tag="smoke_"))],
    "train_cat": [("chain_train", dict(cls="cat"))],
    "train_dog": [("chain_train", dict(cls="dog"))],
    "eval": [("eval_composed", dict())],
    "snapeval": [("snap_eval_from_volume", dict())],
    "resume_cat": [("chain_resume", dict(cls="cat", init="iter04/cat_s12129.pth"))],
    "resume_dog": [("chain_resume", dict(cls="dog", init="iter04/dog_s12386.pth"))],
}


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=int(2.4 * 3600))
def run(chain: list):
    import sys
    sys.path.insert(0, "/root/proj/optimal_transport")
    import tasks_iter04
    t0 = time.time()
    results = []
    for task, kwargs in chain:
        print(f"== {task} {kwargs}", flush=True)
        vol.reload()
        fn = getattr(tasks_iter04, task)
        if "vol" in inspect.signature(fn).parameters:
            kwargs = {**kwargs, "vol": vol}
        try:
            r = fn(**kwargs)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = {"task": task, "error": repr(e), "minutes": 0}
        vol.commit()
        results.append(r)
        print(f"== done {task}: {r.get('minutes', 0):.1f} min", flush=True)
    return {"container_minutes": (time.time() - t0) / 60, "results": results}


@app.local_entrypoint()
def main(chain: str = "smoke_train"):
    print("chain", chain, "on", GPU, flush=True)
    t0 = time.time()
    out = run.remote(CHAINS[chain])
    out["client_wall_minutes"] = (time.time() - t0) / 60
    print("RESULT", json.dumps(out), flush=True)
    os.makedirs(os.path.join(ITER_DIR, "gpu_log"), exist_ok=True)
    with open(os.path.join(ITER_DIR, "gpu_log", f"{chain}_{int(time.time())}.json"), "w") as f:
        json.dump(out, f, indent=1)
    dst = os.path.join(ITER_DIR, "smoke") if chain.startswith("smoke") else ITER_DIR
    os.makedirs(dst, exist_ok=True)
    for r in out["results"]:
        for rel in r.get("downloads", []):
            local = os.path.join(dst, os.path.basename(rel))
            if os.path.exists(local):
                print("exists, not overwriting", local, flush=True)
                continue
            try:
                with open(local + ".part", "wb") as f:
                    for chunk in vol.read_file(rel):
                        f.write(chunk)
                os.replace(local + ".part", local)
                print("downloaded", local, os.path.getsize(local), flush=True)
            except Exception as e:  # noqa: BLE001
                print("download failed", rel, repr(e), flush=True)
