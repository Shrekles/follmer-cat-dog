"""Modal entry point for iteration 06 (iter_06_repick_full: FT pair on the full test set, encoder-aware re-pick).
No training. Deploy and launch through launch_iter06.py (modal deploy + spawn).

    modal deploy optimal_transport/modal_iter06.py
    python3 optimal_transport/launch_iter06.py spawn smoke     # tiny ft_full + tiny repick (one container)
    python3 optimal_transport/launch_iter06.py spawn ft        # container A: FT on test 512-999 + 1,000-image eval
    python3 optimal_transport/launch_iter06.py spawn repick    # container B: stages 1-3 of the re-pick

One A10G container per call. GPU tasks live in tasks_iter06.py; outputs in /vol/iter06/.
Small outputs (JSON, PNG figures rendered in the container) are downloaded to iterations/iter_06_repick_full/.
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
ITER_DIR = os.path.join(OT, "iterations/iter_06_repick_full")

app = modal.App("follmer-cat-dog-iter06")
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
for _f in ("evaluate.py", "pf_ode.py", "tasks_iter.py", "tasks_iter02.py", "tasks_iter03.py", "tasks_iter04.py", "tasks_iter05.py",
           "tasks_iter06.py"):
    image = image.add_local_file(os.path.join(OT, _f), f"/root/proj/optimal_transport/{_f}")

CHAINS = {
    "smoke": [("chain_smoke", dict())],
    "smoke2": [("chain_smoke", dict(smoke_dir="smoke2"))],
    "ft": [("ft_full", dict())],
    "repick": [("repick", dict())],
}


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=int(1.0 * 3600))
def run(chain: list):
    import sys
    sys.path.insert(0, "/root/proj/optimal_transport")
    import tasks_iter06
    t0 = time.time()
    results = []
    for task, kwargs in chain:
        print(f"== {task} {kwargs}", flush=True)
        vol.reload()
        fn = getattr(tasks_iter06, task)
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
def main(chain: str = "smoke"):
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
