"""Modal entry point for iteration 02 (purity). Run from the repository root.

    modal run optimal_transport/modal_iter02.py --chain smoke   # tiny versions of every new task
    modal run optimal_transport/modal_iter02.py --chain A       # prior samples, latents, guided decodes, eval + KID refs
    modal run optimal_transport/modal_iter02.py --chain B       # CINIC-10 download, dog fine-tune, decode, eval

Each chain runs sequentially in ONE A10G container. GPU tasks live in tasks_iter02.py.
Only the needed source files are shipped (several repo files are iCloud-evicted on this Mac; evaluate.py and
pf_ode.py fall back to byte-identical copies in iterations/iter_02_purity/recovered_src/ when evicted).
"""
import inspect
import json
import os
import time

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OT = os.path.join(ROOT, "optimal_transport")
SCORE_SDE_COMMIT = "cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44"
GPU = os.environ.get("MODAL_GPU", "A10G")
ITER_DIR = os.path.join(OT, "iterations/iter_02_purity")
GPU_DEADLINE = float(os.environ.get("GPU_DEADLINE_EPOCH", "0")) or None  # absolute epoch seconds


def _src(name):
    p = os.path.join(OT, name)
    if not modal.is_local():  # inside the container the local paths do not exist (and are not needed)
        return p
    if os.stat(p).st_flags & 0x40000000:  # SF_DATALESS: evicted by iCloud, reading would hang
        p = os.path.join(ITER_DIR, "recovered_src", name)
    return p


app = modal.App("follmer-cat-dog-iter02")
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
    .add_local_file(_src("evaluate.py"), "/root/proj/optimal_transport/evaluate.py")
    .add_local_file(_src("pf_ode.py"), "/root/proj/optimal_transport/pf_ode.py")
    .add_local_file(os.path.join(OT, "tasks_iter.py"), "/root/proj/optimal_transport/tasks_iter.py")
    .add_local_file(os.path.join(OT, "tasks_iter02.py"), "/root/proj/optimal_transport/tasks_iter02.py")
)

SM = dict(steps=10, n=4, bs=4)
DIR_RUNS = lambda tag: [  # noqa: E731
    {"name": "prior_cat", "kind": "prior", "path": f"/vol/iter02/{tag}prior_cat.pt", "cls": 3},
    {"name": "prior_dog", "kind": "prior", "path": f"/vol/iter02/{tag}prior_dog.pt", "cls": 5},
    {"name": "cat_to_dog_w0.5", "kind": "transport", "direction": "cat_to_dog", "path": f"/vol/iter02/{tag}transport_cat_to_dog_w0.5.pt"},
    {"name": "cat_to_dog_w1.5", "kind": "transport", "direction": "cat_to_dog", "path": f"/vol/iter02/{tag}transport_cat_to_dog_w1.5.pt"},
    {"name": "dog_to_cat_w1.0", "kind": "transport", "direction": "dog_to_cat", "path": f"/vol/iter02/{tag}transport_dog_to_cat_w1.0.pt"},
]
BASELINE_RUNS = [
    {"name": "baseline_cat_to_dog_iter01", "kind": "transport", "direction": "cat_to_dog", "path": "/vol/iter01/transport_cat_to_dog_B.pt"},
    {"name": "baseline_dog_to_cat_iter01", "kind": "transport", "direction": "dog_to_cat", "path": "/vol/iter01/transport_dog_to_cat_A.pt"},
]


def chains():
    D = GPU_DEADLINE
    return {
        "smoke": [
            ("prior_samples", dict(models={"cat": "out/cat.pth", "dog": "out/dog.pth"}, tag="smoke_", **SM)),
            ("encode_latents", dict(direction="cat_to_dog", tag="smoke_", **SM)),
            ("guided_decode", dict(direction="cat_to_dog", w=0.5, tag="smoke_", steps=10, bs=4)),
            ("chain_b", dict(deadline=D, smoke=True, tag="smoke_")),
            ("eval_runs", dict(runs=DIR_RUNS("smoke_")[:3], out_name="smoke_evalA", n_boot=10)),
        ],
        "A": [
            ("prior_samples", dict(models={"cat": "out/cat.pth", "dog": "out/dog.pth"}, deadline=D)),
            ("encode_latents", dict(direction="cat_to_dog")),
            ("guided_decode", dict(direction="cat_to_dog", w=0.5, deadline=D)),
            ("guided_decode", dict(direction="cat_to_dog", w=1.5, deadline=D)),
            ("encode_latents", dict(direction="dog_to_cat")),
            ("guided_decode", dict(direction="dog_to_cat", w=1.0, deadline=D)),
            ("eval_runs", dict(runs=DIR_RUNS("") + BASELINE_RUNS, out_name="evalA")),
        ],
        "B": [("chain_b", dict(deadline=D))],
        # relaunch of A after the first A/B containers were cancelled at 13:45 (prior samples already saved)
        "A2": [
            ("encode_latents", dict(direction="cat_to_dog")),
            ("guided_decode", dict(direction="cat_to_dog", w=0.5, deadline=D)),
            ("guided_decode", dict(direction="cat_to_dog", w=1.5, deadline=D)),
            ("encode_latents", dict(direction="dog_to_cat")),
            ("guided_decode", dict(direction="dog_to_cat", w=1.0, deadline=D)),
            ("eval_runs", dict(runs=DIR_RUNS("") + BASELINE_RUNS, out_name="evalA")),
        ],
    }


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=2 * 3600)
def run(chain: list):
    import sys
    sys.path.insert(0, "/root/proj/optimal_transport")
    import tasks_iter02
    t0 = time.time()
    results = []
    for task, kwargs in chain:
        print(f"== {task} {kwargs}", flush=True)
        vol.reload()
        fn = getattr(tasks_iter02, task)
        if "vol" in inspect.signature(fn).parameters:
            kwargs = {**kwargs, "vol": vol}
        try:
            r = fn(**kwargs)
        except Exception as e:  # noqa: BLE001  keep going with the rest of the chain
            import traceback
            traceback.print_exc()
            r = {"task": task, "error": repr(e), "minutes": 0}
        vol.commit()
        results.append(r)
        print(f"== done {task}: {r.get('minutes', 0):.1f} min", flush=True)
    return {"container_minutes": (time.time() - t0) / 60, "results": results}


@app.local_entrypoint()
def main(chain: str = "smoke"):
    print("chain", chain, "on", GPU, "deadline", GPU_DEADLINE, flush=True)
    t0 = time.time()
    out = run.remote(chains()[chain])
    out["client_wall_minutes"] = (time.time() - t0) / 60
    print("RESULT", json.dumps(out), flush=True)
    os.makedirs(os.path.join(ITER_DIR, "gpu_log"), exist_ok=True)
    with open(os.path.join(ITER_DIR, "gpu_log", f"{chain}_{int(time.time())}.json"), "w") as f:
        json.dump(out, f, indent=1)
    tag = "smoke_" if chain == "smoke" else ""
    dst = os.path.join(ITER_DIR, "smoke") if chain == "smoke" else ITER_DIR
    os.makedirs(dst, exist_ok=True)
    names = {"smoke": ["smoke_evalA", "smoke_evalB"], "A": ["evalA"], "A2": ["evalA"], "B": ["evalB"]}[chain]
    for nm in names:
        for ext in ("_stats.json", "_small.pt"):
            local = os.path.join(dst, nm + ext)
            try:
                with open(local + ".part", "wb") as f:
                    for chunk in vol.read_file(f"iter02/{nm}{ext}"):
                        f.write(chunk)
                os.replace(local + ".part", local)
                print("downloaded", local, flush=True)
            except Exception as e:  # noqa: BLE001
                print("download failed", nm + ext, repr(e), flush=True)
