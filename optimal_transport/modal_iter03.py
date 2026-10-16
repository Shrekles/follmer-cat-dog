"""Modal entry point for iteration 03 (CIFAR-only refinement). Run from the repository root.

    modal run optimal_transport/modal_iter03.py --chain smoke   # 20-step bs256 finetune + tiny prior/decode/memo/eval (dog)
    modal run optimal_transport/modal_iter03.py --chain D       # dog fine-tune -> prior -> decode cat->dog -> memo -> eval
    modal run optimal_transport/modal_iter03.py --chain C       # mirror for cat (dog->cat)
    (Dmid / Cmid: fallback on the mid snapshot, no training)

Each chain runs sequentially in ONE A10G container. GPU tasks live in tasks_iter03.py.
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
ITER_DIR = os.path.join(OT, "iterations/iter_03_cifar_refine")
REC_DIR = os.path.join(OT, "iterations/iter_02_purity")
GPU_DEADLINE = float(os.environ.get("GPU_DEADLINE_EPOCH", "0")) or None  # absolute epoch seconds


def _src(name):
    p = os.path.join(OT, name)
    if not modal.is_local():  # inside the container the local paths do not exist (and are not needed)
        return p
    if os.stat(p).st_flags & 0x40000000:  # SF_DATALESS: evicted by iCloud, reading would hang
        p = os.path.join(REC_DIR, "recovered_src", name)
    return p


app = modal.App("follmer-cat-dog-iter03")
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
    .add_local_file(os.path.join(OT, "tasks_iter03.py"), "/root/proj/optimal_transport/tasks_iter03.py")
)


def chains():
    D = GPU_DEADLINE
    return {
        "smoke": [("chain", dict(cls="dog", deadline=D, smoke=True, tag="smoke_"))],
        "D": [("chain", dict(cls="dog", deadline=D))],
        "C": [("chain", dict(cls="cat", deadline=D))],
        "Dmid": [("chain", dict(cls="dog", deadline=D, skip_train=True, use_mid=True))],
        "Cmid": [("chain", dict(cls="cat", deadline=D, skip_train=True, use_mid=True))],
    }


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=2 * 3600)
def run(chain: list):
    import sys
    sys.path.insert(0, "/root/proj/optimal_transport")
    import tasks_iter03
    t0 = time.time()
    results = []
    for task, kwargs in chain:
        print(f"== {task} {kwargs}", flush=True)
        vol.reload()
        fn = getattr(tasks_iter03, task)
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
    print("chain", chain, "on", GPU, "deadline", GPU_DEADLINE, flush=True)
    t0 = time.time()
    out = run.remote(chains()[chain])
    out["client_wall_minutes"] = (time.time() - t0) / 60
    print("RESULT", json.dumps(out), flush=True)
    os.makedirs(os.path.join(ITER_DIR, "gpu_log"), exist_ok=True)
    with open(os.path.join(ITER_DIR, "gpu_log", f"{chain}_{int(time.time())}.json"), "w") as f:
        json.dump(out, f, indent=1)
    dst = os.path.join(ITER_DIR, "smoke") if chain == "smoke" else ITER_DIR
    os.makedirs(dst, exist_ok=True)
    names = {"smoke": ["smoke_evalD", "smoke_memo_dog"], "D": ["evalD", "memo_dog"], "C": ["evalC", "memo_cat"],
             "Dmid": ["evalD_mid", "memo_dog_mid"], "Cmid": ["evalC_mid", "memo_cat_mid"]}[chain]
    for nm in names:
        for ext in ((".json",) if "memo" in nm else ("_stats.json", "_small.pt")):
            local = os.path.join(dst, nm + ext)
            if os.path.exists(local):
                print("exists, not overwriting", local, flush=True)
                continue
            try:
                with open(local + ".part", "wb") as f:
                    for chunk in vol.read_file(f"iter03/{nm}{ext}"):
                        f.write(chunk)
                os.replace(local + ".part", local)
                print("downloaded", local, flush=True)
            except Exception as e:  # noqa: BLE001
                print("download failed", nm + ext, repr(e), flush=True)
