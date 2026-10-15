"""Run the Experiment 3 GPU jobs on Modal and download their results.

    modal run optimal_transport/modal_ot.py                 # both jobs, in parallel
    modal run optimal_transport/modal_ot.py --task ddib     # just one

Needs the cat/dog models on the Modal volume (written there by separate_models/modal_train.py).
Results land in optimal_transport/results/{ddib,ot_flow}.pt. Run from the repository root.
"""
import os

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_SDE_COMMIT = "cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44"
GPU = os.environ.get("MODAL_GPU", "A10G")

app = modal.App("follmer-cat-dog-ot")
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
                   ignore=["__pycache__", "results"])
)


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=6 * 3600)
def run(task: str, kwargs: dict):
    import sys
    sys.path.insert(0, "/root/proj/optimal_transport")
    import tasks
    result = getattr(tasks, task)(**kwargs)
    vol.commit()
    return result


@app.local_entrypoint()
def main(task: str = "all", smoke: bool = False):
    have = {e.path for e in vol.listdir("data")}
    if "data/cifar10_test.pt" not in have:
        print("uploading data/cifar10_test.pt", flush=True)
        with vol.batch_upload() as up:
            up.put_file(os.path.join(ROOT, "data/cifar10_test.pt"), "data/cifar10_test.pt")
    names = ["ddib", "ot_flow"] if task == "all" else [task]
    print("starting", names, "on", GPU, flush=True)
    small = {"ddib": dict(n_real=4, n_model=2, steps=3),
             "ot_flow": dict(steps=5, n_eval=4, ode_steps=2, n_train=200)}
    for done in run.starmap([(n, small[n] if smoke else {}) for n in names], order_outputs=False):
        print("finished:", done, flush=True)
    os.makedirs(os.path.join(ROOT, "optimal_transport/results"), exist_ok=True)
    for name in names:
        local = os.path.join(ROOT, f"optimal_transport/results/{name}.pt")
        with open(local + ".part", "wb") as f:
            for chunk in vol.read_file(f"ot/{name}.pt"):
                f.write(chunk)
        os.replace(local + ".part", local)
        print("downloaded", local, flush=True)
