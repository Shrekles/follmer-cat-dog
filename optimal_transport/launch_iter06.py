"""Launch / collect iteration-06 jobs on the DEPLOYED app, so a job keeps running if the local client dies.

    modal deploy optimal_transport/modal_iter06.py                 # (re)deploy after any code change
    python3 optimal_transport/launch_iter06.py spawn <chain>        # prints and records the call id
    python3 optimal_transport/launch_iter06.py fetch <chain> <id>   # waits for the result, downloads small files
"""
import json
import os
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ITER_DIR = os.path.join(HERE, "iterations/iter_06_repick_full")
APP = "follmer-cat-dog-iter06"


def chains():
    sys.path.insert(0, HERE)
    from modal_iter06 import CHAINS
    return CHAINS


def spawn(chain):
    fn = modal.Function.from_name(APP, "run")
    call = fn.spawn(chains()[chain])
    os.makedirs(os.path.join(ITER_DIR, "gpu_log"), exist_ok=True)
    with open(os.path.join(ITER_DIR, "gpu_log", "calls.txt"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {chain} {call.object_id}\n")
    print("spawned", chain, call.object_id, flush=True)


def fetch(chain, call_id):
    call = modal.FunctionCall.from_id(call_id)
    while True:
        try:
            out = call.get(timeout=60)
            break
        except TimeoutError:
            print(time.strftime("%H:%M:%S"), "still running", flush=True)
    print("RESULT", json.dumps(out), flush=True)
    with open(os.path.join(ITER_DIR, "gpu_log", f"{chain}_{int(time.time())}.json"), "w") as f:
        json.dump(out, f, indent=1)
    vol = modal.Volume.from_name("follmer-cat-dog")
    dst = os.path.join(ITER_DIR, "smoke") if chain.startswith("smoke") else ITER_DIR
    os.makedirs(dst, exist_ok=True)
    for r in out["results"]:
        for rel in r.get("downloads", []):
            local = os.path.join(dst, os.path.basename(rel))
            if os.path.exists(local):
                print("exists, not overwriting", local, flush=True)
                continue
            with open(local + ".part", "wb") as f:
                for chunk in vol.read_file(rel):
                    f.write(chunk)
            os.replace(local + ".part", local)
            print("downloaded", local, os.path.getsize(local), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "spawn":
        spawn(sys.argv[2])
    else:
        fetch(sys.argv[2], sys.argv[3])
