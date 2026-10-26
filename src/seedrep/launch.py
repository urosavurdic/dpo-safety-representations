"""Start one worker per GPU, each pinned to its device, and wait for them.

    python -m src.seedrep.launch --bank hf://USER/seedrep-bank --priorities P1 P2 \\
        --deadline-minutes 660

Each worker sees exactly one GPU (``CUDA_VISIBLE_DEVICES``), so the trainer
never falls back to DataParallel, which would double the effective batch and
change the recipe. Logs go to ``<work>/logs/worker_<gpu>.log``. Their tails are
printed every ``--every`` seconds, so a notebook cell running this shows
progress and keeps the session alive.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from src.seedrep import platform as plat


def tail(path: Path, n: int = 4) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return "(no output yet)"
    keep = [ln for ln in lines if ln.startswith(("worker:", "  [")) or "Error" in ln]
    return "\n".join(keep[-n:]) or "(starting)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--priorities", nargs="+", default=["P1"])
    ap.add_argument("--deadline-minutes", type=float, default=None)
    ap.add_argument("--every", type=float, default=300)
    ap.add_argument("--n-gpus", type=int, default=None)
    args = ap.parse_args()

    n_gpus = args.n_gpus if args.n_gpus is not None else plat.gpu_count()
    if n_gpus < 1:
        sys.exit("no GPU visible")
    logs = plat.paths().work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for gpu in range(min(n_gpus, 2)):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        cmd = [sys.executable, "-m", "src.seedrep.worker", "--bank", args.bank,
               "--gpu", str(gpu), "--n-gpus", str(min(n_gpus, 2)),
               "--priorities", *args.priorities]
        if args.deadline_minutes:
            cmd += ["--deadline-minutes", str(args.deadline_minutes)]
        log = open(logs / f"worker_{gpu}.log", "a", encoding="utf-8")
        procs.append((gpu, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        print(f"started worker {gpu}: {' '.join(cmd[2:])}")
        time.sleep(20)     # stagger the model loads

    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(args.every)
        for gpu, p, _ in procs:
            state = "running" if p.poll() is None else f"exit {p.returncode}"
            print(f"--- worker {gpu} ({state}) ---\n{tail(logs / f'worker_{gpu}.log')}", flush=True)
    codes = {gpu: p.returncode for gpu, p, _ in procs}
    for _, _, log in procs:
        log.close()
    print(f"workers finished: {codes}  (logs in {logs})")
    return 0 if all(c == 0 for c in codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
