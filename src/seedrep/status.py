"""Where does the run stand? Read the bank, print every job, say what is next.

    python -m src.seedrep.status --bank hf://USER/seedrep-bank [--priorities P1 P2 P3 P4]

Works from any machine, with no GPU. A job is ``done`` (marker present),
``partial`` (outputs started: a training run's latest resume checkpoint,
generation shards, judged chunks), ``ready`` (dependencies done) or
``waiting``.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

from src.seedrep.jobs import build_jobs


def describe(bank, priorities=("P1", "P2", "P3", "P4")) -> list[dict]:
    bank.refresh()
    files = bank.list()
    done = {f[len("done/"):-len(".json")] for f in files if f.startswith("done/")}
    rows = []
    for job in build_jobs(tuple(priorities)):
        if job.id in done:
            state, detail = "done", ""
        else:
            detail = ""
            if job.kind == "train":
                rid = job.params["run_id"]
                ckpts = sorted({int(m.group(1)) for f in files
                                if (m := re.match(rf"resume/{rid}/checkpoint-(\d+)/", f))})
                doses = [f for f in files if f.startswith(f"adapters/{rid}/step_")
                         and f.endswith("adapter_model.safetensors")]
                if ckpts or doses:
                    detail = f"resume at step {ckpts[-1] if ckpts else 0}, {len(doses)} dose adapters"
            elif job.kind in ("gen", "inject"):
                unit = job.params.get("run_id") or job.params.get("unit")
                n = sum(1 for f in files if f.startswith(f"gen/{unit}/shard_"))
                detail = f"{n} shards banked" if n else ""
            elif job.kind == "judge":
                n = sum(1 for f in files if f.startswith(f"judge/{job.params['unit']}/chunk_"))
                detail = f"{n} chunks banked" if n else ""
            ready = all(d in done for d in job.deps)
            state = "partial" if detail else ("ready" if ready else "waiting")
        rows.append({"id": job.id, "gpu": job.gpu, "priority": job.priority, "state": state,
                     "detail": detail, "minutes": job.minutes})
    return rows


def main() -> int:
    from src.seedrep import platform as plat
    from src.seedrep.bank import open_bank

    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--priorities", nargs="+", default=["P1", "P2", "P3", "P4"])
    ap.add_argument("--all", action="store_true", help="also list finished jobs")
    args = ap.parse_args()
    rows = describe(open_bank(args.bank, token=plat.secret("HF_TOKEN")), args.priorities)
    for prio in args.priorities:
        part = [r for r in rows if r["priority"] == prio]
        if not part:
            continue
        counts = Counter(r["state"] for r in part)
        left = sum(r["minutes"] for r in part if r["state"] != "done")
        print(f"\n{prio}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
              + f"  (~{left:.0f} GPU-min left)")
        for r in part:
            if args.all or r["state"] != "done":
                print(f"  [{r['state']:7}] gpu{r['gpu']} {r['id']:45} {r['detail']}")
    todo = [r for r in rows if r["state"] != "done"]
    if not todo:
        print("\nEverything is done. Next: python -m src.seedrep.analyze --bank ...")
    else:
        first = todo[0]
        print(f"\nNext: run the worker cells; the first unfinished job is {first['id']} "
              f"({first['state']}). Rerunning is always safe: finished work is skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
