"""Stage-1 orchestration: plan, gate on preconditions, shell out per unit.

Mirrors src/analysis/run_full_steering.py: no torch at module import, a pure
``plan_run`` that is fully unit-testable, one subprocess per unit, and a JSON
manifest. Keeping planning pure is what lets ``--dry-run`` tell you exactly
what would happen -- including *why* something is blocked -- without a GPU.

The activation precondition is deliberately unforgiving. If the four
654-row ``_final`` arrays are not present and bound to the frozen benchmark
and split hashes, every unit reports ``blocked``. This extension never
re-extracts activations and never falls back to the stale 370-era arrays.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from src.analysis.crossbranch.branches import (
    BRANCHES,
    COEFFICIENTS,
    INJECT_LAYER,
    MODEL,
    P0,
    P0_CONDITIONS,
    VECTOR,
    direction_tag,
    get,
    planned_units,
    stages_needed,
)
from src.analysis.crossbranch.delta import DELTAS_DIR, artifact_path
from src.analysis.crossbranch.worker import output_path
from src.v2_io import load_run_inputs

MANIFEST_DIR = Path("results/crossbranch/manifests")

RUN = "run"
SKIP = "skip_already_done"
BLOCKED = "blocked"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def check_split_assigned(rows) -> tuple[bool, str]:
    """Every A/D row must carry a split label.

    Copied rather than imported (15 lines, and run_full_steering already
    duplicates it) so this module keeps a minimal import surface.
    """
    missing = [
        r.get("record_id")
        for r in rows
        if r.get("quadrant") in {"A", "D"} and not r.get("split")
    ]
    if missing:
        return False, (
            f"{len(missing)} quadrant-A/D rows have no split label "
            f"(e.g. {missing[:3]}). Rebuild the eval set."
        )
    return True, "split assigned on all A/D rows"


def check_activations(ctx, source_branch: str, target_branch: str) -> tuple[bool, str]:
    from src.analysis.v2_pipeline import activation_paths, activations_bound

    unbound = []
    for stage in stages_needed(source_branch, target_branch):
        if activations_bound(ctx, stage):
            continue
        final_path, _pooled, metadata_path, binding_path = activation_paths(ctx, stage)
        if not final_path.exists():
            unbound.append(f"{stage}: {final_path.name} missing")
        elif not binding_path.exists():
            unbound.append(
                f"{stage}: no binding sidecar (legacy extraction); "
                "metadata must match the frozen 654-row benchmark to be adopted"
            )
        else:
            unbound.append(f"{stage}: bound to a different benchmark/split, or wrong row count")

    if unbound:
        return False, (
            "Activations not bound to the frozen benchmark:\n      "
            + "\n      ".join(unbound)
            + "\n      Run the frozen v2 extract stage "
            "(python -m src.analysis.v2_pipeline run) first. This extension "
            "never extracts and never uses stale 370-era arrays."
        )
    return True, "all four stages bound to the frozen benchmark"


def check_delta_artifact(condition: str, layer: int, deltas_dir) -> tuple[bool, str]:
    cond = get(condition)
    if cond.kind == MODEL:
        # Model conditions inject nothing; requiring an .npz for them would be
        # a bug, not a safeguard.
        return True, "model condition; no delta artifact required"
    path = artifact_path(cond.artifact, layer, deltas_dir)
    if not path.exists():
        return False, (
            f"missing {path}; run "
            "`python -m src.analysis.crossbranch.delta` (or --assemble-first)"
        )
    return True, f"{path.name} present"


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_run(
    ctx,
    conditions: list[str],
    coefficients: tuple[float, ...],
    source_branch: str,
    target_branch: str,
    *,
    layer: int = INJECT_LAYER,
    deltas_dir=DELTAS_DIR,
    out_dir=None,
    force: bool = False,
    allow_stage2: bool = False,
) -> tuple[list[dict], list[str]]:
    messages: list[str] = []
    tag = direction_tag(source_branch, target_branch)

    split_ok, split_msg = check_split_assigned(ctx.rows)
    act_ok, act_msg = check_activations(ctx, source_branch, target_branch)
    messages.append(f"split:       {'OK ' if split_ok else 'BLOCKED '}{split_msg}")
    messages.append(f"activations: {'OK ' if act_ok else 'BLOCKED '}{act_msg}")

    plan: list[dict] = []
    for condition, coef in planned_units(conditions, coefficients):
        cond = get(condition)
        blockers: list[str] = []
        if not split_ok:
            blockers.append("split not assigned")
        if not act_ok:
            blockers.append("activations not bound")
        if cond.stage_gate != P0 and not allow_stage2:
            blockers.append(f"gated '{cond.stage_gate}' (pass --allow-stage2)")
        art_ok, art_msg = check_delta_artifact(condition, layer, deltas_dir)
        if not art_ok:
            blockers.append(art_msg)

        path = output_path(tag, condition, coef, out_dir) if out_dir else output_path(
            tag, condition, coef
        )
        if blockers:
            status = BLOCKED
        elif path.exists() and not force:
            status = SKIP
        else:
            status = RUN

        plan.append(
            {
                "condition": condition,
                "coef": coef,
                "kind": cond.kind,
                "stage_gate": cond.stage_gate,
                "output": str(path),
                "status": status,
                "blockers": blockers,
            }
        )
    return plan, messages


def build_command(
    condition: str,
    coef: float | None,
    source_branch: str,
    target_branch: str,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable, "-m", "src.analysis.crossbranch.worker",
        "--condition", condition,
        "--source-branch", source_branch,
        "--target-branch", target_branch,
    ]
    if coef is not None:
        cmd += ["--coef", f"{coef:g}"]
    return cmd + list(extra or [])


def print_plan(plan: list[dict], messages: list[str]) -> None:
    print("Preconditions:")
    for m in messages:
        print(f"  {m}")
    counts: dict[str, int] = {}
    print("\nUnits:")
    for item in plan:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
        coef = "  n/a" if item["coef"] is None else f"{item['coef']:5g}"
        print(f"  [{item['status']:>17}] {item['condition']:<26} coef={coef}")
        for b in item["blockers"]:
            for line in str(b).splitlines():
                print(f"      - {line}")
    print("\nTotals: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"Planned units: {len(plan)}")


def run_plan(plan, source_branch, target_branch, extra=None) -> list[dict]:
    results = []
    for item in plan:
        if item["status"] != RUN:
            results.append({**item, "returncode": None})
            continue
        cmd = build_command(
            item["condition"], item["coef"], source_branch, target_branch, extra
        )
        print(f"\n$ {' '.join(cmd)}")
        proc = subprocess.run(cmd)
        results.append({**item, "returncode": proc.returncode})
        if proc.returncode != 0:
            print(f"  FAILED (rc={proc.returncode}); continuing with the next unit.")
    return results


def write_manifest(results, args, ctx, tag) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = MANIFEST_DIR / f"crossbranch_{tag}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "component": "crossbranch_stage1",
                "git_commit": get_git_commit(),
                "timestamp_utc": stamp,
                "benchmark_sha256": ctx.benchmark_sha,
                "split_manifest_sha256": ctx.split_sha,
                "direction": tag,
                "args": {k: v for k, v in vars(args).items()},
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-branch Stage-1 runner.")
    p.add_argument("--conditions", nargs="+", default=list(P0_CONDITIONS))
    p.add_argument("--coefficients", nargs="+", type=float, default=list(COEFFICIENTS))
    p.add_argument("--source-branch", default="A", choices=sorted(BRANCHES))
    p.add_argument("--target-branch", default="B", choices=sorted(BRANCHES))
    p.add_argument("--layer", type=int, default=INJECT_LAYER)
    p.add_argument("--deltas-dir", default=str(DELTAS_DIR))
    p.add_argument("--eval-set", default=None)
    p.add_argument("--benchmark-sha256", default=None)
    p.add_argument("--split-manifest", default="logs/direction_split_manifest.json")
    p.add_argument("--assemble-first", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-stage2", action="store_true")
    args = p.parse_args()

    from src.analysis.v2_pipeline import build_context

    benchmark_path, benchmark_sha, split_path, split_sha = load_run_inputs(
        args.eval_set, args.benchmark_sha256, args.split_manifest
    )
    print(f"benchmark: {benchmark_path} ({benchmark_sha[:12]}...)")
    print(f"split:     {split_path} ({split_sha[:12]}...)\n")

    ctx = build_context(SimpleNamespace(**vars(args)))
    tag = direction_tag(args.source_branch, args.target_branch)

    if args.assemble_first:
        cmd = [
            sys.executable, "-m", "src.analysis.crossbranch.delta",
            "--source-branch", args.source_branch,
            "--target-branch", args.target_branch,
            "--layer", str(args.layer),
            "--out-dir", args.deltas_dir,
        ]
        print(f"$ {' '.join(cmd)}")
        if subprocess.run(cmd).returncode != 0:
            raise SystemExit("delta assembly failed; nothing was run.")

    plan, messages = plan_run(
        ctx,
        args.conditions,
        tuple(args.coefficients),
        args.source_branch,
        args.target_branch,
        layer=args.layer,
        deltas_dir=args.deltas_dir,
        force=args.force,
        allow_stage2=args.allow_stage2,
    )
    print_plan(plan, messages)

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    runnable = [i for i in plan if i["status"] == RUN]
    if not runnable:
        print("\nNothing to run.")
        return

    extra = ["--layer", str(args.layer), "--deltas-dir", args.deltas_dir]
    results = run_plan(plan, args.source_branch, args.target_branch, extra)
    print(f"\nManifest: {write_manifest(results, args, ctx, tag)}")


if __name__ == "__main__":
    main()
