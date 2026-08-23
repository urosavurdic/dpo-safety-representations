"""
Orchestrates eval_steering_v2.py across all 8 non-M0 training stages
(M1, M2, M3, M3_direct, M1_alt, M2_alt, M3_alt, M3_direct_alt), quadrants
A+D, held-out-behavioral split -- the "8-stage steering notebook" referenced
in CLAUDE.md/README's Next Steps item 1. No such notebook is tracked in
git (colab_*.ipynb files are gitignored by convention -- see CLAUDE.md
"Repo structure"), so this reconstructs the same loop pattern the Colab
analysis notebook uses for causal ablation (STAGES_FOR_CAUSAL) as a local,
resumable, GPU-required CLI script instead, matching the existing
CPU-only src/reproduce.py's manifest/never-overwrite conventions.

Deliberately has NO torch/transformers/peft import at module level (unlike
eval_steering_v2.py itself) -- this file only shells out to
`python -m src.analysis.eval_steering_v2` as a subprocess per stage, so it
stays importable/testable in a torch-less CPU sandbox (matches the
project's "Testing status" convention in CLAUDE.md: CPU-pure logic should
be collectible even without the real ML stack installed). The tiny
eval-set loading/validation logic below is deliberately duplicated rather
than imported from eval_extract_activations.py / eval_steering_v2.py, for
the same reason those two already duplicate load_controlled_eval() from
each other instead of sharing it across a torch-importing module boundary.

Preconditions this checks BEFORE spending any GPU time (does not run
anything if these fail -- see README/CLAUDE.md Next Steps item 1's "the
expensive GPU steering pass happens once, not twice" concern):
  1. Every quadrant-A/D row in the live controlled_eval.jsonl has a "split"
     key assigned (proves assign_direction_split has actually run over the
     CURRENT eval set, not a stale pre-split copy).
  2. For each stage being run, results/activations/{stage}_metadata.json
     exists and matches the live eval set exactly (same equality check
     eval_extract_activations.py's own resumability logic uses) -- catches
     "you rebuilt the eval set but forgot to re-extract activations for
     this stage" before wasting a GPU steering run on a stale direction.
  3. results/refusal_direction/{stage}_direction.npy exists (built from
     those same activations via `python -m src.reproduce direction` /
     `eval_refusal_direction.py`).

Usage:
    python -m src.analysis.run_full_steering --dry-run          # plan only
    python -m src.analysis.run_full_steering                    # real run
    python -m src.analysis.run_full_steering --stages M3 M3_alt # subset
    python -m src.analysis.run_full_steering --force             # rerun stages whose output already exists
"""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Duplicated from src.training.model.STAGE_ADAPTER_CHAINS on purpose (that
# module imports peft/transformers at load time) -- keep in sync by hand if
# a stage is ever added/removed there. M0 excluded: it has no DPO/SFT-safety
# training yet, steering it isn't part of the sufficiency test (Finding 4
# only ever covered post-M0 stages).
ALL_STAGES = ["M1", "M2", "M3", "M3_direct", "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt"]

DEFAULT_LAYERS = [24]
DEFAULT_ALPHA_SOURCE = "quadrant_a_projection"
DEFAULT_ALPHA_COEFFICIENT = 1.0
DEFAULT_QUADRANTS = ["A", "D"]

EVAL_SET_PATH = "data/processed/controlled_eval.jsonl"
ACTIVATIONS_DIR = Path("results/activations")
DIRECTIONS_DIR = Path("results/refusal_direction")
RAW_DIR = Path("results/raw")
MANIFEST_DIR = Path("results/manifests")


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def load_controlled_eval(path=EVAL_SET_PATH):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_split_assigned(eval_rows):
    """Returns (ok, message). ok is False if any A/D row is missing a
    'split' key -- mirrors filter_to_held_out_behavioral_split's own
    "rows with no split key are dropped, not silently included" stance,
    but surfaces it as a precondition failure here instead of a silent
    filter, since running with a partially-split eval set would produce a
    quietly-undersized steering run rather than an obvious error."""
    ad_rows = [r for r in eval_rows if r["quadrant"] in ("A", "D")]
    missing = [r for r in ad_rows if "split" not in r]
    if missing:
        return False, (
            f"{len(missing)}/{len(ad_rows)} quadrant A/D rows have no 'split' key -- "
            f"the live {EVAL_SET_PATH} predates assign_direction_split, or the split "
            "hasn't been (re-)applied since the eval set was last rebuilt. Run "
            "`python -m src.data_pipeline.build_eval_set` first."
        )
    n_est = sum(1 for r in ad_rows if r["split"] == "direction_estimation")
    n_held = sum(1 for r in ad_rows if r["split"] == "held_out_behavioral")
    from collections import Counter
    quadrant_counts = Counter(r["quadrant"] for r in eval_rows)
    return True, (
        f"split OK: {n_est} direction_estimation + {n_held} held_out_behavioral "
        f"across quadrants A/D. Live quadrant counts: {dict(quadrant_counts)} "
        "(sanity-check these against what you expect before trusting the run -- "
        "this script doesn't hardcode a specific target count, since that's the "
        "data pipeline's concern, not this one's)."
    )


def activation_metadata_matches(stage, eval_rows):
    """True only if results/activations/{stage}_metadata.json exists and its
    saved (prompt, quadrant, source, split) rows exactly match the live eval
    set, same order -- the same equality check
    eval_extract_activations.eval_set_matches_saved_metadata uses, kept as a
    local copy for the reason explained in the module docstring."""
    meta_path = ACTIVATIONS_DIR / f"{stage}_metadata.json"
    if not meta_path.exists():
        return False, f"missing {meta_path}"
    with open(meta_path, encoding="utf-8") as f:
        saved = json.load(f)
    current = [{"prompt": r["prompt"], "quadrant": r["quadrant"], "source": r["source"],
                "split": r.get("split")} for r in eval_rows]
    if saved != current:
        return False, (
            f"{meta_path} does not match the live eval set (stale -- re-run "
            "`python -m src.analysis.eval_extract_activations` for this stage)"
        )
    return True, f"{meta_path} matches the live eval set"


def direction_exists(stage):
    path = DIRECTIONS_DIR / f"{stage}_direction.npy"
    return path.exists(), path


def default_tag(stage, layers, alpha_source, alpha_coefficient, quadrants):
    """Must stay byte-identical to eval_steering_v2.default_tag's output --
    duplicated (not imported, see module docstring) so this script can
    predict a stage's output path/skip status without importing torch."""
    layers_str = "-".join(str(l) for l in sorted(set(layers)))
    quad_str = "".join(quadrants)
    coef_str = f"{alpha_coefficient:g}".replace(".", "p")
    return f"{stage}_L{layers_str}_{alpha_source}_coef{coef_str}_Q{quad_str}"


def output_path_for(tag):
    return RAW_DIR / f"steering_v2_{tag}.json"


def build_command(stage, layers, alpha_source, alpha_coefficient, quadrants, tag, force):
    cmd = [
        "python", "-m", "src.analysis.eval_steering_v2",
        "--stage", stage,
        "--layers", *[str(l) for l in layers],
        "--alpha-source", alpha_source,
        "--alpha-coefficient", str(alpha_coefficient),
        "--quadrants", *quadrants,
        "--tag", tag,
    ]
    if force:
        cmd.append("--overwrite")
    return cmd


def plan_run(stages, layers, alpha_source, alpha_coefficient, quadrants, force):
    """Pure planning step (no subprocess, no GPU) -- returns a list of dicts
    describing what WOULD happen for each stage. Used by both --dry-run and
    the real run (so the printed plan and the actual execution can never
    silently diverge)."""
    eval_rows = load_controlled_eval()
    split_ok, split_msg = check_split_assigned(eval_rows)

    plan = []
    for stage in stages:
        tag = default_tag(stage, layers, alpha_source, alpha_coefficient, quadrants)
        out_path = output_path_for(tag)
        act_ok, act_msg = activation_metadata_matches(stage, eval_rows)
        dir_ok, dir_path = direction_exists(stage)

        blockers = []
        if not split_ok:
            blockers.append(split_msg)
        if not act_ok:
            blockers.append(act_msg)
        if not dir_ok:
            blockers.append(f"missing {dir_path} -- run `python -m src.reproduce --components direction` "
                             "(after activations exist) to build it")

        if blockers:
            status = "blocked"
        elif out_path.exists() and not force:
            status = "skip_already_done"
        else:
            status = "run"

        plan.append({
            "stage": stage, "tag": tag, "output_path": str(out_path),
            "status": status, "blockers": blockers,
            "command": build_command(stage, layers, alpha_source, alpha_coefficient, quadrants, tag, force),
        })
    return plan, split_msg


def print_plan(plan, split_msg):
    print(f"Eval set check: {split_msg}\n")
    for item in plan:
        print(f"[{item['status']:18s}] {item['stage']:15s} -> {item['output_path']}")
        for b in item["blockers"]:
            print(f"    BLOCKED: {b}")
    n_run = sum(1 for i in plan if i["status"] == "run")
    n_skip = sum(1 for i in plan if i["status"] == "skip_already_done")
    n_blocked = sum(1 for i in plan if i["status"] == "blocked")
    print(f"\n{n_run} to run, {n_skip} already done (skipped), {n_blocked} blocked.")


def run_plan(plan):
    """Executes each 'run' item in order, stage-by-stage, stopping to report
    (not crashing the whole batch) on a per-stage failure -- later stages
    still get attempted, since stages are fully independent (each loads its
    own model/direction). Returns the per-stage results for the manifest."""
    results = []
    for item in plan:
        if item["status"] != "run":
            results.append({**item, "ran": False})
            continue
        print(f"\n=== {item['stage']}: {' '.join(item['command'])} ===")
        result = subprocess.run(item["command"])
        ok = result.returncode == 0
        results.append({**item, "ran": True, "returncode": result.returncode, "succeeded": ok})
        if not ok:
            print(f"  {item['stage']} FAILED (exit {result.returncode}) -- continuing to the next stage "
                  "(stages are independent, no reason to abort the whole batch on one failure).")
    return results


def write_manifest(results, args):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    manifest = {
        "component": "run_full_steering",
        "git_commit": get_git_commit(),
        "timestamp_utc": timestamp.isoformat(),
        "args": vars(args),
        "results": results,
    }
    path = MANIFEST_DIR / f"full_steering_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES,
                         help=f"Subset of stages to run (default: all 8 -- {ALL_STAGES})")
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--alpha-source", default=DEFAULT_ALPHA_SOURCE,
                         choices=["quadrant_a_projection", "fixed"])
    parser.add_argument("--alpha-coefficient", type=float, default=DEFAULT_ALPHA_COEFFICIENT)
    parser.add_argument("--quadrants", nargs="+", default=DEFAULT_QUADRANTS, choices=["A", "B", "C", "D"])
    parser.add_argument("--force", action="store_true",
                         help="Rerun stages whose output file already exists (passes --overwrite through)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit -- runs nothing")
    args = parser.parse_args()

    plan, split_msg = plan_run(args.stages, args.layers, args.alpha_source,
                                args.alpha_coefficient, args.quadrants, args.force)
    print_plan(plan, split_msg)

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    any_blocked = any(item["status"] == "blocked" for item in plan)
    if any_blocked:
        print("\nAt least one stage is blocked (see above). Proceeding with the stages that "
              "aren't -- rerun with --dry-run after fixing blockers to confirm before spending GPU time.")

    results = run_plan(plan)
    manifest_path = write_manifest(results, args)

    print(f"\nManifest written to {manifest_path}")
    print("\nPer-stage stats commands (once results exist):")
    for item in results:
        if item["status"] == "run" and item.get("succeeded"):
            print(f"  python -m src.analysis.summarize_steering --file {item['output_path']}")
            for q in args.quadrants:
                cat = "refusal"  # induced-refusal is the relevant test for both A (side-effect check) and D (over-refusal test)
                print(f"  python -m src.analysis.mcnemar_steering --file {item['output_path']} "
                      f"--quadrant {q} --category {cat}")


if __name__ == "__main__":
    main()
