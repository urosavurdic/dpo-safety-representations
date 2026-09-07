"""Repair condition labels in tagged causal-ablation files written before the
``stage_causal`` relabel fix.

THE BUG (fixed at source in v2_pipeline.stage_causal; this repairs the files
already on disk): ``--all-ad-sensitivity`` builds each condition's shard-store
unit as ``{stage}{legacy_suffix}{tag}`` and passed that straight through as the
generated row's condition name. Only ``ablated_AD`` was relabelled afterwards,
so a ``_fullAD`` run wrote::

    M3_baseline_fullAD          <- wrong; CF2 expects M3_baseline
    M3_ablated_AD               <- right (relabelled)
    M3_ablated_random_fullAD    <- wrong; CF2 expects M3_ablated_random

Two silent consequences, neither of which raised anything:

* ``behavioral_judges._row_in_scope`` accepts a quadrant-A causal row whose
  condition ``endswith("_baseline")``. ``M3_baseline_fullAD`` does not, so
  every full-A baseline row was marked ``out_of_scope`` and never scored.
* ``confirmatory_behavioral_endpoints`` matches an exact condition tuple, so
  every full-A triple was incomplete and dropped -> ``estimation_split_only``
  stayed at n=0 and ``full_A_sensitivity`` silently equalled the held-out
  ``primary`` block.

The RESPONSES are fine - the model, prompts, direction and hooks were all
correct. Only the labels are wrong, so this is a pure metadata repair: no
regeneration needed. Rows that still lack judge scores do need re-judging
afterwards (see behavioral_judges --resume-from).

Idempotent: a file already carrying canonical labels is reported and left
alone. Never touches the confirmatory ``*_L24-28.json`` files - their labels
were always correct, and they are the frozen preregistered artifact.

    python -m src.analysis.relabel_causal_conditions --dry-run
    python -m src.analysis.relabel_causal_conditions
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

RAW_DIR = Path("results/raw")

# tagged causal files only; the untagged confirmatory file is off-limits
TAGGED_GLOB = "causal_ablation_v2_*_L24-28_*.json"

# shard-unit spelling -> canonical condition suffix
LEGACY_SUFFIX = {"_ablated": "_ablated_AD"}
CANONICAL = ("_baseline", "_ablated_AD", "_ablated_random", "_ablated_AB")

# tags stage_causal may append to a shard-unit name
TAG_RE = re.compile(r"(_fullAD|_xfit\d+(?:f\d+)?|_dirfrom_[A-Za-z0-9_]+)$")


def canonical_condition(label: str, stage: str) -> str:
    """``M3_baseline_fullAD`` -> ``M3_baseline``; ``M3_ablated`` ->
    ``M3_ablated_AD``. A label already canonical comes back unchanged.

    Cross-fit labels (``M3_xfit_baseline``) are deliberately left alone: that
    infix is a real condition distinction, not a shard tag, and collapsing it
    would let cross-fitted rows overwrite ordinary ones for the same prompt.
    """
    if not label.startswith(stage):
        return label
    suffix = label[len(stage):]
    if suffix.startswith("_xfit_"):
        return label
    suffix = TAG_RE.sub("", suffix)
    suffix = LEGACY_SUFFIX.get(suffix, suffix)
    return f"{stage}{suffix}" if suffix in CANONICAL else label


def stage_of(path: Path) -> str:
    """``causal_ablation_v2_M3_direct_alt_L24-28_fullAD.json`` -> ``M3_direct_alt``."""
    name = path.name
    prefix, marker = "causal_ablation_v2_", "_L24-28"
    if not name.startswith(prefix) or marker not in name:
        raise ValueError(f"not a causal-ablation filename: {name}")
    return name[len(prefix):name.index(marker)]


def relabel_rows(rows: list, stage: str) -> tuple[list, Counter]:
    """Returns (rows, changes) with ``stage`` and ``condition`` canonicalised."""
    changes = Counter()
    for row in rows:
        for field in ("stage", "condition"):
            old = row.get(field)
            if not isinstance(old, str):
                continue
            new = canonical_condition(old, stage)
            if new != old:
                row[field] = new
                changes[f"{old} -> {new}"] += 1
    return rows, changes


def relabel_file(path: Path, *, dry_run: bool = False) -> dict:
    stage = stage_of(path)
    rows = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    before = Counter(r.get("stage") for r in rows)
    rows, changes = relabel_rows(rows, stage)
    after = Counter(r.get("stage") for r in rows)

    if changes and not dry_run:
        # write via a temp file so an interrupted run cannot truncate a
        # 900-row artifact that took GPU time to produce
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        tmp.replace(path)

    return {"file": path.name, "stage": stage, "n_rows": len(rows),
            "conditions_before": dict(before), "conditions_after": dict(after),
            "changes": dict(changes), "modified": bool(changes) and not dry_run}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    files = sorted(raw.glob(TAGGED_GLOB))
    files = [f for f in files if not f.name.endswith("_binding.json")]
    if not files:
        print(f"no tagged causal files under {raw}")
        return

    total = 0
    for path in files:
        r = relabel_file(path, dry_run=args.dry_run)
        total += sum(r["changes"].values())
        flag = "DRY-RUN" if args.dry_run else ("REWROTE" if r["modified"] else "ok")
        print(f"\n{r['file']}  ({r['n_rows']} rows)  [{flag}]")
        print(f"  before: {r['conditions_before']}")
        if r["changes"]:
            for k, v in sorted(r["changes"].items()):
                print(f"    {v:5d}  {k}")
            print(f"  after : {r['conditions_after']}")
        else:
            print("  labels already canonical - nothing to do")

    print(f"\n{total} label(s) {'would be' if args.dry_run else ''} changed "
          f"across {len(files)} file(s).")
    if total and not args.dry_run:
        print("Re-judge the newly in-scope rows next:\n"
              "  python -m src.analysis.behavioral_judges --response-manifest ... "
              "--run-live --scope confirmatory --resume-from <previous judged file>")


if __name__ == "__main__":
    main()
