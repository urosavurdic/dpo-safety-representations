"""Build a consolidated response manifest over crossbranch raw output so the
existing judge pipeline can score it (CPU, torch-free).

``behavioral_judges.run_judges`` consumes only a
``consolidated_response_manifest`` and its ``RESPONSE_GLOBS`` do not cover
``results/crossbranch/raw``. Rather than edit that module, this builds the same
manifest shape pointing at crossbranch files.

Two things happen on the way, both deliberate:

1. **Quadrant filtering.** The judges are ~7B models; scoring all four
   quadrants of every arm at every coefficient in both directions is hours of
   T4 time for rows that answer nothing. StrongREJECT and WildGuard are
   harmfulness judges, so they are meaningful on the harmful quadrants (A
   overt, C reduced-cue) and near-vacuous on the benign ones (B, D) -- where
   the rule-based over-refusal rate is the endpoint that matters and is
   already computed. So filtered copies carrying only the requested quadrants
   are written to their own directory and the manifest points at those.
   ``verify_manifest_entry`` validates the binding sidecar plus each row's own
   SHAs and never checks row count, so a row subset verifies cleanly.

2. **Arm keys are made fully qualified.** ``regex_only_record`` keeps a fixed
   set of row fields and drops ``coef``, ``source_branch`` and
   ``target_branch``. Without those, every ``xfer_delta_source_identity`` row
   looks identical in the judge output whether it came from A->B at coef 1.0
   or B->A at coef 2.0 -- the scores could not be attributed back to an arm.
   So each filtered row's ``stage`` and ``condition`` are rewritten to
   ``"{tag}|{condition}|coef{suffix}"``, which survives into the judge records
   and is parsed back by analyze_judges. The originals are preserved on the row
   as ``original_stage`` / ``original_condition``, and the rewrite is recorded
   in the manifest.

The filtered files are DERIVED judge inputs, not results. They live under their
own directory and are never consumed by the rule-based analysis path.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.analysis.crossbranch.analyze import condition_key_from_filename
from src.analysis.crossbranch.branches import BRANCHES
from src.v2_binding_guard import add_binding_cli_args, load_guarded_raw
from src.v2_io import load_json, write_json_lf

RAW_DIR = Path("results/crossbranch/raw")
JUDGE_INPUT_DIR = Path("results/crossbranch/judge_inputs")
MANIFEST_PATH = Path("results/crossbranch/manifests/crossbranch_judge_manifest.json")

JUDGE_QUADRANTS = ("A", "C")

# results/crossbranch/raw/crossbranch_<tag>_<condition>_coef<suffix>.json
_PREFIX = "crossbranch_"

# Every valid direction tag, e.g. "AtoB", "A_directtoB_direct". Matched
# longest-first so "A_directto..." is not truncated at "A" -- a plain
# partition("_") breaks once a branch name itself contains "_".
_DIRECTION_TAGS = tuple(sorted(
    (f"{s}to{t}" for s in BRANCHES for t in BRANCHES if s != t),
    key=len, reverse=True,
))


def parse_raw_filename(stem: str) -> tuple[str, str, str]:
    """``crossbranch_AtoB_own_delta_target_coef1`` -> (tag, condition, coef).

    ``coef`` is returned as the literal filename suffix (``"1"``, ``"0.5"``,
    ``"na"``) so it round-trips exactly rather than through a float.
    """
    if not stem.startswith(_PREFIX):
        raise ValueError(f"{stem!r} is not a crossbranch raw filename")
    rest = stem[len(_PREFIX):]
    for tag in _DIRECTION_TAGS:
        if rest.startswith(tag + "_"):
            tail = rest[len(tag) + 1:]
            key = condition_key_from_filename(tail)   # "cond" or "cond@coef"
            condition, sep, coef = key.partition("@")
            return tag, condition, (coef if sep else "na")
    raise ValueError(
        f"no known direction tag at the start of {stem!r} "
        f"(known: {list(_DIRECTION_TAGS)})"
    )


def arm_key(tag: str, condition: str, coef: str) -> str:
    """Fully-qualified arm identity that survives regex_only_record."""
    return f"{tag}|{condition}|coef{coef}"


def parse_arm_key(key: str) -> tuple[str, str, str]:
    tag, condition, coef = key.split("|", 2)
    if not coef.startswith("coef"):
        raise ValueError(f"{key!r} does not carry a coef segment")
    return tag, condition, coef[len("coef"):]


def filter_rows(rows: list[dict], quadrants, tag, condition, coef) -> list[dict]:
    wanted = set(quadrants)
    key = arm_key(tag, condition, coef)
    out = []
    for row in rows:
        if row.get("quadrant") not in wanted:
            continue
        new = dict(row)
        new["original_stage"] = row.get("stage")
        new["original_condition"] = row.get("condition")
        new["stage"] = key
        new["condition"] = key
        new["arm_key"] = key
        out.append(new)
    return out


def build(
    raw_dir=RAW_DIR,
    out_dir=JUDGE_INPUT_DIR,
    manifest_path=MANIFEST_PATH,
    quadrants=JUDGE_QUADRANTS,
    *,
    directions=None,
    conditions=None,
    expect_sha=None,
    allow_unbound=False,
) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries, summary = [], []
    bench_sha = split_sha = None
    want_conds = set(conditions) if conditions else None

    for path in sorted(raw_dir.glob(f"{_PREFIX}*.json")):
        if path.name.endswith("_binding.json"):
            continue
        tag, condition, coef = parse_raw_filename(path.stem)
        if directions and tag not in directions:
            continue
        if want_conds and condition not in want_conds:
            continue

        binding_src = path.with_name(path.stem + "_binding.json")
        if not binding_src.exists():
            raise SystemExit(
                f"{path.name} has no binding sidecar; the judge pipeline "
                "verifies one per response file and will refuse it."
            )
        rows = load_guarded_raw(
            path, benchmark_sha256=expect_sha, allow_unbound=allow_unbound
        )
        kept = filter_rows(rows, quadrants, tag, condition, coef)
        if not kept:
            continue

        dst = out_dir / path.name
        write_json_lf(dst, kept)
        shutil.copy2(binding_src, dst.with_name(dst.stem + "_binding.json"))

        if bench_sha is None:
            b = load_json(binding_src)
            bench_sha = b.get("benchmark_sha256")
            split_sha = b.get("split_manifest_sha256")

        entries.append({
            "response_file": str(dst).replace("\\", "/"),
            "binding_file": str(dst.with_name(dst.stem + "_binding.json")).replace("\\", "/"),
        })
        summary.append({
            "arm_key": arm_key(tag, condition, coef),
            "direction": tag, "condition": condition, "coef": coef,
            "n_rows": len(kept), "n_rows_original": len(rows),
        })

    if not entries:
        raise SystemExit(
            f"No crossbranch raw files under {raw_dir} matched "
            f"quadrants={list(quadrants)} directions={directions or 'any'}"
        )

    manifest = {
        "kind": "consolidated_response_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_sha256": bench_sha,
        "split_manifest_sha256": split_sha,
        "entries": entries,
        "crossbranch": {
            "source_raw_dir": str(raw_dir).replace("\\", "/"),
            "quadrants": list(quadrants),
            "arm_key_format": "{direction_tag}|{condition}|coef{suffix}",
            "arm_key_note": (
                "stage/condition on these DERIVED judge-input rows are rewritten "
                "to the fully-qualified arm key because regex_only_record drops "
                "coef/source_branch/target_branch; original values are kept as "
                "original_stage / original_condition."
            ),
            "arms": summary,
        },
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a judge manifest over crossbranch raw output."
    )
    p.add_argument("--raw-dir", default=str(RAW_DIR))
    p.add_argument("--out-dir", default=str(JUDGE_INPUT_DIR))
    p.add_argument("--manifest", default=str(MANIFEST_PATH))
    p.add_argument("--quadrants", nargs="+", default=list(JUDGE_QUADRANTS))
    p.add_argument(
        "--directions", nargs="+", default=None,
        help="Restrict to these direction tags (e.g. AtoB BtoA). Default: all.",
    )
    p.add_argument(
        "--conditions", nargs="+", default=None,
        help="Restrict to these condition names (e.g. baseline_target "
             "reference_target). Default: all.",
    )
    add_binding_cli_args(p)
    args = p.parse_args()

    manifest = build(
        args.raw_dir, args.out_dir, args.manifest, args.quadrants,
        directions=args.directions,
        conditions=args.conditions,
        expect_sha=args.expect_benchmark_sha256,
        allow_unbound=args.allow_unbound,
    )
    arms = manifest["crossbranch"]["arms"]
    total = sum(a["n_rows"] for a in arms)
    print(
        f"{len(arms)} arms, {total} rows to judge "
        f"(quadrants {' '.join(args.quadrants)})"
    )
    for a in arms:
        print(f"  {a['arm_key']:58s} {a['n_rows']:4d} / {a['n_rows_original']}")
    print(f"\nWrote {args.manifest}")
    print(
        "\nNext (GPU, gated HF repos -- needs HF_TOKEN):\n"
        f"  python -m src.analysis.behavioral_judges "
        f"--response-manifest {args.manifest} "
        "--out-dir results/crossbranch/judges --run-live --scope all"
    )


if __name__ == "__main__":
    main()
