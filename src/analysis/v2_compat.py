"""Milestone 5A: make v2 GPU outputs consumable by the existing CPU
analysis/statistics layer.

v2's stage-major runner (v2_pipeline.py) writes under v2-specific names so
its artifacts never collide with pre-v2 committed results while a rerun is
in progress:

    refusal_direction/{stage}_v2_direction.npy
    refusal_direction/cosine_similarity_v2.json
    refusal_direction/quadrant_projections_v2.json
    probes_v2/{stage}_probe_results.json
    behavioral_eval/v2_raw.json   (per row: "response", not "completion")

The existing CPU statistics scripts (eval_refusal_direction.py's
downstream readers, summarize_probe_findings.py, summarize_cross_branch.py,
reproduce.py's "probes"/"direction"/"behavioral_stats" components) were
written against the legacy, un-suffixed names and the legacy "completion"
field. This module bridges the two without changing either side's math,
schema, or filenames -- it only copies/relabels already-computed output.

Bridged, main results/ root only. A namespaced companion run (e.g.
--namespace C-source-authored) is intentionally NOT bridged here: it is a
separate optional robustness arm the legacy readers never look at.

    refusal_direction/{stage}_v2_direction.npy    -> {stage}_direction.npy
    refusal_direction/cosine_similarity_v2.json   -> cosine_similarity.json
    refusal_direction/quadrant_projections_v2.json -> quadrant_projections.json
    probes_v2/{stage}_probe_results.json      -> probes/{stage}_probe_results.json
    behavioral_eval/v2_raw.json               -> behavioral_eval/raw.json
        (per row, "response" is also copied to "completion" -- the legacy
        field name eval_behavioral.py / reclassify_behavioral.py /
        summarize_cross_branch.py read -- alongside every original v2
        field, so nothing already downstream of the v2 name is lost.)

Safety: a legacy path this module has never bridged before (e.g. the real,
already-committed M0-M3 results from a prior non-v2 run) is never
overwritten silently. Each bridged file gets a companion
"..._v2_compat_binding.json" carrying the same benchmark/split identity the
source v2 artifact was bound to (via src.v2_io, the same mechanism
v2_pipeline.py itself uses). A later call only touches an existing legacy
file if that binding is present and matches, or if --force is passed.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from src.analysis.v2_pipeline import ALL_STAGES
from src.v2_io import (
    assert_binding,
    binding,
    load_json,
    write_json_lf,
)


def _provenance(binding_path: Path) -> dict[str, str]:
    data = load_json(binding_path)
    return binding(
        data["benchmark_path"],
        data["benchmark_sha256"],
        data["split_manifest_path"],
        data["split_manifest_sha256"],
    )


def _may_write(
    dest: Path,
    dest_binding: Path,
    provenance: dict[str, str],
    force: bool,
) -> bool:
    """Whether `dest` may be (re)written by the bridge.

    True if `dest` does not exist yet, or --force was passed, or `dest`
    was itself produced by an earlier bridge call bound to the same v2
    source (rerunning the bridge is then a safe no-op). False if `dest`
    already exists with no matching compat binding -- most likely one of
    the real, already-committed pre-v2 results -- so it is left alone.
    """
    if force or not dest.exists():
        return True
    try:
        assert_binding(
            dest_binding,
            provenance["benchmark_sha256"],
            provenance["split_manifest_sha256"],
        )
        return True
    except Exception:
        return False


def _atomic_copy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, dest)


def _skip_message(dest: Path) -> str:
    return (
        f"  compat: {dest} already present from a non-v2 source, "
        "leaving it alone (pass --force to overwrite)"
    )


# --------------------------------------------------------------------------
# per-artifact bridges
# --------------------------------------------------------------------------


def sync_direction(
    stage: str, root: Path = Path("results"), force: bool = False
) -> bool:
    refusal_dir = root / "refusal_direction"
    source = refusal_dir / f"{stage}_v2_direction.npy"
    source_binding = refusal_dir / f"{stage}_v2_direction_binding.json"

    if not source.exists() or not source_binding.exists():
        return False

    provenance = _provenance(source_binding)
    dest = refusal_dir / f"{stage}_direction.npy"
    dest_binding = refusal_dir / f"{stage}_direction_v2_compat_binding.json"

    if not _may_write(dest, dest_binding, provenance, force):
        print(_skip_message(dest))
        return False

    _atomic_copy(source, dest)
    write_json_lf(
        dest_binding, {**provenance, "bridged_from": str(source)}
    )
    print(f"  compat: {source} -> {dest}")
    return True


def sync_diagnostics(root: Path = Path("results"), force: bool = False) -> bool:
    refusal_dir = root / "refusal_direction"
    source_binding = refusal_dir / "v2_diagnostics_binding.json"

    if not source_binding.exists():
        return False

    provenance = _provenance(source_binding)
    wrote_any = False

    for v2_name, legacy_name in (
        ("cosine_similarity_v2.json", "cosine_similarity.json"),
        ("quadrant_projections_v2.json", "quadrant_projections.json"),
    ):
        source = refusal_dir / v2_name
        if not source.exists():
            continue

        dest = refusal_dir / legacy_name
        dest_binding = refusal_dir / f"{dest.stem}_v2_compat_binding.json"

        if not _may_write(dest, dest_binding, provenance, force):
            print(_skip_message(dest))
            continue

        _atomic_copy(source, dest)
        write_json_lf(
            dest_binding, {**provenance, "bridged_from": str(source)}
        )
        print(f"  compat: {source} -> {dest}")
        wrote_any = True

    return wrote_any


def sync_probes(
    stage: str, root: Path = Path("results"), force: bool = False
) -> bool:
    source = root / "probes_v2" / f"{stage}_probe_results.json"
    source_binding = root / "probes_v2" / f"{stage}_probe_binding.json"

    if not source.exists() or not source_binding.exists():
        return False

    provenance = _provenance(source_binding)
    dest_dir = root / "probes"
    dest = dest_dir / f"{stage}_probe_results.json"
    dest_binding = dest_dir / f"{stage}_probe_results_v2_compat_binding.json"

    if not _may_write(dest, dest_binding, provenance, force):
        print(_skip_message(dest))
        return False

    _atomic_copy(source, dest)
    write_json_lf(
        dest_binding, {**provenance, "bridged_from": str(source)}
    )
    print(f"  compat: {source} -> {dest}")
    return True


def sync_behavioral(root: Path = Path("results"), force: bool = False) -> bool:
    beh_dir = root / "behavioral_eval"
    source = beh_dir / "v2_raw.json"
    source_binding = beh_dir / "v2_binding.json"

    if not source.exists() or not source_binding.exists():
        return False

    provenance = _provenance(source_binding)
    dest = beh_dir / "raw.json"
    dest_binding = beh_dir / "raw_v2_compat_binding.json"

    if not _may_write(dest, dest_binding, provenance, force):
        print(_skip_message(dest))
        return False

    combined = load_json(source)
    bridged = {
        stage: [
            {**row, "completion": row.get("response")}
            for row in rows
        ]
        for stage, rows in combined.items()
    }

    write_json_lf(dest, bridged)
    write_json_lf(
        dest_binding, {**provenance, "bridged_from": str(source)}
    )
    print(f"  compat: {source} -> {dest}")
    return True


# --------------------------------------------------------------------------
# orchestration / CLI
# --------------------------------------------------------------------------


def sync_all(
    stages: list[str] | None = None,
    root: Path = Path("results"),
    force: bool = False,
) -> dict[str, bool]:
    stages = stages if stages is not None else ALL_STAGES

    results: dict[str, bool] = {}
    for stage in stages:
        results[f"direction:{stage}"] = sync_direction(
            stage, root=root, force=force
        )
        results[f"probes:{stage}"] = sync_probes(
            stage, root=root, force=force
        )

    results["diagnostics"] = sync_diagnostics(root=root, force=force)
    results["behavioral"] = sync_behavioral(root=root, force=force)

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge v2 GPU output filenames/fields to the legacy names "
            "the CPU statistics layer (src.reproduce and friends) reads."
        )
    )
    parser.add_argument(
        "--stages", nargs="+", choices=ALL_STAGES, default=ALL_STAGES
    )
    parser.add_argument("--root", default="results")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite a legacy artifact even if it wasn't produced by "
            "a previous compat run."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = sync_all(
        stages=args.stages, root=Path(args.root), force=args.force
    )
    bridged = sorted(name for name, ok in results.items() if ok)
    skipped = sorted(name for name, ok in results.items() if not ok)
    print(f"\nBridged: {bridged}")
    print(f"Skipped (no v2 source yet, or already present): {skipped}")


if __name__ == "__main__":
    main()
