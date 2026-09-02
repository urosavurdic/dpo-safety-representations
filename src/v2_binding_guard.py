"""Fail-closed guard: reject legacy-370-era or unbound result artifacts (WP-Repro).

The frozen-v2 T4 run writes every response/intervention file with a
``benchmark_sha256`` + ``split_manifest_sha256`` on each row and a
``<file>_binding.json`` sidecar (see ``src/v2_io.binding`` / ``assert_binding``).
Pre-freeze (370-era) artifacts carry none of that. This module lets the CPU
statistics scripts and ``src/reproduce.py`` refuse to silently summarise a
370-era file as if it were a frozen-v2 (654-row, benchmark-bound) result.

CPU-only, no torch. The single source of truth for the frozen SHA / counts is
``src.analysis.v2_pipeline`` (imported lazily so tiny stat scripts that pass an
explicit expected SHA don't pull numpy/v2_shards).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.v2_io import load_json

# Known pre-freeze artifact basenames. A results file with any of these
# unambiguous 370-era names is refused outright. Deliberately NOT including
# over-generic names like "raw.json" - the per-row binding-field check below is
# the real signal and does not depend on the filename.
LEGACY_370_BASENAMES = frozenset({
    "causal_ablation_raw.json",
    "causal_ablation_raw_narrow.json",
    "causal_ablation_raw_wide.json",
    "causal_ablation_raw_narrow_M3.json",
    "steering_raw_D.json",
    "steering_raw_D_L21.json",
    "steering_raw_D_MULTILAYER_14to28_DEPRECATED.json",
    "steering_raw_D_L21_exploratory_DEPRECATED.json",
})

BINDING_KEYS = ("benchmark_sha256", "split_manifest_sha256")


class LegacyArtifactError(RuntimeError):
    """Raised when an unbound / 370-era artifact reaches a frozen-v2-only path."""


def _frozen_sha() -> str:
    from src.analysis.v2_pipeline import FROZEN_V2_BENCHMARK_SHA256

    return FROZEN_V2_BENCHMARK_SHA256


def assert_not_legacy_basename(path: str | Path) -> None:
    name = Path(path).name
    if name in LEGACY_370_BASENAMES:
        raise LegacyArtifactError(
            f"{name!r} is a known pre-freeze (370-era) artifact. The frozen-v2 "
            "path writes results/raw/causal_ablation_v2_<stage>_L24-28.json (and "
            "the behavioural/steering equivalents) with a *_binding.json sidecar. "
            "Refusing to consume the legacy file. Pass --allow-unbound only for "
            "deliberate historical/manual reproduction."
        )
    if name.startswith("causal_ablation_raw_") and "_v2_" not in name:
        raise LegacyArtifactError(
            f"{name!r} matches the pre-freeze causal_ablation_raw_* naming. The "
            "frozen-v2 file is causal_ablation_v2_<stage>_L24-28.json."
        )


def iter_rows(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("rows", "records", "results"):
            if isinstance(obj.get(key), list):
                return obj[key]
    raise LegacyArtifactError(
        "Results file is neither a JSON list of rows nor a {rows:[...]} object."
    )


def assert_rows_bound(
    rows: Iterable[dict[str, Any]],
    *,
    benchmark_sha256: str | None = None,
    split_manifest_sha256: str | None = None,
    require_frozen: bool = True,
) -> list[dict[str, Any]]:
    """Every row must carry both binding SHAs; if ``require_frozen`` (default),
    ``benchmark_sha256`` must equal the pinned frozen-v2 SHA unless an explicit
    ``benchmark_sha256`` override is supplied (fixture tests do this)."""
    rows = list(rows)
    if not rows:
        raise LegacyArtifactError("Results file has no rows.")

    expected_bench = benchmark_sha256 or (_frozen_sha() if require_frozen else None)

    for i, row in enumerate(rows):
        missing = [k for k in BINDING_KEYS if not row.get(k)]
        if missing:
            raise LegacyArtifactError(
                f"row {i} is missing binding field(s) {missing}. A 370-era file "
                "carries no benchmark/split SHA per row - it cannot be consumed "
                "on the frozen-v2 path."
            )
        if expected_bench and row["benchmark_sha256"] != expected_bench:
            raise LegacyArtifactError(
                f"row {i} benchmark_sha256={row['benchmark_sha256']!r} does not "
                f"match the expected frozen-v2 benchmark {expected_bench!r}."
            )
        if split_manifest_sha256 and row["split_manifest_sha256"] != split_manifest_sha256:
            raise LegacyArtifactError(
                f"row {i} split_manifest_sha256={row['split_manifest_sha256']!r} "
                f"!= expected {split_manifest_sha256!r}."
            )
    return rows


def load_guarded_raw(
    path: str | Path,
    *,
    benchmark_sha256: str | None = None,
    split_manifest_sha256: str | None = None,
    require_frozen: bool = True,
    allow_unbound: bool = False,
) -> list[dict[str, Any]]:
    """Load a v2 results JSON, refusing legacy/unbound files unless
    ``allow_unbound`` is set (the CLI ``--allow-unbound`` escape hatch)."""
    obj = load_json(path)
    rows = iter_rows(obj)
    if allow_unbound:
        return rows
    assert_not_legacy_basename(path)
    return assert_rows_bound(
        rows,
        benchmark_sha256=benchmark_sha256,
        split_manifest_sha256=split_manifest_sha256,
        require_frozen=require_frozen,
    )


def add_binding_cli_args(parser) -> None:
    """Shared argparse surface for the guarded stat scripts."""
    parser.add_argument(
        "--allow-unbound",
        action="store_true",
        help="Consume a pre-freeze / unbound results file on purpose (historical "
             "or manual reproduction). Without it, only frozen-v2 benchmark-bound "
             "files are accepted.",
    )
    parser.add_argument(
        "--expect-benchmark-sha256",
        default=None,
        help="Override the expected benchmark SHA (fixture tests / a deliberate "
             "re-freeze). Defaults to the pinned frozen-v2 SHA.",
    )
