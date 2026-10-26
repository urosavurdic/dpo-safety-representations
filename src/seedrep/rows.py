"""The prompts every job runs on, in a fixed order, with their bindings.

Benchmark rows come from the tracked ``results/activations/M1_metadata.json``.
It is the identity snapshot written by the original extraction, so its row
order is the order of every stored activation array, and it carries the
split labels. The benchmark and split-manifest SHA-256s come from its binding
sidecar and are stamped on every response row, exactly as ``result_row`` does.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH_META = ROOT / "results" / "activations" / "M1_metadata.json"
BENCH_BINDING = ROOT / "results" / "activations" / "M1_metadata_binding.json"
HELDOUT_PAIRS = ROOT / "data" / "processed" / "pku_heldout_pairs.jsonl"
SFT_DATA = ROOT / "data" / "processed" / "sft_safety.jsonl"
DPO_DATA = ROOT / "data" / "processed" / "dpo_pairs.jsonl"

N_BENCH = 654


def benchmark_rows(path: Path = BENCH_META) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(rows) != N_BENCH:
        raise RuntimeError(f"{path}: expected {N_BENCH} rows, got {len(rows)}")
    return rows


def benchmark_shas(path: Path = BENCH_BINDING) -> tuple[str, str]:
    binding = json.loads(Path(path).read_text(encoding="utf-8"))
    return binding["benchmark_sha256"], binding["split_manifest_sha256"]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def heldout_pairs(path: Path = HELDOUT_PAIRS, n: int | None = None) -> list[dict]:
    pairs = read_jsonl(path)
    return pairs[:n] if n else pairs
