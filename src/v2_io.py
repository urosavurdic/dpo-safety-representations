"""Strict benchmark-bound I/O for the v2 Colab rerun."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


LATEST_BENCHMARK = Path("data/frozen_v2/LATEST_BENCHMARK.json")
DEFAULT_SPLIT_MANIFEST = Path("logs/direction_split_manifest.json")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_json(data: Any) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def write_json_lf(path: str | Path, data: Any, indent: int = 2) -> None:
    """Write UTF-8 JSON without platform newline translation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def normalize_json_path(value: str | Path) -> Path:
    return Path(str(value).replace("\\", "/"))


def identity_snapshot(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "record_id": row.get("record_id"),
            "prompt": row.get("prompt"),
            "scored_prompt": row.get("scored_prompt"),
            "quadrant": row.get("quadrant"),
            "source": row.get(
                "source",
                row.get("source_dataset"),
            ),
            "source_dataset": row.get("source_dataset"),
            "c_construction": row.get("c_construction"),
            "split": row.get("split"),
        }
        for row in rows
    ]


def resolve_benchmark(
    eval_set: str | Path | None = None,
    latest_path: str | Path = LATEST_BENCHMARK,
) -> tuple[Path, str]:
    latest_path = Path(latest_path)
    if not latest_path.exists():
        raise FileNotFoundError(
            f"Missing frozen benchmark pointer: {latest_path}"
        )

    latest = load_json(latest_path)
    pointer_path = normalize_json_path(
        latest.get("benchmark_path", "")
    )
    pointer_sha = latest.get("benchmark_sha256")

    if not pointer_path or not pointer_sha:
        raise RuntimeError(
            "LATEST_BENCHMARK.json must contain benchmark_path and "
            "benchmark_sha256."
        )

    requested_path = (
        normalize_json_path(eval_set)
        if eval_set is not None
        else pointer_path
    )

    if not requested_path.exists():
        raise FileNotFoundError(
            f"Frozen benchmark does not exist: {requested_path}"
        )

    actual_sha = sha256_file(requested_path)
    if actual_sha != pointer_sha:
        raise RuntimeError(
            "Frozen benchmark hash mismatch:\n"
            f"  path:     {requested_path}\n"
            f"  recorded: {pointer_sha}\n"
            f"  actual:   {actual_sha}"
        )

    if requested_path.resolve() != pointer_path.resolve():
        raise RuntimeError(
            "Requested eval set is not the benchmark referenced by "
            f"{latest_path}:\n"
            f"  requested: {requested_path}\n"
            f"  pointer:   {pointer_path}"
        )

    return requested_path, actual_sha


def split_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {
            "split_manifest_sha256",
            "split_file_sha256",
        }
    }


def load_run_inputs(
    eval_set: str | Path | None = None,
    benchmark_sha256: str | None = None,
    split_manifest: str | Path = DEFAULT_SPLIT_MANIFEST,
    latest_path: str | Path = LATEST_BENCHMARK,
) -> tuple[Path, str, Path, str]:
    """Resolve and cross-check the frozen inputs for one run.

    `latest_path` names the pointer file to bind against. It defaults to the
    main benchmark's; a companion set (paired source prompts, the
    C-source-authored arm) passes its own pointer so it is hash-bound just
    as strictly, without being mistaken for the main benchmark.
    """
    benchmark_path, actual_benchmark_sha = resolve_benchmark(
        eval_set,
        latest_path,
    )

    if (
        benchmark_sha256 is not None
        and benchmark_sha256 != actual_benchmark_sha
    ):
        raise RuntimeError(
            "Supplied benchmark SHA does not match the frozen benchmark: "
            f"{benchmark_sha256} != {actual_benchmark_sha}"
        )

    split_path = Path(split_manifest)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing direction split manifest: {split_path}"
        )

    manifest = load_json(split_path)

    if manifest.get("benchmark_sha256") != actual_benchmark_sha:
        raise RuntimeError(
            "Direction split manifest is bound to a different benchmark: "
            f"{manifest.get('benchmark_sha256')} != "
            f"{actual_benchmark_sha}"
        )

    recorded_split_sha = manifest.get("split_manifest_sha256")
    if not recorded_split_sha:
        raise RuntimeError(
            f"{split_path} has no split_manifest_sha256."
        )

    calculated_split_sha = sha256_bytes(
        canonical_json(split_payload(manifest))
    )
    if calculated_split_sha != recorded_split_sha:
        raise RuntimeError(
            "Direction split manifest content hash mismatch: "
            f"{calculated_split_sha} != {recorded_split_sha}"
        )

    return (
        benchmark_path,
        actual_benchmark_sha,
        split_path,
        recorded_split_sha,
    )


def binding(
    benchmark_path: str | Path,
    benchmark_sha256: str,
    split_path: str | Path,
    split_manifest_sha256: str,
) -> dict[str, str]:
    return {
        "benchmark_path": Path(benchmark_path).as_posix(),
        "benchmark_sha256": benchmark_sha256,
        "split_manifest_path": Path(split_path).as_posix(),
        "split_manifest_sha256": split_manifest_sha256,
    }


def assert_binding(
    path: str | Path,
    benchmark_sha256: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact binding: {path}")

    data = load_json(path)

    if data.get("benchmark_sha256") != benchmark_sha256:
        raise RuntimeError(
            f"Artifact {path} is bound to a different benchmark."
        )

    if data.get("split_manifest_sha256") != split_manifest_sha256:
        raise RuntimeError(
            f"Artifact {path} is bound to a different split manifest."
        )

    return data
