"""Shard-level checkpointing, length-sorted batching and deadline control.

The v2 GPU run targets free-tier T4 sessions with a hard wall-clock cap of
roughly 5:30. Some single stage-conditions take longer than that, so
checkpointing at stage granularity is not enough: a session that ends
mid-stage would repeat the whole stage. Work is therefore split into shards
of BATCH_SIZE rows, each shard is committed to disk as soon as it finishes,
and a resumed session skips shards that are already on disk.

Three invariants hold every checkpoint honest:

1. Progress is bound to (benchmark_sha256, split_manifest_sha256). Resuming
   against a different benchmark raises instead of silently mixing rows from
   two eval sets into one output file.
2. A shard file is written to a temporary name and then os.replace()d, so a
   process killed during the write leaves either the old file or the new one,
   never a truncated one.
3. Merged output is restored to benchmark row order by record_id, so
   length-sorted batching cannot leak into the ordering of results.

This module also carries the two OOM-related utilities used to keep a T4
session's chosen batch size honest:

* `run_with_oom_backoff` halves a shard in place when a forward/generate
  call raises a CUDA out-of-memory error, retries each half, and stitches
  the pieces back together with a caller-supplied `combine`. This changes
  how many model calls one shard costs; it never changes which rows the
  shard contains or their order, so it composes with `plan_shards` and
  `ShardStore` without affecting determinism.
* `probe_batch_capacity` measures the largest batch size that a given unit
  of work survives before OOM, so a batch size can be selected from
  measured GPU capacity rather than guessed.

Both take the actual retry/measurement callable as a parameter rather than
importing torch, so this module stays test-light: the recursion and
doubling logic is exercised with plain Python fakes, and only the real T4
run supplies a callable that touches the GPU.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from src.v2_io import load_json, write_json_lf


PROGRESS_VERSION = 2


class DeadlineReached(Exception):
    """Raised to unwind cleanly when the session budget is exhausted."""


class Deadline:
    """Wall-clock budget for one Colab session.

    `minutes=None` means unlimited, which is what local CPU tests use.
    Checks happen at shard boundaries only: a shard is either fully
    committed or not started, never half-done.
    """

    def __init__(self, minutes: float | None = None) -> None:
        self.minutes = minutes
        self._start = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start

    @property
    def remaining_seconds(self) -> float:
        if self.minutes is None:
            return float("inf")
        return self.minutes * 60.0 - self.elapsed_seconds

    def expired(self) -> bool:
        return self.remaining_seconds <= 0

    def would_exceed(self, estimated_seconds: float) -> bool:
        """True if a unit of work this long would run past the budget.

        Used to stop *before* starting a shard that cannot finish, rather
        than being killed part-way through it.
        """
        if self.minutes is None:
            return False
        return estimated_seconds > self.remaining_seconds

    def check(self, estimated_seconds: float = 0.0) -> None:
        if self.expired() or self.would_exceed(estimated_seconds):
            raise DeadlineReached(self.describe())

    def describe(self) -> str:
        if self.minutes is None:
            return "no deadline"
        return (
            f"{self.elapsed_seconds / 60.0:.1f} min elapsed of "
            f"{self.minutes:.0f} min budget "
            f"({self.remaining_seconds / 60.0:.1f} min left)"
        )


def _length_key(row: dict[str, Any], measure: Callable[[str], int]):
    # record_id breaks ties so the plan is identical across sessions and
    # machines; without it, equal-length rows could be ordered by dict
    # iteration and produce different batch composition on a resume.
    return (measure(row["prompt"]), row.get("record_id") or "")


def plan_shards(
    rows: Sequence[dict[str, Any]],
    batch_size: int,
    measure: Callable[[str], int] | None = None,
    sort_by_length: bool = True,
) -> list[list[dict[str, Any]]]:
    """Split rows into deterministic shards, optionally length-sorted.

    Grouping similar-length prompts together collapses padding waste, which
    on a bandwidth-bound T4 is a larger throughput win than raising the
    batch size. `measure` defaults to a whitespace word count; callers with
    a tokenizer should pass real token lengths.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if measure is None:
        def measure(text: str) -> int:
            return len(text.split())

    ordered = list(rows)
    if sort_by_length:
        ordered.sort(key=lambda row: _length_key(row, measure))

    return [
        ordered[start:start + batch_size]
        for start in range(0, len(ordered), batch_size)
    ]


class ShardStore:
    """Append-only shard storage for one component (behavior, causal, ...).

    Layout:
        <root>/parts/<stage>__<condition>__<index:04d>.json
        <root>/progress.json
    """

    def __init__(
        self,
        root: str | Path,
        benchmark_sha256: str,
        split_manifest_sha256: str,
    ) -> None:
        self.root = Path(root)
        self.parts_dir = self.root / "parts"
        self.progress_path = self.root / "progress.json"
        self.benchmark_sha256 = benchmark_sha256
        self.split_manifest_sha256 = split_manifest_sha256
        self._progress = self._load_progress()

    # ---- progress bookkeeping -------------------------------------------

    def _empty_progress(self) -> dict[str, Any]:
        return {
            "progress_version": PROGRESS_VERSION,
            "benchmark_sha256": self.benchmark_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "units": {},
        }

    def _load_progress(self) -> dict[str, Any]:
        if not self.progress_path.exists():
            return self._empty_progress()

        data = load_json(self.progress_path)

        recorded_benchmark = data.get("benchmark_sha256")
        recorded_split = data.get("split_manifest_sha256")

        if recorded_benchmark != self.benchmark_sha256:
            raise RuntimeError(
                f"{self.progress_path} records benchmark "
                f"{recorded_benchmark} but this run is bound to "
                f"{self.benchmark_sha256}. Refusing to resume across two "
                "different benchmarks; move the directory aside to start "
                "a fresh run."
            )
        if recorded_split != self.split_manifest_sha256:
            raise RuntimeError(
                f"{self.progress_path} records split manifest "
                f"{recorded_split} but this run is bound to "
                f"{self.split_manifest_sha256}."
            )
        if data.get("progress_version") != PROGRESS_VERSION:
            raise RuntimeError(
                f"{self.progress_path} has progress_version "
                f"{data.get('progress_version')!r}; expected "
                f"{PROGRESS_VERSION}."
            )

        data.setdefault("units", {})
        return data

    def _save_progress(self) -> None:
        write_json_lf(self.progress_path, self._progress)

    @staticmethod
    def unit_key(stage: str, condition: str) -> str:
        return f"{stage}__{condition}"

    def _unit(self, unit_key: str) -> dict[str, Any]:
        return self._progress["units"].setdefault(
            unit_key,
            {"completed_shards": [], "n_shards": None, "n_rows": None},
        )

    def declare_unit(
        self,
        unit_key: str,
        n_shards: int,
        n_rows: int,
    ) -> None:
        """Record the expected shape of a unit before any shard runs.

        A shard count that changes between sessions means the plan changed
        (different batch size, different row filter) and any existing
        shards are not reusable. n_shards alone is not sufficient: a row
        filter or benchmark edit can change the total row count while
        leaving the *shard count* unchanged by coincidence (ceil(n/batch)
        is many-to-one), which would otherwise let stale shards for a
        different row set be silently reused. n_rows is checked too so
        that case fails closed as well.
        """
        unit = self._unit(unit_key)

        if unit["n_shards"] is not None and unit["n_shards"] != n_shards:
            raise RuntimeError(
                f"{unit_key}: shard plan changed ({unit['n_shards']} -> "
                f"{n_shards}). Existing shards cannot be reused; change "
                "the batch size back or clear this unit's parts."
            )

        if unit["n_rows"] is not None and unit["n_rows"] != n_rows:
            raise RuntimeError(
                f"{unit_key}: shard plan changed (row count "
                f"{unit['n_rows']} -> {n_rows} with n_shards={n_shards} "
                "unchanged). Existing shards cannot be reused; change the "
                "row filter back or clear this unit's parts."
            )

        unit["n_shards"] = n_shards
        unit["n_rows"] = n_rows
        self._save_progress()

    # ---- shard I/O ------------------------------------------------------

    def shard_path(self, unit_key: str, index: int) -> Path:
        return self.parts_dir / f"{unit_key}__{index:04d}.json"

    def completed_shards(self, unit_key: str) -> set[int]:
        """Indices already on disk.

        Cross-checked against the filesystem rather than trusted from
        progress.json alone, so a lost/partial Drive sync shows up as work
        to redo instead of silently missing rows in the merge.
        """
        recorded = set(self._unit(unit_key)["completed_shards"])
        return {
            index
            for index in recorded
            if self.shard_path(unit_key, index).exists()
        }

    def write_shard(
        self,
        unit_key: str,
        index: int,
        rows: Iterable[dict[str, Any]],
    ) -> Path:
        path = self.shard_path(unit_key, index)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = list(rows)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)

        unit = self._unit(unit_key)
        if index not in unit["completed_shards"]:
            unit["completed_shards"].append(index)
            unit["completed_shards"].sort()
        self._save_progress()

        return path

    def unit_complete(self, unit_key: str) -> bool:
        unit = self._unit(unit_key)
        if unit["n_shards"] is None:
            return False
        return len(self.completed_shards(unit_key)) == unit["n_shards"]

    def merge_unit(
        self,
        unit_key: str,
        order: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Concatenate a unit's shards, restored to benchmark row order.

        `order` maps record_id -> benchmark index. Rows whose record_id is
        absent from `order` sort last, keeping their relative shard order,
        rather than being dropped.
        """
        if not self.unit_complete(unit_key):
            unit = self._unit(unit_key)
            raise RuntimeError(
                f"{unit_key} is incomplete: "
                f"{len(self.completed_shards(unit_key))}/"
                f"{unit['n_shards']} shards present."
            )

        merged: list[dict[str, Any]] = []
        for index in sorted(self.completed_shards(unit_key)):
            merged.extend(load_json(self.shard_path(unit_key, index)))

        if order is not None:
            fallback = len(order)
            merged.sort(
                key=lambda row: order.get(
                    row.get("record_id"),
                    fallback,
                )
            )

        return merged

    # ---- reporting ------------------------------------------------------

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for unit_key, unit in sorted(self._progress["units"].items()):
            done = len(self.completed_shards(unit_key))
            total = unit["n_shards"]
            rows.append(
                {
                    "unit": unit_key,
                    "shards_done": done,
                    "shards_total": total,
                    "fraction": (
                        None if not total else round(done / total, 4)
                    ),
                    "complete": self.unit_complete(unit_key),
                }
            )
        return rows


def run_sharded(
    store: ShardStore,
    unit_key: str,
    shards: Sequence[Sequence[dict[str, Any]]],
    process: Callable[[Sequence[dict[str, Any]]], list[dict[str, Any]]],
    deadline: Deadline,
    label: str | None = None,
) -> bool:
    """Run `process` over unfinished shards, committing each immediately.

    Returns True when the unit finished, False when the deadline stopped it
    early. Raising DeadlineReached out of here is deliberate only for the
    caller that wants to unwind the whole session; the common case is a
    clean False so the caller can move on and report.
    """
    store.declare_unit(
        unit_key,
        n_shards=len(shards),
        n_rows=sum(len(shard) for shard in shards),
    )

    done = store.completed_shards(unit_key)
    pending = [index for index in range(len(shards)) if index not in done]
    name = label or unit_key

    if not pending:
        print(f"  {name}: already complete ({len(shards)} shards)")
        return True

    if done:
        print(
            f"  {name}: resuming with {len(done)}/{len(shards)} shards "
            "already on disk"
        )

    observed: list[float] = []

    for index in pending:
        # Estimate from this unit's own measured shards. The first shard is
        # always attempted, otherwise a fresh session with a nearly-spent
        # budget could stall forever without producing anything.
        estimate = (
            sum(observed) / len(observed) if observed else 0.0
        )
        if deadline.would_exceed(estimate) or deadline.expired():
            print(
                f"  {name}: stopping at shard {index}/{len(shards)} - "
                f"{deadline.describe()}"
            )
            return False

        started = time.monotonic()
        store.write_shard(unit_key, index, process(shards[index]))
        observed.append(time.monotonic() - started)

        print(
            f"  {name}: shard {index + 1}/{len(shards)} "
            f"({observed[-1]:.1f}s, {deadline.describe()})"
        )

    return True


# --------------------------------------------------------------------------
# OOM backoff and capacity probing
# --------------------------------------------------------------------------


def _looks_like_oom(exc: BaseException) -> bool:
    """Heuristic CUDA-OOM detection that does not require importing torch.

    torch.cuda.OutOfMemoryError is itself a RuntimeError subclass, and both
    it and the plain RuntimeError older torch versions raise for the same
    condition carry "out of memory" in the message. Matching on the
    message keeps this module import-light (no torch dependency) and works
    across whichever torch version a given Colab session has installed.
    """
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def run_with_oom_backoff(
    rows: Sequence[Any],
    call: Callable[[Sequence[Any]], Any],
    combine: Callable[[Any, Any], Any],
    min_batch_size: int = 1,
    on_retry: Callable[[], None] | None = None,
) -> Any:
    """Run `call(rows)`, halving the batch and retrying on CUDA OOM.

    `combine(left_result, right_result)` merges the two halves' results
    back into the shape `call` would have produced for the whole batch - a
    list concatenation for generated rows, an array concatenation for
    stacked activations. `on_retry` is called once per halving (typically
    `torch.cuda.empty_cache`) before the smaller calls are attempted.

    This only changes how many forward/generate calls one shard costs; the
    shard's rows, their order, and record_id identity are untouched, so it
    composes with `plan_shards`/`ShardStore` without affecting determinism
    or checkpoint granularity - a shard is still committed whole or not at
    all.

    Halving stops at `min_batch_size` (default 1): an OOM on a single row
    is a genuine failure and is re-raised rather than looping forever.
    """
    try:
        return call(rows)
    except RuntimeError as exc:
        if not _looks_like_oom(exc) or len(rows) <= min_batch_size:
            raise
        if on_retry is not None:
            on_retry()
        mid = max(min_batch_size, len(rows) // 2)
        left = run_with_oom_backoff(
            rows[:mid], call, combine, min_batch_size, on_retry
        )
        right = run_with_oom_backoff(
            rows[mid:], call, combine, min_batch_size, on_retry
        )
        return combine(left, right)


def probe_batch_capacity(
    run_batch: Callable[[int], None],
    start: int = 1,
    cap: int = 64,
    on_retry: Callable[[], None] | None = None,
) -> int:
    """Largest batch size for which `run_batch(size)` completes without OOM.

    Doubles from `start` (1, 2, 4, 8, ...) until a size OOMs or `cap` is
    reached, and returns the last size that succeeded - 0 if even `start`
    OOMs. This is a calibration-time measurement only: the real run always
    uses whatever act_batch/gen_batch was selected (measured or supplied),
    it does not re-probe per shard.
    """
    if start < 1:
        raise ValueError("start must be >= 1")

    working = 0
    size = start
    while size <= cap:
        try:
            run_batch(size)
            working = size
            size *= 2
        except RuntimeError as exc:
            if not _looks_like_oom(exc):
                raise
            if on_retry is not None:
                on_retry()
            break

    return working
