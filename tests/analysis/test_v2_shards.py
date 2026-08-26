"""Tests for shard checkpointing, length-sorted batching and deadlines.

These protect the T4 session budget: if resume is wrong, a session boundary
either loses hours of generation or silently drops rows from a merged file.
"""

import json

import pytest

from src.analysis.v2_shards import (
    Deadline,
    ShardStore,
    plan_shards,
    run_sharded,
)


BENCH_SHA = "a" * 64
SPLIT_SHA = "b" * 64


def make_rows(n, prefix="r"):
    return [
        {"record_id": f"{prefix}{i:03d}", "prompt": " ".join(["w"] * (i + 1))}
        for i in range(n)
    ]


def store_at(tmp_path, bench=BENCH_SHA, split=SPLIT_SHA):
    return ShardStore(tmp_path / "component", bench, split)


# ---- plan_shards --------------------------------------------------------


def test_plan_shards_covers_every_row_exactly_once():
    rows = make_rows(10)
    shards = plan_shards(rows, batch_size=3)

    assert [len(s) for s in shards] == [3, 3, 3, 1]
    flat = [row["record_id"] for shard in shards for row in shard]
    assert sorted(flat) == sorted(r["record_id"] for r in rows)
    assert len(flat) == len(set(flat))


def test_plan_shards_groups_by_length():
    rows = [
        {"record_id": "long", "prompt": "a b c d e"},
        {"record_id": "short", "prompt": "a"},
        {"record_id": "mid", "prompt": "a b c"},
    ]
    shards = plan_shards(rows, batch_size=1)
    assert [s[0]["record_id"] for s in shards] == ["short", "mid", "long"]


def test_plan_shards_is_deterministic_for_equal_lengths():
    # Equal-length prompts must not depend on input order, or a resumed
    # session would build different batches than the one it resumes from.
    rows = [{"record_id": rid, "prompt": "a b"} for rid in ("c", "a", "b")]
    first = plan_shards(rows, batch_size=2)
    second = plan_shards(list(reversed(rows)), batch_size=2)

    assert [
        [r["record_id"] for r in s] for s in first
    ] == [[r["record_id"] for r in s] for s in second]


def test_plan_shards_respects_custom_measure():
    rows = [
        {"record_id": "a", "prompt": "xxxxxx"},
        {"record_id": "b", "prompt": "y y y"},
    ]
    shards = plan_shards(rows, batch_size=1, measure=len)
    assert [s[0]["record_id"] for s in shards] == ["b", "a"]


def test_plan_shards_can_preserve_input_order():
    rows = make_rows(4)
    shards = plan_shards(rows, batch_size=2, sort_by_length=False)
    assert [r["record_id"] for r in shards[0]] == ["r000", "r001"]


def test_plan_shards_rejects_bad_batch_size():
    with pytest.raises(ValueError):
        plan_shards(make_rows(2), batch_size=0)


# ---- ShardStore binding -------------------------------------------------


def test_progress_refuses_a_different_benchmark(tmp_path):
    store = store_at(tmp_path)
    store.declare_unit("M3__baseline", n_shards=1, n_rows=1)

    with pytest.raises(RuntimeError, match="different benchmark"):
        store_at(tmp_path, bench="c" * 64)


def test_progress_refuses_a_different_split_manifest(tmp_path):
    store = store_at(tmp_path)
    store.declare_unit("M3__baseline", n_shards=1, n_rows=1)

    with pytest.raises(RuntimeError, match="split manifest"):
        store_at(tmp_path, split="d" * 64)


def test_declare_unit_rejects_a_changed_shard_plan(tmp_path):
    store = store_at(tmp_path)
    store.declare_unit("M3__baseline", n_shards=4, n_rows=32)

    with pytest.raises(RuntimeError, match="shard plan changed"):
        store.declare_unit("M3__baseline", n_shards=8, n_rows=32)


def test_completed_shards_ignores_a_missing_file(tmp_path):
    # Drive sync can lose a part file after progress.json recorded it.
    store = store_at(tmp_path)
    store.declare_unit("u", n_shards=2, n_rows=2)
    store.write_shard("u", 0, [{"record_id": "r000"}])
    store.write_shard("u", 1, [{"record_id": "r001"}])
    assert store.completed_shards("u") == {0, 1}

    store.shard_path("u", 1).unlink()
    assert store.completed_shards("u") == {0}
    assert store.unit_complete("u") is False


# ---- multi-unit isolation -----------------------------------------------
#
# run_paired_conditions puts a baseline condition and a treated condition
# for the same stage into one ShardStore (see v2_pipeline.py). Completing
# or merging one must never touch the other's shard files or progress.


def test_units_in_the_same_store_do_not_share_progress_or_shard_files(
    tmp_path,
):
    store = store_at(tmp_path)
    baseline = ShardStore.unit_key("M3", "baseline")
    ablated = ShardStore.unit_key("M3", "ablated")

    store.declare_unit(baseline, n_shards=2, n_rows=2)
    store.declare_unit(ablated, n_shards=2, n_rows=2)

    store.write_shard(baseline, 0, [{"record_id": "r000"}])
    store.write_shard(baseline, 1, [{"record_id": "r001"}])

    # Finishing baseline must not mark ablated complete or create its files.
    assert store.unit_complete(baseline) is True
    assert store.unit_complete(ablated) is False
    assert store.completed_shards(ablated) == set()
    assert store.shard_path(baseline, 0) != store.shard_path(ablated, 0)
    assert store.shard_path(ablated, 0).exists() is False

    baseline_merged = store.merge_unit(baseline)
    assert [r["record_id"] for r in baseline_merged] == ["r000", "r001"]

    store.write_shard(ablated, 0, [{"record_id": "a000"}])
    store.write_shard(ablated, 1, [{"record_id": "a001"}])
    ablated_merged = store.merge_unit(ablated)

    # No cross-contamination between the two conditions' rows.
    assert [r["record_id"] for r in ablated_merged] == ["a000", "a001"]
    assert set(r["record_id"] for r in baseline_merged).isdisjoint(
        r["record_id"] for r in ablated_merged
    )

    # Isolation survives a fresh process picking the store back up.
    reopened = store_at(tmp_path)
    assert reopened.unit_complete(baseline) is True
    assert reopened.unit_complete(ablated) is True
    assert reopened.completed_shards(baseline) == {0, 1}
    assert reopened.completed_shards(ablated) == {0, 1}


# ---- merge --------------------------------------------------------------


def test_merge_restores_benchmark_order(tmp_path):
    rows = make_rows(6)
    order = {row["record_id"]: i for i, row in enumerate(rows)}
    shards = plan_shards(rows, batch_size=2)

    store = store_at(tmp_path)
    store.declare_unit("u", n_shards=len(shards), n_rows=len(rows))
    for index, shard in enumerate(shards):
        store.write_shard("u", index, shard)

    merged = store.merge_unit("u", order=order)
    assert [r["record_id"] for r in merged] == [
        r["record_id"] for r in rows
    ]


def test_merge_keeps_unknown_record_ids_last(tmp_path):
    store = store_at(tmp_path)
    store.declare_unit("u", n_shards=1, n_rows=2)
    store.write_shard(
        "u",
        0,
        [{"record_id": "mystery"}, {"record_id": "known"}],
    )

    merged = store.merge_unit("u", order={"known": 0})
    assert [r["record_id"] for r in merged] == ["known", "mystery"]


def test_merge_refuses_an_incomplete_unit(tmp_path):
    store = store_at(tmp_path)
    store.declare_unit("u", n_shards=3, n_rows=3)
    store.write_shard("u", 0, [{"record_id": "r000"}])

    with pytest.raises(RuntimeError, match="incomplete"):
        store.merge_unit("u")


# ---- run_sharded / resume ----------------------------------------------


def test_run_sharded_completes_and_is_idempotent(tmp_path):
    rows = make_rows(7)
    shards = plan_shards(rows, batch_size=3)
    store = store_at(tmp_path)
    calls = []

    def process(shard):
        calls.append(len(shard))
        return [{"record_id": r["record_id"], "response": "x"} for r in shard]

    assert run_sharded(store, "u", shards, process, Deadline()) is True
    assert sum(calls) == 7

    # Second pass must not recompute anything.
    calls.clear()
    assert run_sharded(store, "u", shards, process, Deadline()) is True
    assert calls == []


class BudgetAfter(Deadline):
    """Deadline that reports itself spent once `allow` shards are done.

    Deterministic stand-in for a session cutoff: sleeping to burn a real
    wall-clock budget would make these tests slow and flaky.
    """

    def __init__(self, counter, allow):
        super().__init__(minutes=None)
        self.counter = counter
        self.allow = allow

    def expired(self):
        return self.counter["done"] >= self.allow

    def would_exceed(self, estimated_seconds=0.0):
        return self.expired()

    def describe(self):
        return f"stub budget ({self.counter['done']}/{self.allow} shards)"


def test_run_sharded_stops_at_the_cutoff_then_resumes(tmp_path):
    rows = make_rows(9)
    shards = plan_shards(rows, batch_size=3)
    store = store_at(tmp_path)

    processed = []
    counter = {"done": 0}

    def process(shard):
        processed.extend(r["record_id"] for r in shard)
        counter["done"] += 1
        return [{"record_id": r["record_id"]} for r in shard]

    finished = run_sharded(
        store, "u", shards, process, BudgetAfter(counter, allow=1)
    )
    assert finished is False
    assert len(processed) == 3
    assert len(store.completed_shards("u")) == 1

    processed.clear()
    assert run_sharded(store, "u", shards, process, Deadline()) is True
    # Only the two remaining shards; the committed one is not redone.
    assert len(processed) == 6

    order = {row["record_id"]: i for i, row in enumerate(rows)}
    merged = store.merge_unit("u", order=order)
    ids = [r["record_id"] for r in merged]
    assert ids == [r["record_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_run_sharded_does_nothing_when_already_over_budget(tmp_path):
    rows = make_rows(6)
    shards = plan_shards(rows, batch_size=2)
    store = store_at(tmp_path)
    calls = []

    def process(shard):
        calls.append(shard)
        return []

    assert run_sharded(
        store, "u", shards, process, Deadline(minutes=-1)
    ) is False
    assert calls == []


def test_run_sharded_survives_a_new_store_instance(tmp_path):
    # Mirrors a real session boundary: fresh process, fresh ShardStore.
    rows = make_rows(6)
    shards = plan_shards(rows, batch_size=2)
    counter = {"done": 0}

    def process(shard):
        counter["done"] += 1
        return [{"record_id": r["record_id"]} for r in shard]

    run_sharded(
        store_at(tmp_path),
        "u",
        shards,
        process,
        BudgetAfter(counter, allow=1),
    )

    reopened = store_at(tmp_path)
    assert len(reopened.completed_shards("u")) == 1
    assert run_sharded(reopened, "u", shards, process, Deadline()) is True
    assert reopened.unit_complete("u")


def test_shard_write_is_atomic_and_leaves_no_tmp(tmp_path):
    store = store_at(tmp_path)
    store.declare_unit("u", n_shards=1, n_rows=1)
    path = store.write_shard("u", 0, [{"record_id": "r000"}])

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == [
        {"record_id": "r000"}
    ]
    assert list(store.parts_dir.glob("*.tmp")) == []


def test_summary_reports_fraction(tmp_path):
    store = store_at(tmp_path)
    store.declare_unit("u", n_shards=4, n_rows=8)
    store.write_shard("u", 0, [])

    summary = store.summary()
    assert summary == [
        {
            "unit": "u",
            "shards_done": 1,
            "shards_total": 4,
            "fraction": 0.25,
            "complete": False,
        }
    ]


# ---- Deadline -----------------------------------------------------------


def test_deadline_none_is_unlimited():
    deadline = Deadline()
    assert deadline.expired() is False
    assert deadline.remaining_seconds == float("inf")
    assert deadline.would_exceed(10_000) is False
    assert deadline.describe() == "no deadline"


def test_deadline_expires_and_predicts_overrun():
    spent = Deadline(minutes=-1)
    assert spent.expired() is True

    budget = Deadline(minutes=60)
    assert budget.expired() is False
    assert budget.would_exceed(30) is False
    assert budget.would_exceed(60 * 60 + 10) is True


def test_deadline_check_raises_only_when_over():
    Deadline(minutes=60).check(1.0)

    with pytest.raises(Exception):
        Deadline(minutes=-1).check()
