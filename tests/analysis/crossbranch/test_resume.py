"""Resumability.

Two classes of guarantee, kept explicitly separate because conflating them
would overstate what is verified here:

ALREADY GUARANTEED by src/analysis/v2_shards.py -- tested as integration, to
confirm this extension composes with it correctly:
  * atomic shard writes (.tmp then os.replace)
  * skipping only shards that are complete AND present on disk
  * refusing to resume across a different benchmark/split
  * detecting a changed shard plan

GENUINELY NEW here, because ShardStore.merge_unit does NOT provide it: it
sorts unknown record_ids LAST rather than raising, and performs no duplicate
detection. Exact record-ID coverage, duplicate failure and missing failure
are therefore our own asserts in crossbranch.analyze.
"""
import json

import pytest

from src.analysis.crossbranch.analyze import assert_shared_rows, label_map
from src.analysis.v2_shards import Deadline, ShardStore, plan_shards, run_sharded

BENCH, SPLIT = "bench" * 8, "split" * 8
COMPLY = "Sure! First, gather your ingredients and preheat the oven to 350 degrees."


def rows(n=10):
    return [{"record_id": f"r{i}", "prompt": f"prompt number {i}"} for i in range(n)]


def store(tmp_path):
    return ShardStore(tmp_path / "shards", BENCH, SPLIT)


def responder(shard):
    return [{"record_id": r["record_id"], "response": COMPLY} for r in shard]


# ---- already guaranteed by v2_shards (integration) ------------------------


def test_shard_writes_are_atomic_and_leave_no_tmp_behind(tmp_path):
    s = store(tmp_path)
    shards = plan_shards(rows(6), 2)
    run_sharded(s, "u", shards, responder, Deadline(None))
    parts = s.parts_dir
    assert not list(parts.glob("*.tmp")), "a .tmp must never survive a completed write"
    assert len(list(parts.glob("*.json"))) == len(shards)


def test_a_stray_tmp_file_is_never_merged(tmp_path):
    s = store(tmp_path)
    shards = plan_shards(rows(4), 2)
    run_sharded(s, "u", shards, responder, Deadline(None))
    (s.parts_dir / "u__0099.json.tmp").write_text('[{"record_id": "ghost"}]', "utf-8")
    merged = s.merge_unit("u")
    assert all(r["record_id"] != "ghost" for r in merged)


def test_only_complete_and_present_shards_are_skipped(tmp_path):
    s = store(tmp_path)
    shards = plan_shards(rows(6), 2)
    run_sharded(s, "u", shards, responder, Deadline(None))
    assert s.unit_complete("u")

    # A lost Drive sync: progress.json still records the shard, the file is gone.
    s.shard_path("u", 1).unlink()
    assert not s.unit_complete("u")
    assert 1 not in s.completed_shards("u")


def test_resume_across_a_different_benchmark_is_refused(tmp_path):
    s = store(tmp_path)
    run_sharded(s, "u", plan_shards(rows(4), 2), responder, Deadline(None))
    with pytest.raises(RuntimeError, match="Refusing to resume"):
        ShardStore(tmp_path / "shards", "different" * 4, SPLIT)


def test_a_changed_shard_plan_is_refused(tmp_path):
    s = store(tmp_path)
    run_sharded(s, "u", plan_shards(rows(6), 2), responder, Deadline(None))
    s2 = ShardStore(tmp_path / "shards", BENCH, SPLIT)
    with pytest.raises(RuntimeError, match="shard plan changed"):
        run_sharded(s2, "u", plan_shards(rows(6), 3), responder, Deadline(None))


# ---- simulated interruption, then exact resume ---------------------------


def test_interruption_then_resume_reproduces_the_uninterrupted_output(tmp_path):
    data = rows(10)
    shards = plan_shards(data, 3)

    # Uninterrupted reference run.
    ref_store = ShardStore(tmp_path / "ref", BENCH, SPLIT)
    run_sharded(ref_store, "u", shards, responder, Deadline(None))
    order = {r["record_id"]: i for i, r in enumerate(data)}
    reference = ref_store.merge_unit("u", order=order)

    # Interrupted run: shard index 2 dies mid-unit.
    s = store(tmp_path)
    calls = {"n": 0}

    def flaky(shard):
        if calls["n"] == 2:
            calls["n"] += 1
            raise RuntimeError("simulated session kill")
        calls["n"] += 1
        return responder(shard)

    with pytest.raises(RuntimeError, match="simulated session kill"):
        run_sharded(s, "u", shards, flaky, Deadline(None))

    assert not s.unit_complete("u")
    partial = len(s.completed_shards("u"))
    assert 0 < partial < len(shards)

    # Resume in a fresh store object, as a new session would.
    s2 = ShardStore(tmp_path / "shards", BENCH, SPLIT)
    finished = run_sharded(s2, "u", shards, responder, Deadline(None))
    assert finished and s2.unit_complete("u")

    resumed = s2.merge_unit("u", order=order)
    assert resumed == reference
    assert [r["record_id"] for r in resumed] == [r["record_id"] for r in data]


def test_resume_does_not_redo_already_committed_shards(tmp_path):
    data = rows(9)
    shards = plan_shards(data, 3)
    s = store(tmp_path)
    seen = []

    def counting(shard):
        seen.append(tuple(r["record_id"] for r in shard))
        return responder(shard)

    run_sharded(s, "u", shards[:2], counting, Deadline(None))
    first_pass = len(seen)

    s2 = ShardStore(tmp_path / "shards", BENCH, SPLIT)
    # declare_unit guards the plan, so the resumed run must use the same plan
    with pytest.raises(RuntimeError, match="shard plan changed"):
        run_sharded(s2, "u", shards, counting, Deadline(None))
    assert len(seen) == first_pass, "no extra work before the guard fired"


# ---- our own coverage asserts (merge_unit does not provide these) ---------


def test_merge_unit_does_not_itself_reject_a_foreign_record_id(tmp_path):
    """Documents exactly why the coverage assert in analyze.py has to exist."""
    s = store(tmp_path)
    s.declare_unit("u", n_shards=1, n_rows=2)
    s.write_shard("u", 0, [{"record_id": "r0"}, {"record_id": "ghost"}])
    merged = s.merge_unit("u", order={"r0": 0})
    assert [r["record_id"] for r in merged] == ["r0", "ghost"]  # sorted last, kept


def test_our_coverage_assert_rejects_a_missing_record_id():
    labels = {
        "cond_a": {"r0": "comply", "r1": "comply"},
        "cond_b": {"r0": "comply"},
    }
    with pytest.raises(RuntimeError, match="identical record_id set"):
        assert_shared_rows(labels)


def test_our_coverage_assert_rejects_a_foreign_record_id():
    labels = {
        "cond_a": {"r0": "comply"},
        "cond_b": {"r0": "comply", "ghost": "comply"},
    }
    with pytest.raises(RuntimeError, match="identical record_id set"):
        assert_shared_rows(labels)


def test_our_duplicate_assert_rejects_a_repeated_record_id():
    duplicated = [
        {"record_id": "r0", "response": COMPLY},
        {"record_id": "r0", "response": COMPLY},
    ]
    with pytest.raises(RuntimeError, match="duplicate record_id"):
        label_map(duplicated)


def test_coverage_assert_accepts_an_exactly_matching_set():
    labels = {
        "cond_a": {"r0": "comply", "r1": "refusal"},
        "cond_b": {"r1": "comply", "r0": "refusal"},
    }
    assert assert_shared_rows(labels) == ["r0", "r1"]


def test_progress_file_is_valid_json_after_a_run(tmp_path):
    s = store(tmp_path)
    run_sharded(s, "u", plan_shards(rows(4), 2), responder, Deadline(None))
    data = json.loads(s.progress_path.read_text(encoding="utf-8"))
    assert data["benchmark_sha256"] == BENCH
    assert data["split_manifest_sha256"] == SPLIT
