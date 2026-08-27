"""End-to-end proof that shard-level T4 resumability actually works.

CPU-only and fully deterministic: a fake monotonic clock stands in for
wall-clock time so a deliberately short deadline trips at an exact,
reproducible shard boundary rather than depending on how fast this
machine happens to run. Exercises the real production entry point
(stage_behavior) end to end, not just the generic ShardStore/run_sharded
primitives already covered in test_v2_shards.py and test_v2_pipeline.py -
this is the proof that those primitives are wired up correctly for an
actual interrupted-and-resumed Colab session, covering (in one continuous
run):

1. A synthetic/bound execution with a deliberately short deadline.
2. Clean termination at a shard boundary.
3. Completed shards preserved on disk.
4. Resume, in a fresh session pointed at the same output directory.
5. Completed shards are not recomputed.
6. The merged result is byte-identical to an uninterrupted equivalent run.
7. Benchmark/split binding remains valid (and correctly rejects a resume
   against a mismatched benchmark - a check that does nothing is not a
   check).
8. Status before and after resume correctly reflects progress, including
   through the real print_status()/`status` command output.

stage_extract (forward-only activation extraction) is deliberately NOT
covered here: per its own inline comment and the Milestone 2B commit that
introduced stage_start_blocked, it is atomic rather than sharded/
checkpointed - a stage either completes in one pass or is not started at
all, guarded by a pre-stage deadline estimate (already covered by
test_v2_pipeline_deadline.py). Shard-level resume only applies to the
generative/paired-condition stages (stage_behavior, stage_causal,
stage_steering), which all share the same ShardStore/run_sharded/
plan_shards machinery proven here through stage_behavior.
"""

from __future__ import annotations

import time

import pytest

from src.analysis import v2_pipeline as vp
from src.analysis.v2_shards import Deadline, ShardStore
from src.v2_io import assert_binding, load_json


BENCH_SHA = "c" * 64
SPLIT_SHA = "d" * 64
OTHER_BENCH_SHA = "e" * 64


class FakeClock:
    """A controllable stand-in for time.monotonic().

    A deterministic test should never depend on how fast the test host
    happens to run; advancing this by a fixed amount per simulated shard
    makes "a deliberately short deadline trips after exactly N shards" an
    exact, reproducible fact rather than a race against real wall time.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_rows(n):
    return [
        {
            "record_id": f"r{i:03d}",
            "prompt": " ".join(["w"] * (i + 1)) + f" idx{i}",
            "quadrant": "A",
            "source_dataset": "unit_test",
            "split": "direction_estimation",
        }
        for i in range(n)
    ]


def make_ctx(
    root,
    rows,
    deadline,
    gen_batch=3,
    benchmark_sha=BENCH_SHA,
    split_sha=SPLIT_SHA,
    paths_root=None,
):
    """`paths_root` lets a caller point output at a different directory
    while keeping benchmark_path/split_path (and hence the binding
    sidecar's recorded paths) identical - used to build a fair
    "uninterrupted equivalent" comparison run: same benchmark identity,
    only the shard/session history differs.
    """
    return vp.RunContext(
        benchmark_path=root / "bench.jsonl",
        benchmark_sha=benchmark_sha,
        split_path=root / "split.json",
        split_sha=split_sha,
        rows=rows,
        paths=vp.ArtifactPaths((paths_root or root) / "results"),
        deadline=deadline,
        gen_batch=gen_batch,
    )


def _idx_of(prompt: str) -> int:
    return int(prompt.rsplit("idx", 1)[-1])


def test_shard_level_resumability_proof(tmp_path, monkeypatch):
    """The 8-point proof the milestone asks for, in one continuous run."""
    clock = FakeClock(0.0)
    monkeypatch.setattr(time, "monotonic", clock)

    rows = make_rows(9)  # gen_batch=3 -> exactly 3 shards
    calls: list[list[str]] = []
    SHARD_SECONDS = 100.0

    def fake_generation_batch(model, tokenizer, prompts, device, max_new_tokens):
        calls.append(list(prompts))
        clock.advance(SHARD_SECONDS)
        return [f"resp_{_idx_of(p)}" for p in prompts]

    monkeypatch.setattr(vp, "generation_batch", fake_generation_batch)

    session_root = tmp_path / "session"
    unit_key = ShardStore.unit_key("M3", "M3_behavior")

    # ---- 1. Deliberately short deadline --------------------------------
    # 2.5 min = 150s budget: shard 0 (100s) fits, but the estimate for
    # shard 1 (the running average of this unit's own observed shards,
    # i.e. also ~100s) would leave only 50s remaining - must stop before
    # starting it, per run_sharded's would_exceed check.
    ctx1 = make_ctx(session_root, rows, Deadline(minutes=2.5))
    finished = vp.stage_behavior(
        ctx1, "M3", model=None, tokenizer=object(), device="cpu"
    )

    # ---- 2. Clean termination at a shard boundary ----------------------
    assert finished is False
    assert len(calls) == 1  # exactly one shard's worth of work done
    assert len(calls[0]) == 3
    output_path = ctx1.paths.behavioral / "v2_raw_M3.json"
    assert not output_path.exists(), (
        "no merged output should exist until every shard is committed"
    )

    # ---- 3. Completed shards preserved on disk --------------------------
    store1 = ctx1.store(ctx1.paths.behavior_shards)
    assert store1.completed_shards(unit_key) == {0}
    assert store1.shard_path(unit_key, 0).exists()

    # ---- 8a. Status before resume reflects partial progress -------------
    [progress] = [row for row in store1.summary() if row["unit"] == unit_key]
    assert progress["shards_done"] == 1
    assert progress["shards_total"] == 3
    assert progress["fraction"] == round(1 / 3, 4)
    assert progress["complete"] is False

    # ---- 4. Resume: same output directory, a fresh session's budget -----
    ctx2 = make_ctx(session_root, rows, Deadline(minutes=None))
    finished = vp.stage_behavior(
        ctx2, "M3", model=None, tokenizer=object(), device="cpu"
    )
    assert finished is True

    # ---- 5. Completed shards were not recomputed -------------------------
    # 3 shards total, 1 already done before resume -> exactly 2 more
    # calls, and the resumed session never touched shard 0's prompts.
    assert len(calls) == 1 + 2
    resumed_prompts = {p for batch in calls[1:] for p in batch}
    assert resumed_prompts.isdisjoint(set(calls[0]))

    # ---- 8b. Status after resume reflects completion ----------------------
    store2 = ctx2.store(ctx2.paths.behavior_shards)
    [progress] = [row for row in store2.summary() if row["unit"] == unit_key]
    assert progress["shards_done"] == 3
    assert progress["shards_total"] == 3
    assert progress["complete"] is True
    assert output_path.exists()

    # ---- 7. Benchmark/split binding remains valid --------------------------
    binding_path = ctx2.paths.behavioral / "v2_raw_M3_binding.json"
    assert_binding(binding_path, BENCH_SHA, SPLIT_SHA)  # must not raise

    # ---- 6. Merged result byte-identical to an uninterrupted run -----------
    # Same benchmark identity (same benchmark_path/split_path/shas) as
    # ctx1/ctx2, so the binding sidecar records the same paths too - only
    # the output directory and the shard/session history differ.
    uninterrupted_root = tmp_path / "uninterrupted"
    ctx3 = make_ctx(
        session_root, rows, Deadline(minutes=None), paths_root=uninterrupted_root
    )
    calls_before_uninterrupted = len(calls)
    finished = vp.stage_behavior(
        ctx3, "M3", model=None, tokenizer=object(), device="cpu"
    )
    assert finished is True
    assert len(calls) == calls_before_uninterrupted + 3  # one shot, 3 shards

    resumed_output = output_path.read_bytes()
    uninterrupted_output = (
        ctx3.paths.behavioral / "v2_raw_M3.json"
    ).read_bytes()
    assert resumed_output == uninterrupted_output

    resumed_binding = binding_path.read_bytes()
    uninterrupted_binding = (
        ctx3.paths.behavioral / "v2_raw_M3_binding.json"
    ).read_bytes()
    assert resumed_binding == uninterrupted_binding

    # Sanity: the content itself is correct, not just self-consistent.
    merged = load_json(output_path)
    assert [row["record_id"] for row in merged] == [
        row["record_id"] for row in rows
    ]
    assert [row["response"] for row in merged] == [
        f"resp_{i}" for i in range(len(rows))
    ]


def test_resume_against_a_different_benchmark_raises(tmp_path, monkeypatch):
    """Resuming shard progress bound to one benchmark against a different
    one must refuse outright, never silently mix rows from two eval sets.
    """
    clock = FakeClock(0.0)
    monkeypatch.setattr(time, "monotonic", clock)

    rows = make_rows(6)

    def fake_generation_batch(model, tokenizer, prompts, device, max_new_tokens):
        clock.advance(50.0)
        return [f"resp_{_idx_of(p)}" for p in prompts]

    monkeypatch.setattr(vp, "generation_batch", fake_generation_batch)

    root = tmp_path / "session"
    # Deliberately short deadline again, so at least one shard is
    # committed to disk (gen_batch=2 -> 3 shards of 50s each; a 60s
    # budget fits shard 0 but not shard 1) before the "session" ends.
    ctx1 = make_ctx(root, rows, Deadline(minutes=1.0), gen_batch=2)
    finished = vp.stage_behavior(
        ctx1, "M3", model=None, tokenizer=object(), device="cpu"
    )
    assert finished is False

    ctx_other_benchmark = make_ctx(
        root,
        rows,
        Deadline(minutes=None),
        gen_batch=2,
        benchmark_sha=OTHER_BENCH_SHA,
    )
    with pytest.raises(RuntimeError, match="different benchmark"):
        vp.stage_behavior(
            ctx_other_benchmark,
            "M3",
            model=None,
            tokenizer=object(),
            device="cpu",
        )


def test_status_before_and_after_resume_via_print_status(
    tmp_path, monkeypatch, capsys
):
    """The actual `status` command output, not just the underlying data."""
    clock = FakeClock(0.0)
    monkeypatch.setattr(time, "monotonic", clock)

    rows = make_rows(6)

    def fake_generation_batch(model, tokenizer, prompts, device, max_new_tokens):
        clock.advance(50.0)
        return [f"resp_{_idx_of(p)}" for p in prompts]

    monkeypatch.setattr(vp, "generation_batch", fake_generation_batch)

    root = tmp_path / "session"
    # 60s budget, 50s/shard, 3 shards -> exactly 1 completes before stop.
    ctx1 = make_ctx(root, rows, Deadline(minutes=1.0), gen_batch=2)
    finished = vp.stage_behavior(
        ctx1, "M3", model=None, tokenizer=object(), device="cpu"
    )
    assert finished is False

    capsys.readouterr()
    vp.print_status(ctx1)
    before = capsys.readouterr().out
    assert "behavioral complete: []" in before
    assert "M3__M3_behavior: 1/3 shards" in before

    ctx2 = make_ctx(root, rows, Deadline(minutes=None), gen_batch=2)
    finished = vp.stage_behavior(
        ctx2, "M3", model=None, tokenizer=object(), device="cpu"
    )
    assert finished is True

    vp.print_status(ctx2)
    after = capsys.readouterr().out
    assert "behavioral complete: ['M3']" in after
    assert "in-flight behavior units" not in after
