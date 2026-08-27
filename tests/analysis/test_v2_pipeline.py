"""Pipeline-level tests for the throughput/reproducibility machinery.

tests/analysis/test_v2_shards.py already covers plan_shards/ShardStore in
isolation. These tests exercise the real production entry points
(stage_extract, stage_behavior, resolve_batch_size) with the model/tokenizer
calls mocked out, so a regression in how v2_pipeline.py *wires up*
length-sorted batching, order restoration, record_id identity, or OOM
backoff would be caught here even if the underlying utilities in
v2_shards.py still pass their own tests.

No torch/GPU is required: activation_batch and generation_batch are
monkeypatched, and the tokenizer passed in is a plain object so
token_measure() falls back to its word-count path.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import v2_pipeline as vp
from src.analysis.v2_shards import Deadline
from src.v2_io import load_json


BENCH_SHA = "a" * 64
SPLIT_SHA = "b" * 64


def make_rows(n):
    """Rows with strictly increasing prompt length, indexed by record_id.

    The word count increases with i, so length-sorted batching visits
    them in the same order as record_id here - which is exactly what lets
    the tests below tell "processed out of order, restored correctly"
    apart from "never reordered in the first place".
    """
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


def make_ctx(tmp_path, rows, act_batch=2, gen_batch=2):
    return vp.RunContext(
        benchmark_path=tmp_path / "bench.jsonl",
        benchmark_sha=BENCH_SHA,
        split_path=tmp_path / "split.json",
        split_sha=SPLIT_SHA,
        rows=rows,
        paths=vp.ArtifactPaths(tmp_path / "results"),
        deadline=Deadline(None),
        act_batch=act_batch,
        gen_batch=gen_batch,
    )


def _idx_of(prompt: str) -> int:
    return int(prompt.rsplit("idx", 1)[-1])


# ---- stage_extract (forward-only path) -----------------------------------


def test_stage_extract_restores_benchmark_order_and_record_id(
    tmp_path, monkeypatch
):
    rows = make_rows(7)
    ctx = make_ctx(tmp_path, rows, act_batch=2)

    seen_batches: list[list[str]] = []

    def fake_activation_batch(model, tokenizer, prompts, device):
        seen_batches.append(list(prompts))
        n = len(prompts)
        final = np.zeros((n, 2, 3), dtype=np.float32)
        pooled = np.zeros((n, 2, 3), dtype=np.float32)
        for row_index, prompt in enumerate(prompts):
            final[row_index, :, :] = _idx_of(prompt)
            pooled[row_index, :, :] = _idx_of(prompt) * 100
        return final, pooled

    monkeypatch.setattr(vp, "activation_batch", fake_activation_batch)

    assert vp.stage_extract(ctx, "M3", model=None, tokenizer=object(), device="cpu")

    # Length sorting: batches were dispatched shortest-prompt-first (word
    # count is monotonic with the encoded index by construction).
    flat_indices = [_idx_of(p) for batch in seen_batches for p in batch]
    assert flat_indices == sorted(flat_indices)
    # Every shard is <= act_batch rows.
    assert all(len(batch) <= ctx.act_batch for batch in seen_batches)

    final_path, pooled_path, metadata_path, binding_path = (
        vp.activation_paths(ctx, "M3")
    )

    final = np.load(final_path)
    pooled = np.load(pooled_path)

    # Order restoration + record_id preservation: row i of the saved array
    # must correspond to ctx.rows[i], regardless of how shards were batched.
    assert final.shape[0] == len(rows)
    for i in range(len(rows)):
        assert final[i, 0, 0] == i
        assert pooled[i, 0, 0] == i * 100

    metadata = load_json(metadata_path)
    assert [row["record_id"] for row in metadata] == [
        row["record_id"] for row in rows
    ]

    binding = load_json(binding_path)
    assert binding["batching"] == "length_sorted_restored_to_benchmark_order"
    assert binding["benchmark_sha256"] == BENCH_SHA
    assert binding["split_manifest_sha256"] == SPLIT_SHA


def test_stage_extract_survives_an_oom_shard(tmp_path, monkeypatch):
    """A shard that OOMs at act_batch is retried smaller, not lost."""
    rows = make_rows(6)
    ctx = make_ctx(tmp_path, rows, act_batch=2)

    def fake_activation_batch(model, tokenizer, prompts, device):
        if len(prompts) > 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        n = len(prompts)
        final = np.full((n, 1, 1), _idx_of(prompts[0]), dtype=np.float32)
        pooled = final.copy()
        return final, pooled

    monkeypatch.setattr(vp, "activation_batch", fake_activation_batch)

    assert vp.stage_extract(ctx, "M3", model=None, tokenizer=object(), device="cpu")

    final_path, _, metadata_path, _ = vp.activation_paths(ctx, "M3")
    final = np.load(final_path)
    metadata = load_json(metadata_path)

    assert final.shape[0] == len(rows)
    assert [row["record_id"] for row in metadata] == [
        row["record_id"] for row in rows
    ]
    for i in range(len(rows)):
        assert final[i, 0, 0] == i


# ---- stage_behavior (generative path) -------------------------------------


def test_stage_behavior_restores_benchmark_order_and_record_id(
    tmp_path, monkeypatch
):
    rows = make_rows(5)
    ctx = make_ctx(tmp_path, rows, gen_batch=2)

    def fake_generation_batch(model, tokenizer, prompts, device, max_new_tokens):
        return [f"resp_{_idx_of(p)}" for p in prompts]

    monkeypatch.setattr(vp, "generation_batch", fake_generation_batch)

    assert vp.stage_behavior(ctx, "M3", model=None, tokenizer=object(), device="cpu")

    output_path = ctx.paths.behavioral / "v2_raw_M3.json"
    output = load_json(output_path)

    assert [row["record_id"] for row in output] == [
        row["record_id"] for row in rows
    ]
    for i, row in enumerate(output):
        assert row["response"] == f"resp_{i}"
        assert row["record_id"] == rows[i]["record_id"]


def test_stage_behavior_survives_an_oom_shard(tmp_path, monkeypatch):
    rows = make_rows(5)
    ctx = make_ctx(tmp_path, rows, gen_batch=3)

    def fake_generation_batch(model, tokenizer, prompts, device, max_new_tokens):
        if len(prompts) > 1:
            raise RuntimeError("CUDA out of memory")
        return [f"resp_{_idx_of(p)}" for p in prompts]

    monkeypatch.setattr(vp, "generation_batch", fake_generation_batch)

    assert vp.stage_behavior(ctx, "M3", model=None, tokenizer=object(), device="cpu")

    output = load_json(ctx.paths.behavioral / "v2_raw_M3.json")
    assert [row["record_id"] for row in output] == [
        row["record_id"] for row in rows
    ]
    assert [row["response"] for row in output] == [
        f"resp_{i}" for i in range(len(rows))
    ]


def test_stage_behavior_is_idempotent_once_bound(tmp_path, monkeypatch):
    rows = make_rows(3)
    ctx = make_ctx(tmp_path, rows, gen_batch=2)

    calls = []

    def fake_generation_batch(model, tokenizer, prompts, device, max_new_tokens):
        calls.append(list(prompts))
        return [f"resp_{_idx_of(p)}" for p in prompts]

    monkeypatch.setattr(vp, "generation_batch", fake_generation_batch)

    assert vp.stage_behavior(ctx, "M3", model=None, tokenizer=object(), device="cpu")
    n_calls_first_pass = len(calls)
    assert n_calls_first_pass > 0

    # Re-running against the same bound benchmark must not regenerate.
    assert vp.stage_behavior(ctx, "M3", model=None, tokenizer=object(), device="cpu")
    assert len(calls) == n_calls_first_pass


# ---- resolve_batch_size / batch_size_arg ("auto" batch selection) --------


def test_batch_size_arg_parses_auto_and_int():
    assert vp.batch_size_arg("auto") == "auto"
    assert vp.batch_size_arg("16") == 16


def test_resolve_batch_size_passes_through_non_auto_values(tmp_path):
    missing = tmp_path / "t4_calibration.json"
    assert vp.resolve_batch_size(8, "recommended_act_batch", missing) == 8


def test_resolve_batch_size_reads_recommendation_from_calibration_file(
    tmp_path,
):
    from src.v2_io import write_json_lf

    calibration_path = tmp_path / "t4_calibration.json"
    write_json_lf(
        calibration_path,
        {"recommended_act_batch": 24, "recommended_gen_batch": 12},
    )

    assert (
        vp.resolve_batch_size(
            "auto", "recommended_act_batch", calibration_path
        )
        == 24
    )
    assert (
        vp.resolve_batch_size(
            "auto", "recommended_gen_batch", calibration_path
        )
        == 12
    )


def test_resolve_batch_size_auto_without_calibration_file_raises(tmp_path):
    missing = tmp_path / "t4_calibration.json"
    with pytest.raises(RuntimeError, match="requires"):
        vp.resolve_batch_size("auto", "recommended_act_batch", missing)


def test_resolve_batch_size_auto_without_recorded_value_raises(tmp_path):
    from src.v2_io import write_json_lf

    calibration_path = tmp_path / "t4_calibration.json"
    write_json_lf(calibration_path, {"recommended_act_batch": None})

    with pytest.raises(RuntimeError, match="recommended_act_batch"):
        vp.resolve_batch_size(
            "auto", "recommended_act_batch", calibration_path
        )
