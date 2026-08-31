"""
Focused tests for src/data_pipeline/build_r104_human_review_packet.py.

Per the R104 human-review task brief, this does not rerun any prior
statistical audit and does not run the broad historical suite. Mirrors
the convention in tests/data_pipeline/test_blind_construct_check_3d_h.py:
a handful of unit tests on pure logic (review-ID assignment, presentation
order, provenance forbidden-string checks, private-key path safety) plus
integration checks against the real frozen repository artifacts, since
the task brief specifically requires verifying the eligible population
against current files rather than a historical count.
"""
import json

import numpy as np
import pytest

from src.data_pipeline.blind_construct_check_3d_h import validate_private_key_path
from src.data_pipeline.build_r104_human_review_packet import (
    CSV_FIELDNAMES,
    PRESENTATION_SEED,
    REPO_ROOT,
    InvariantError,
    assign_review_ids,
    build_provenance,
    compute_presentation_order,
    load_eligible_population,
)


# ── review-ID assignment ────────────────────────────────────────────────
def test_assign_review_ids_is_sorted_and_sequential():
    population = {"zzz": {}, "aaa": {}, "mmm": {}}
    pairs = assign_review_ids(population)
    assert [rid for rid, _ in pairs] == ["R001", "R002", "R003"]
    assert [record_id for _, record_id in pairs] == ["aaa", "mmm", "zzz"]


def test_assign_review_ids_covers_every_record_exactly_once():
    population = {f"rec_{i}": {} for i in range(37)}
    pairs = assign_review_ids(population)
    assert len(pairs) == 37
    assert {rid for _, rid in pairs} == set(population.keys())
    assert len({review_id for review_id, _ in pairs}) == 37


# ── presentation order ──────────────────────────────────────────────────
def test_presentation_order_is_deterministic_and_a_permutation():
    pairs = [(f"R{i:03d}", f"rec_{i}") for i in range(20)]
    p1 = compute_presentation_order(pairs)
    p2 = compute_presentation_order(pairs)
    assert p1 == p2
    assert sorted(p1) == sorted(pairs)
    assert p1 != pairs  # astronomically unlikely to be identity at n=20


def test_presentation_order_uses_the_module_seed():
    pairs = [(f"R{i:03d}", f"rec_{i}") for i in range(10)]
    expected_idx = np.random.default_rng(PRESENTATION_SEED).permutation(len(pairs))
    expected = [pairs[i] for i in expected_idx]
    assert compute_presentation_order(pairs) == expected


# ── provenance safety ────────────────────────────────────────────────────
def test_build_provenance_never_contains_forbidden_strings():
    prov = build_provenance(
        n_total=104,
        input_hashes={
            "review_queue_csv_sha256": "x",
            "frozen_benchmark_path": "data/frozen_v2/whatever.jsonl",
            "frozen_benchmark_sha256": "y",
        },
        blind_csv_sha256="csvhash",
        instructions_sha256="instrhash",
        answer_key_sha256="keyhash",
        generation_commit="deadbeef",
    )
    serialized = json.dumps(prov)
    for forbidden in ("StrongREJECT", "HarmBench", "quadrant", "Quadrant"):
        assert forbidden not in serialized
    assert prov["n_eligible_pairs"] == 104
    assert prov["presentation_seed"] == PRESENTATION_SEED
    assert prov["selection_seed"] is None


def test_build_provenance_contains_no_row_level_mapping():
    prov = build_provenance(
        n_total=3,
        input_hashes={
            "review_queue_csv_sha256": "x",
            "frozen_benchmark_path": "p",
            "frozen_benchmark_sha256": "y",
        },
        blind_csv_sha256="csvhash",
        instructions_sha256="instrhash",
        answer_key_sha256="keyhash",
        generation_commit="deadbeef",
    )
    # No review_id/record_id pair-level keys anywhere in the structure.
    assert "review_id_pairs" not in prov
    assert "record_ids" not in prov
    assert "mapping" not in prov


# ── private-key path safety (reused from the 3D-H module) ──────────────
def test_validate_private_key_path_rejects_inside_repo():
    with pytest.raises(SystemExit):
        validate_private_key_path(str(REPO_ROOT / "logs" / "leak.json"))


def test_validate_private_key_path_accepts_outside_repo(tmp_path):
    outside = tmp_path / "answer_key.json"
    resolved = validate_private_key_path(str(outside))
    assert resolved == outside.resolve()


# ── integration checks against the real frozen repository artifacts ────
def test_real_frozen_inputs_yield_104_eligible_pairs():
    population, hashes = load_eligible_population()
    assert len(population) == 104
    assert len(set(population.keys())) == 104
    pair_ids = [v["pair_id"] for v in population.values()]
    assert len(set(pair_ids)) == 104
    assert set(hashes.keys()) == {
        "review_queue_csv_sha256",
        "frozen_benchmark_path",
        "frozen_benchmark_sha256",
    }


def test_real_frozen_inputs_full_pipeline_is_reproducible():
    population, _ = load_eligible_population()
    pairs1 = assign_review_ids(population)
    pairs2 = assign_review_ids(population)
    assert pairs1 == pairs2

    presented1 = compute_presentation_order(pairs1)
    presented2 = compute_presentation_order(pairs2)
    assert presented1 == presented2
    assert sorted(presented1) == sorted(pairs1)


def test_real_frozen_inputs_csv_fieldnames_match_required_review_data():
    assert CSV_FIELDNAMES == [
        "review_id",
        "source_prompt",
        "candidate_prompt",
        "decision",
        "notes",
    ]
