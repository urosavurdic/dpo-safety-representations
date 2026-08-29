"""
Focused tests for src/data_pipeline/blind_construct_check_3d_h.py.

Per the 3D-H task brief (section 10), this does NOT rerun S2/S3, does NOT
regenerate grouping, and does NOT run the broad historical suite. Most
tests use small synthetic fixtures mirroring the convention already
established in tests/data_pipeline/test_lexical_outlierness.py; a handful
of integration checks run against the real frozen repository artifacts,
since the task brief specifically requires verifying "exact
population/group invariants" and "feasible k" against the actual frozen
inputs, not just synthetic stand-ins.
"""
import json

import numpy as np
import pytest

from src.data_pipeline.blind_construct_check_3d_h import (
    AUDIT_JSON_PATH,
    GROUPS_ARTIFACT_PATH,
    PILOT_JSON_PATH,
    REQUIRED_SOURCES,
    STRATUM_ORDER,
    InvariantError,
    assign_review_ids,
    assign_strata_from_p_tfidf,
    build_provenance,
    compute_presentation_order,
    select_blind_sample,
    validate_private_key_path,
    verify_frozen_invariants,
)
from src.data_pipeline.lexical_outlierness import REPO_ROOT, load_population


# ── synthetic fixtures ───────────────────────────────────────────────────
def _mk_pilot_row(source, p_tfidf, category="unknown", p_selfinfo=0.5):
    tail = "high" if p_tfidf >= 0.75 else ("low" if p_tfidf <= 0.25 else "mid")
    return {
        "source": source,
        "p_tfidf": p_tfidf,
        "p_selfinfo": p_selfinfo,
        "category": category,
        "tail_tfidf": tail,
    }


def _mk_pilot(row_level):
    return {"scoring": {"row_level": row_level}}


def _mk_groups(record_ids):
    return {"record_id_to_group_id": {rid: f"g_{rid}" for rid in record_ids}}


SR, SST = REQUIRED_SOURCES


# ── Section 1: invariants ────────────────────────────────────────────────
def test_verify_frozen_invariants_accepts_matching_population():
    row_level = {
        "a": _mk_pilot_row(SR, 0.9),
        "b": _mk_pilot_row(SST, 0.1),
    }
    pilot = _mk_pilot(row_level)
    groups = _mk_groups(["a", "b"])
    population_rows = [{"record_id": "a"}, {"record_id": "b"}]

    # Patch EXPECTED_POPULATION_SIZE via monkeypatch-free approach: call
    # the underlying assertions directly by using a population of size 2
    # and temporarily relying on the module constant only for the count
    # check -- so we assert the *shape* of the invariant logic using the
    # real function but with the module's constant overridden.
    import src.data_pipeline.blind_construct_check_3d_h as mod

    old_expected = mod.EXPECTED_POPULATION_SIZE
    old_pilot_path = mod.PILOT_JSON_PATH
    old_groups_path = mod.GROUPS_ARTIFACT_PATH
    old_audit_path = mod.AUDIT_JSON_PATH
    old_pop_path = mod.POPULATION_ARTIFACT_PATH
    try:
        mod.EXPECTED_POPULATION_SIZE = 2
        # verify_frozen_invariants only hashes files at the very end, via
        # module-level path constants; point them at real, readable files
        # (any two existing repo files) purely so file_sha256 succeeds.
        mod.PILOT_JSON_PATH = REPO_ROOT / "logs/3d_b_lexical_outlierness_pilot.json"
        mod.GROUPS_ARTIFACT_PATH = REPO_ROOT / "data/quadrant_c_pipeline/lexical_outlierness_groups_v1.json"
        mod.AUDIT_JSON_PATH = REPO_ROOT / "logs/3d_c_length_dependence_audit.json"
        mod.POPULATION_ARTIFACT_PATH = REPO_ROOT / "data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl"
        hashes = verify_frozen_invariants(pilot, groups, population_rows)
        assert set(hashes.keys()) == {
            "pilot_json_sha256",
            "audit_json_sha256",
            "population_sha256",
            "groups_sha256",
        }
    finally:
        mod.EXPECTED_POPULATION_SIZE = old_expected
        mod.PILOT_JSON_PATH = old_pilot_path
        mod.GROUPS_ARTIFACT_PATH = old_groups_path
        mod.AUDIT_JSON_PATH = old_audit_path
        mod.POPULATION_ARTIFACT_PATH = old_pop_path


def test_verify_frozen_invariants_rejects_group_mismatch():
    import src.data_pipeline.blind_construct_check_3d_h as mod

    row_level = {"a": _mk_pilot_row(SR, 0.9), "b": _mk_pilot_row(SST, 0.1)}
    pilot = _mk_pilot(row_level)
    groups = _mk_groups(["a", "c"])  # "c" doesn't exist in population -> mismatch
    population_rows = [{"record_id": "a"}, {"record_id": "b"}]

    old_expected = mod.EXPECTED_POPULATION_SIZE
    try:
        mod.EXPECTED_POPULATION_SIZE = 2
        with pytest.raises(SystemExit):
            verify_frozen_invariants(pilot, groups, population_rows)
    finally:
        mod.EXPECTED_POPULATION_SIZE = old_expected


def test_verify_frozen_invariants_rejects_wrong_population_count():
    import src.data_pipeline.blind_construct_check_3d_h as mod

    pilot = _mk_pilot({"a": _mk_pilot_row(SR, 0.9)})
    groups = _mk_groups(["a"])
    population_rows = [{"record_id": "a"}, {"record_id": "b"}]  # 2, expects 3

    old_expected = mod.EXPECTED_POPULATION_SIZE
    try:
        mod.EXPECTED_POPULATION_SIZE = 3
        with pytest.raises(SystemExit):
            verify_frozen_invariants(pilot, groups, population_rows)
    finally:
        mod.EXPECTED_POPULATION_SIZE = old_expected


# ── Section 2: p_tfidf-only stratification ──────────────────────────────
def test_assign_strata_uses_p_tfidf_only_and_drops_mid():
    row_level = {
        "hi_sr": _mk_pilot_row(SR, 0.75),      # exactly at cutoff -> high
        "hi_sst": _mk_pilot_row(SST, 0.9),
        "lo_sr": _mk_pilot_row(SR, 0.25),      # exactly at cutoff -> low
        "lo_sst": _mk_pilot_row(SST, 0.1),
        "mid": _mk_pilot_row(SR, 0.5),         # ineligible
    }
    buckets = assign_strata_from_p_tfidf(row_level)
    assert buckets[("high", SR)] == ["hi_sr"]
    assert buckets[("high", SST)] == ["hi_sst"]
    assert buckets[("low", SR)] == ["lo_sr"]
    assert buckets[("low", SST)] == ["lo_sst"]
    all_selected = [rid for v in buckets.values() for rid in v]
    assert "mid" not in all_selected


# ── Section 3: deterministic selection ──────────────────────────────────
def _synthetic_buckets(n_high_sr=3, n_high_sst=3, n_low_sr=5, n_low_sst=5):
    return {
        ("high", SR): [f"h_sr_{i}" for i in range(n_high_sr)],
        ("high", SST): [f"h_sst_{i}" for i in range(n_high_sst)],
        ("low", SR): [f"l_sr_{i}" for i in range(n_low_sr)],
        ("low", SST): [f"l_sst_{i}" for i in range(n_low_sst)],
    }


def test_select_blind_sample_is_deterministic():
    buckets = _synthetic_buckets()
    k1, sel1 = select_blind_sample(buckets)
    k2, sel2 = select_blind_sample(buckets)
    assert k1 == k2
    assert sel1 == sel2


def test_select_blind_sample_respects_smallest_stratum():
    # Smallest stratum has 3 rows -> k/2 <= 3 -> k <= 6.
    buckets = _synthetic_buckets(n_high_sr=3, n_high_sst=10, n_low_sr=10, n_low_sst=10)
    k, selected = select_blind_sample(buckets)
    assert k == 6
    for key in STRATUM_ORDER:
        assert len(selected[key]) == k // 2


def test_select_blind_sample_caps_at_20():
    buckets = _synthetic_buckets(n_high_sr=100, n_high_sst=100, n_low_sr=100, n_low_sst=100)
    k, selected = select_blind_sample(buckets)
    assert k == 20
    for key in STRATUM_ORDER:
        assert len(selected[key]) == 10


def test_select_blind_sample_fails_closed_when_infeasible():
    buckets = _synthetic_buckets(n_high_sr=0, n_high_sst=5, n_low_sr=5, n_low_sst=5)
    with pytest.raises(SystemExit):
        select_blind_sample(buckets)


def test_review_id_assignment_order_matches_stratum_order():
    buckets = _synthetic_buckets(n_high_sr=2, n_high_sst=2, n_low_sr=2, n_low_sst=2)
    k, selected = select_blind_sample(buckets)
    pairs = assign_review_ids(k, selected)
    ids_in_order = [rid for _, rid in pairs]
    expected_order = (
        selected[("high", SR)]
        + selected[("high", SST)]
        + selected[("low", SR)]
        + selected[("low", SST)]
    )
    assert ids_in_order == expected_order
    assert [rid for rid, _ in pairs] == [f"R{i+1:03d}" for i in range(len(pairs))]


def test_presentation_order_is_deterministic_and_a_permutation():
    pairs = [(f"R{i:03d}", f"rec_{i}") for i in range(10)]
    p1 = compute_presentation_order(pairs)
    p2 = compute_presentation_order(pairs)
    assert p1 == p2
    assert sorted(p1) == sorted(pairs)
    # presentation order should generally differ from selection order for
    # n=10 (astronomically unlikely to be identity by chance for this seed)
    assert p1 != pairs


# ── Section 6/8: provenance and answer-key path safety ──────────────────
def test_build_provenance_never_contains_source_strings():
    prov = build_provenance(
        k=16,
        unique_group_count=32,
        input_hashes={
            "pilot_json_sha256": "x",
            "audit_json_sha256": "y",
            "population_sha256": "z",
            "groups_sha256": "w",
        },
        blind_csv_sha256="csvhash",
        answer_key_sha256="keyhash",
        generation_commit="deadbeef",
    )
    serialized = json.dumps(prov)
    for forbidden in REQUIRED_SOURCES:
        assert forbidden not in serialized
    assert prov["number_of_sources"] == 2
    assert prov["high_tail_count_per_source"] == 8
    assert prov["low_tail_count_per_source"] == 8


def test_validate_private_key_path_rejects_inside_repo():
    with pytest.raises(SystemExit):
        validate_private_key_path(str(REPO_ROOT / "logs" / "leak.json"))


def test_validate_private_key_path_accepts_outside_repo(tmp_path):
    outside = tmp_path / "answer_key.json"
    resolved = validate_private_key_path(str(outside))
    assert resolved == outside.resolve()


# ── Integration checks against the real frozen repository artifacts ─────
def test_real_frozen_inputs_satisfy_all_invariants():
    pilot = json.loads(PILOT_JSON_PATH.read_text(encoding="utf-8"))
    groups = json.loads(GROUPS_ARTIFACT_PATH.read_text(encoding="utf-8"))
    population_rows = load_population()
    hashes = verify_frozen_invariants(pilot, groups, population_rows)
    assert all(isinstance(v, str) and len(v) == 64 for v in hashes.values())


def test_real_frozen_inputs_realize_k_16():
    pilot = json.loads(PILOT_JSON_PATH.read_text(encoding="utf-8"))
    buckets = assign_strata_from_p_tfidf(pilot["scoring"]["row_level"])
    k, selected = select_blind_sample(buckets)
    assert k == 16
    for key in STRATUM_ORDER:
        assert len(selected[key]) == 8
    all_ids = [rid for v in selected.values() for rid in v]
    assert len(all_ids) == len(set(all_ids)) == 32


def test_real_frozen_inputs_group_ids_all_unique_in_sample():
    pilot = json.loads(PILOT_JSON_PATH.read_text(encoding="utf-8"))
    groups = json.loads(GROUPS_ARTIFACT_PATH.read_text(encoding="utf-8"))
    group_map = groups["record_id_to_group_id"]
    buckets = assign_strata_from_p_tfidf(pilot["scoring"]["row_level"])
    k, selected = select_blind_sample(buckets)
    pairs = assign_review_ids(k, selected)
    group_ids = [group_map[rid] for _, rid in pairs]
    assert len(set(group_ids)) == len(group_ids) == 2 * k
