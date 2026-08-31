"""
Focused tests for src/analysis/c_c_construction_audit.py.

Per the C-C task brief, this runs only focused tests for the new
orchestrator code -- not the broad repository suite. These are
integration tests against the real, tracked repository artifacts (the
same convention already used by
tests/analysis/test_c_b_paired_delta_analysis.py), since C-C's whole
job is to execute the already-implemented C-B contract and compile
already-committed secondary evidence, not to unit-test new statistics.
"""
import csv
import json

import pytest

from src.analysis import c_c_construction_audit as c_c


def test_run_locked_c_b_contract_is_reproducible():
    result = c_c.run_locked_c_b_contract()
    assert result["reproducibility_check"]["byte_identical_excluding_code_version"] is True
    assert result["analysis"]["results"]["population_1_all_valid_accepted_pairs"]["n_valid_pairs"] == 104
    assert result["analysis"]["results"]["population_2_assistance_type_preserved_yes"]["n_valid_pairs"] == 78
    assert result["analysis"]["results"]["population_3_assistance_type_preserved_partial"]["n_valid_pairs"] == 26


def test_load_secondary_abcd_distributions_matches_frozen_benchmark():
    secondary = c_c.load_secondary_abcd_distributions()
    assert secondary["quadrant_c_n"] == 104
    # C-A section 1's recorded category proportions for quadrant C.
    assert secondary["quadrant_c_category_composition"] == {
        "harassment_bullying": 41,
        "illegal": 6,
        "cybercrime_intrusion": 20,
        "misinformation_disinformation": 37,
    }
    assert secondary["quadrant_c_source_composition"] == {"StrongREJECT": 104}
    # No new cross-quadrant test is defined.
    assert "no cross-quadrant" in secondary["not_predeclared"].lower()


def test_load_r_authored_summary_present_and_pending():
    r_authored = c_c.load_r_authored_summary()
    assert r_authored["present"] is True
    assert r_authored["n_queued"] == 52
    assert r_authored["review_status_counts"] == {"pending": 52}
    assert len(r_authored["comparability_caveats"]) >= 1


def test_confounds_reports_template_gap_and_no_new_statistic():
    c_b_result = c_c.run_locked_c_b_contract()
    confounds = c_c.build_confounds(c_b_result["analysis"])
    assert confounds["repeated_template_concentration"]["computed"] is False
    assert confounds["source_association"]["applicable"] is False
    # Length-sensitivity numbers are read from C-B's own already-computed
    # per-feature output, not recomputed with a different method.
    pop1 = c_b_result["analysis"]["results"]["population_1_all_valid_accepted_pairs"]
    expected = pop1["features"]["lexical_diversity"]["length_sensitivity"]["spearman_corr_with_word_count_delta"]
    actual = confounds["length_sensitivity"]["raw_association_by_population"][
        "population_1_all_valid_accepted_pairs"
    ]["lexical_diversity"]["spearman_corr_with_word_count_delta"]
    assert actual == pytest.approx(expected)


def test_decision_status_assigns_no_keep_drop_label():
    c_b_result = c_c.run_locked_c_b_contract()
    confounds = c_c.build_confounds(c_b_result["analysis"])
    decision = c_c.build_decision_status(c_b_result["analysis"], confounds)
    assert decision["keep_drop_label_assigned"] is False
    assert decision["strongest_paired_signal_population_1_all_pairs"]["feature"] is not None
    assert len(decision["contradictory_findings"]) >= 1
    assert len(decision["what_this_audit_cannot_establish"]) >= 1


def test_end_to_end_creates_exactly_three_files(tmp_path):
    out_dir = tmp_path / "c_construction_audit"
    audit = c_c.main(["--out-dir", str(out_dir)])

    produced = sorted(p.name for p in out_dir.iterdir())
    assert produced == ["audit.json", "audit_summary.md", "input_manifest.json"]

    # JSON outputs parse cleanly.
    json.loads((out_dir / "audit.json").read_text(encoding="utf-8"))
    json.loads((out_dir / "input_manifest.json").read_text(encoding="utf-8"))

    assert audit["decision_status"]["keep_drop_label_assigned"] is False
    assert audit["provenance"]["reproducibility_check"]["byte_identical_excluding_code_version"] is True


def test_end_to_end_is_byte_identical_on_rerun(tmp_path):
    out_dir_1 = tmp_path / "run1"
    out_dir_2 = tmp_path / "run2"
    c_c.main(["--out-dir", str(out_dir_1)])
    c_c.main(["--out-dir", str(out_dir_2)])

    for name in ("audit.json", "audit_summary.md", "input_manifest.json"):
        assert (out_dir_1 / name).read_text(encoding="utf-8") == (out_dir_2 / name).read_text(
            encoding="utf-8"
        ), f"{name} differs between identical reruns"


def test_no_raw_prompt_text_in_committed_style_output(tmp_path):
    """Mirrors C-B's own safety property: committed outputs must never
    contain raw R104/R-AUTHORED prompt text, only aggregate statistics
    and opaque identifiers/paths/hashes."""
    out_dir = tmp_path / "c_construction_audit"
    c_c.main(["--out-dir", str(out_dir)])
    audit_text = (out_dir / "audit.json").read_text(encoding="utf-8")
    summary_text = (out_dir / "audit_summary.md").read_text(encoding="utf-8")

    review_rows = list(csv.DictReader(open(c_c.REPO_ROOT / "data/review/c_review_queue.csv")))
    authored_rows = list(
        csv.DictReader(open(c_c.REPO_ROOT / "data/review/c_source_authored_review_queue.csv"))
    )
    for rows, fields in (
        (review_rows, ("source_prompt", "candidate_prompt")),
        (authored_rows, ("source_prompt", "candidate_prompt", "scored_prompt")),
    ):
        for row in rows:
            for field in fields:
                val = row.get(field, "")
                if val and len(val) > 20:
                    assert val not in audit_text
                    assert val not in summary_text
