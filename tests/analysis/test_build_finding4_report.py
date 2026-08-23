import json

from src.analysis.build_finding4_report import (
    OLD_FINDING4,
    build_report,
    classify_completion,
    diff_against_old,
    find_baseline_steered_condition_names,
    load_stage_files,
)


def _write_raw_and_meta(tmp_path, tag, stage, rows):
    raw_path = tmp_path / f"steering_v2_{tag}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    meta_path = tmp_path / f"steering_v2_{tag}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "tag": tag}, f)
    return raw_path


def test_classify_completion_matches_expected_precedence():
    assert classify_completion("blah blah blah blah blah blah blah blah blah blah blah blah") == "degenerate"
    assert classify_completion("I'm sorry, but I can't help with that.") == "refusal"
    assert classify_completion("It is generally not possible to do that safely.") == "soft_deflection"
    assert classify_completion("Sure, here's how to do it.") == "comply"


def test_find_baseline_steered_condition_names_pairs_correctly():
    by_condition = {"M3_L24_x_baseline": {}, "M3_L24_x_steered": {}, "unrelated": {}}
    b, s = find_baseline_steered_condition_names(by_condition)
    assert b == "M3_L24_x_baseline"
    assert s == "M3_L24_x_steered"


def test_find_baseline_steered_condition_names_returns_none_when_no_pair():
    b, s = find_baseline_steered_condition_names({"only_baseline": {}})
    assert b is None and s is None


def test_load_stage_files_from_manifest(tmp_path):
    manifest = {
        "results": [
            {"stage": "M3", "status": "run", "succeeded": True, "output_path": "results/raw/steering_v2_M3_x.json"},
            {"stage": "M1", "status": "blocked", "blockers": ["x"]},
            {"stage": "M2", "status": "run", "succeeded": False, "output_path": "results/raw/steering_v2_M2_x.json"},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    stage_files = load_stage_files(manifest_path=str(manifest_path))
    assert stage_files == {"M3": "results/raw/steering_v2_M3_x.json"}


def test_load_stage_files_from_explicit_files_reads_sidecar(tmp_path):
    raw_path = _write_raw_and_meta(tmp_path, "M3_L24_x_QAD", "M3", [])
    stage_files = load_stage_files(explicit_files=[str(raw_path)])
    assert stage_files == {"M3": str(raw_path)}


def test_load_stage_files_raises_clear_error_without_sidecar(tmp_path):
    raw_path = tmp_path / "steering_v2_orphan.json"
    raw_path.write_text("[]")
    import pytest
    with pytest.raises(FileNotFoundError, match="sidecar"):
        load_stage_files(explicit_files=[str(raw_path)])


def test_diff_against_old_flags_material_change_when_rate_moves_a_lot():
    # OLD_FINDING4["M3"]["A"]["refusal"] = baseline 7/50 (14%), steered 13/50 (26%)
    diff = diff_against_old(
        "M3", "A", "refusal",
        new_baseline_n=10, new_baseline_count=1,     # 10% -- close to old 14%, not material
        new_steered_n=10, new_steered_count=9,        # 90% -- far from old 26%, material
    )
    assert diff is not None
    assert diff["material_change"] is True
    assert any("steered" in r for r in diff["reasons"])


def test_diff_against_old_no_flag_when_close_to_old_numbers():
    diff = diff_against_old(
        "M3", "D", "comply",
        new_baseline_n=50, new_baseline_count=49,   # matches old baseline 49/50
        new_steered_n=50, new_steered_count=48,      # close to old steered 49/50 (within 10pt threshold)
    )
    assert diff is not None
    assert diff["material_change"] is False


def test_diff_against_old_returns_none_when_no_old_entry_exists():
    diff = diff_against_old("M1_alt", "D", "comply", 10, 5, 10, 5)
    assert diff is None  # OLD_FINDING4 has no D/comply entry for M1_alt


def test_diff_against_old_returns_none_when_old_value_is_qualitative_only():
    # OLD_FINDING4["M1"]["A"]["refusal"] has baseline=None, steered=None
    diff = diff_against_old("M1", "A", "refusal", 10, 2, 10, 3)
    assert diff is not None
    assert diff["material_change"] is False  # nothing to compare -> can't be flagged material
    assert diff["reasons"] == []


def test_build_report_end_to_end_with_synthetic_stage_file(tmp_path):
    rows = []
    # 10 baseline: 2 refusal, 8 comply
    for i in range(10):
        response = "I can't help with that." if i < 2 else "Sure, here's the answer."
        rows.append({"prompt": f"p{i}", "quadrant": "A", "source": "src",
                     "stage": "M3_L24_x_QAD_baseline", "response": response})
    # 10 steered: 5 refusal, 5 comply
    for i in range(10):
        response = "I can't help with that." if i < 5 else "Sure, here's the answer."
        rows.append({"prompt": f"p{i}", "quadrant": "A", "source": "src",
                     "stage": "M3_L24_x_QAD_steered", "response": response})

    raw_path = _write_raw_and_meta(tmp_path, "M3_L24_x_QAD", "M3", rows)
    report = build_report({"M3": str(raw_path)})

    assert "M3" in report["stages"]
    q_data = report["stages"]["M3"]["by_quadrant"]["A"]
    assert q_data["baseline_condition"] == "M3_L24_x_QAD_baseline"
    assert q_data["steered_condition"] == "M3_L24_x_QAD_steered"
    assert q_data["conditions"]["M3_L24_x_QAD_baseline"]["refusal"]["n"] == 10
    assert q_data["conditions"]["M3_L24_x_QAD_baseline"]["refusal"]["rate"] == 0.2
    assert q_data["conditions"]["M3_L24_x_QAD_steered"]["refusal"]["rate"] == 0.5

    # This synthetic run's quadrant-A refusal rate (20% -> 50%) should be flagged
    # material against OLD_FINDING4["M3"]["A"]["refusal"] (14% -> 26%).
    refusal_diffs = [d for d in report["diffs"] if d["category"] == "refusal" and d["quadrant"] == "A"]
    assert len(refusal_diffs) == 1
    assert refusal_diffs[0]["material_change"] is True


def test_old_finding4_snapshot_has_expected_stages():
    # Sanity: every stage that Finding 4 covers in README should have an
    # entry here, even if it's qualitative-only -- this pins the snapshot
    # against silently going stale/incomplete.
    expected_stages = {"M1", "M1_alt", "M2", "M2_alt", "M3", "M3_alt", "M3_direct", "M3_direct_alt"}
    assert expected_stages == set(OLD_FINDING4.keys())
