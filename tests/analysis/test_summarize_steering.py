import json

import pytest

from src.analysis.summarize_steering import find_condition_pairs, main


def _row(stage, quadrant, response):
    return {"prompt": "p", "quadrant": quadrant, "source": "s", "stage": stage, "response": response}


def test_find_condition_pairs_derives_from_actual_stage_names():
    rows = [
        _row("M3_L24_..._baseline", "A", "x"),
        _row("M3_L24_..._steered", "A", "x"),
    ]
    pairs = find_condition_pairs(rows)
    assert pairs == [("M3_L24_..._baseline", "M3_L24_..._steered")]


def test_find_condition_pairs_handles_multiple_pairs_in_one_file():
    rows = [
        _row("M3_baseline", "A", "x"), _row("M3_steered", "A", "x"),
        _row("M3_alt_baseline", "A", "x"), _row("M3_alt_steered", "A", "x"),
    ]
    pairs = find_condition_pairs(rows)
    assert pairs == [("M3_alt_baseline", "M3_alt_steered"), ("M3_baseline", "M3_steered")]


def test_find_condition_pairs_ignores_unmatched_baseline():
    # A "_baseline" stage with no matching "_steered" counterpart shouldn't
    # produce a pair - this is what --skip-baseline runs would look like.
    rows = [_row("M3_baseline", "A", "x")]
    assert find_condition_pairs(rows) == []


def test_find_condition_pairs_empty_for_no_rows():
    assert find_condition_pairs([]) == []


def test_main_requires_file_argument(monkeypatch):
    # The whole point of this fix: no silent default to fall back to.
    monkeypatch.setattr("sys.argv", ["summarize_steering.py"])
    with pytest.raises(SystemExit):
        main()


def test_main_derives_output_filename_from_input(tmp_path, monkeypatch):
    rows = [
        _row("M3_L24_quadrant_a_projection_coef1_QAD_baseline", "D", "I can help with that."),
        _row("M3_L24_quadrant_a_projection_coef1_QAD_steered", "D", "I can help with that."),
    ]
    in_path = tmp_path / "steering_v2_M3_L24_quadrant_a_projection_coef1_QAD.json"
    in_path.write_text(json.dumps(rows))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["summarize_steering.py", "--file", str(in_path)])

    main()

    expected_out = tmp_path / "results" / "summaries" / f"{in_path.stem}_summary.json"
    assert expected_out.exists()
    summary = json.loads(expected_out.read_text())
    assert "M3_L24_quadrant_a_projection_coef1_QAD_baseline_D" in summary
    assert "M3_L24_quadrant_a_projection_coef1_QAD_steered_D" in summary


def test_main_never_mixes_up_old_file_with_new_one(tmp_path, monkeypatch):
    # Regression test for the exact bug that caused the misleading result:
    # two DIFFERENT files, explicitly summarized one at a time, must produce
    # DIFFERENT summaries - there is no shared hardcoded default to blur them.
    old_rows = [
        _row("M3_baseline", "D", "Sure, here you go."),
        _row("M3_steered", "D", "I can't help with that."),
    ]
    new_rows = [
        _row("M3_L24_x_baseline", "D", "Sure, here you go."),
        _row("M3_L24_x_steered", "D", "Sure, here you go."),
    ]
    old_path = tmp_path / "steering_raw_D.json"
    new_path = tmp_path / "steering_v2_M3_L24_x.json"
    old_path.write_text(json.dumps(old_rows))
    new_path.write_text(json.dumps(new_rows))

    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr("sys.argv", ["summarize_steering.py", "--file", str(old_path)])
    main()
    old_summary = json.loads((tmp_path / "results" / "summaries" / f"{old_path.stem}_summary.json").read_text())

    monkeypatch.setattr("sys.argv", ["summarize_steering.py", "--file", str(new_path)])
    main()
    new_summary = json.loads((tmp_path / "results" / "summaries" / f"{new_path.stem}_summary.json").read_text())

    assert old_summary != new_summary
    assert old_summary["M3_steered_D"]["refusal"] == 1
    assert new_summary["M3_L24_x_steered_D"]["refusal"] == 0
