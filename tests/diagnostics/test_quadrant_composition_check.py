from src.diagnostics.quadrant_composition_check import compute_quadrant_stats, check_design_predictions


def make_rows(quadrant, prompts):
    return [{"prompt": p, "quadrant": quadrant} for p in prompts]


def test_compute_quadrant_stats_basic():
    rows = make_rows("A", ["one two three", "four five"])
    stats = compute_quadrant_stats(rows)
    assert stats["A"]["n"] == 2
    assert stats["A"]["mean_words"] == 2.5
    assert stats["A"]["median_words"] == 2.5


def test_compute_quadrant_stats_cue_hits():
    rows = make_rows("C", ["unauthorized access here", "a perfectly clean sentence"])
    stats = compute_quadrant_stats(rows)
    assert stats["C"]["mean_cue_hits"] == 0.5  # one prompt has 1 hit, other has 0
    assert stats["C"]["pct_with_cue_hit"] == 50.0


def test_compute_quadrant_stats_separates_quadrants():
    rows = make_rows("A", ["a b"]) + make_rows("D", ["c d e"])
    stats = compute_quadrant_stats(rows)
    assert set(stats.keys()) == {"A", "D"}
    assert stats["A"]["n"] == 1
    assert stats["D"]["n"] == 1


def test_prediction_b_holds_when_b_scores_comparably_to_a():
    stats = {"A": {"mean_cue_hits": 1.0}, "B": {"mean_cue_hits": 0.8}}
    predictions = check_design_predictions(stats)
    b_pred = [p for p in predictions if p[0].startswith("B (")][0]
    assert b_pred[1] is True


def test_prediction_b_fails_when_b_scores_near_zero():
    stats = {"A": {"mean_cue_hits": 1.0}, "B": {"mean_cue_hits": 0.05}}
    predictions = check_design_predictions(stats)
    b_pred = [p for p in predictions if p[0].startswith("B (")][0]
    assert b_pred[1] is False


def test_prediction_c_lower_than_a_holds():
    stats = {"A": {"mean_cue_hits": 1.0}, "C": {"mean_cue_hits": 0.2}}
    predictions = check_design_predictions(stats)
    c_pred = [p for p in predictions if "lower than A" in p[0]][0]
    assert c_pred[1] is True


def test_prediction_c_lower_than_a_fails_when_equal():
    stats = {"A": {"mean_cue_hits": 0.5}, "C": {"mean_cue_hits": 0.5}}
    predictions = check_design_predictions(stats)
    c_pred = [p for p in predictions if "lower than A" in p[0]][0]
    assert c_pred[1] is False


def test_prediction_c_approaches_d_holds_when_close():
    stats = {"C": {"mean_cue_hits": 0.15}, "D": {"mean_cue_hits": 0.05}}
    predictions = check_design_predictions(stats)
    cd_pred = [p for p in predictions if "approaches D" in p[0]][0]
    assert cd_pred[1] is True


def test_prediction_c_approaches_d_fails_when_far_apart():
    stats = {"C": {"mean_cue_hits": 0.8}, "D": {"mean_cue_hits": 0.05}}
    predictions = check_design_predictions(stats)
    cd_pred = [p for p in predictions if "approaches D" in p[0]][0]
    assert cd_pred[1] is False


def test_check_design_predictions_skips_missing_quadrants():
    # Only A present - shouldn't crash, shouldn't fabricate predictions
    # about quadrants that aren't there.
    stats = {"A": {"mean_cue_hits": 0.5}}
    predictions = check_design_predictions(stats)
    assert predictions == []
