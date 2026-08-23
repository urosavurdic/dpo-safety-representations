from src.analysis.mcnemar_steering import build_contingency


def test_build_contingency_counts_matched_prompts_only():
    rows = [
        {"prompt": "p1", "stage": "M3_baseline", "response": "Sure, here's how."},
        {"prompt": "p1", "stage": "M3_steered", "response": "I can't help with that."},
        {"prompt": "p2", "stage": "M3_baseline", "response": "Sure, here's how."},
        {"prompt": "p2", "stage": "M3_steered", "response": "Sure, here's how."},
        {"prompt": "p3", "stage": "M3_baseline", "response": "Sure, here's how."},  # unmatched, no M3_steered row
    ]
    table, n_paired = build_contingency(rows, "M3_baseline", "M3_steered", category="refusal")
    assert n_paired == 2  # p3 excluded, unmatched
    assert table == [[0, 0], [1, 1]]  # p1: no->yes(steered), p2: no->no


def test_build_contingency_works_with_real_eval_steering_v2_tag_names():
    """Regression test for the actual bug found: real eval_steering_v2.py
    output uses "{tag}_baseline"/"{tag}_steered" condition names, never the
    literal "M3_baseline"/"M3_steered" the old hardcoded version required --
    against a real file it silently matched 0 rows. Pin that a realistic
    tag-based pair is handled correctly now that the stage names are
    parameters, not a hardcoded literal."""
    baseline = "M3_L24_quadrant_a_projection_coef1_QAD_baseline"
    steered = "M3_L24_quadrant_a_projection_coef1_QAD_steered"
    rows = [
        {"prompt": "p1", "stage": baseline, "response": "Sure, here's how."},
        {"prompt": "p1", "stage": steered, "response": "I can't help with that."},
        {"prompt": "p2", "stage": baseline, "response": "Sure, here's how."},
        {"prompt": "p2", "stage": steered, "response": "Sure, here's how."},
    ]
    table, n_paired = build_contingency(rows, baseline, steered, category="refusal")
    assert n_paired == 2
    assert table == [[0, 0], [1, 1]]

    # And confirm the OLD hardcoded literal really would have found nothing --
    # i.e. this isn't a change in behavior for old-style files, only a fix
    # for files that were previously silently mishandled.
    _, old_style_n = build_contingency(rows, "M3_baseline", "M3_steered", category="refusal")
    assert old_style_n == 0


def test_build_contingency_filters_by_stage_pair_when_file_has_multiple_conditions():
    """A file can contain more than one baseline/steered pair (e.g. two
    different layer configs saved under the same output path via --tag) --
    build_contingency must only pair prompts within the given stage names,
    not across unrelated conditions that happen to share a prompt."""
    rows = [
        {"prompt": "p1", "stage": "M3_L24_a", "response": "Sure."},
        {"prompt": "p1", "stage": "M3_L24_b", "response": "I can't help with that."},
        {"prompt": "p1", "stage": "M3_L28_a", "response": "Sure."},
        {"prompt": "p1", "stage": "M3_L28_b", "response": "Sure."},
    ]
    table, n_paired = build_contingency(rows, "M3_L24_a", "M3_L24_b", category="refusal")
    assert n_paired == 1
    assert table == [[0, 0], [1, 0]]
