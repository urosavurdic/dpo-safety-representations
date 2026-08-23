import json

from src.analysis.eval_extract_activations import compute_pool_window, eval_set_matches_saved_metadata


def test_pool_window_uses_full_length_when_shorter():
    assert compute_pool_window(3) == 3


def test_pool_window_caps_at_default():
    assert compute_pool_window(10) == 5


def test_eval_set_matches_saved_metadata_true_when_identical(tmp_path):
    eval_rows = [{"prompt": "p1", "quadrant": "A", "source": "s1", "split": "direction_estimation", "extra": "ignored"}]
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps([{"prompt": "p1", "quadrant": "A", "source": "s1", "split": "direction_estimation"}]))
    assert eval_set_matches_saved_metadata(meta_path, eval_rows) is True


def test_eval_set_matches_saved_metadata_true_with_null_split(tmp_path):
    # Quadrant B/C rows have split=None - must still compare equal, not
    # spuriously mismatch because .get() vs a literal null differ in type.
    eval_rows = [{"prompt": "p1", "quadrant": "B", "source": "s1", "split": None}]
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps([{"prompt": "p1", "quadrant": "B", "source": "s1", "split": None}]))
    assert eval_set_matches_saved_metadata(meta_path, eval_rows) is True


def test_eval_set_matches_saved_metadata_false_when_eval_set_grew(tmp_path):
    # The core bug this guards against: eval set grew (e.g. quadrant C
    # expanded), old metadata file still has the smaller original set.
    eval_rows = [
        {"prompt": "p1", "quadrant": "A", "source": "s1", "split": "direction_estimation"},
        {"prompt": "p2", "quadrant": "C", "source": "s2", "split": None},  # new row
    ]
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps([{"prompt": "p1", "quadrant": "A", "source": "s1", "split": "direction_estimation"}]))
    assert eval_set_matches_saved_metadata(meta_path, eval_rows) is False


def test_eval_set_matches_saved_metadata_false_when_missing(tmp_path):
    eval_rows = [{"prompt": "p1", "quadrant": "A", "source": "s1", "split": "direction_estimation"}]
    assert eval_set_matches_saved_metadata(tmp_path / "does_not_exist.json", eval_rows) is False


def test_eval_set_matches_saved_metadata_false_when_content_differs(tmp_path):
    eval_rows = [{"prompt": "p1 EDITED", "quadrant": "A", "source": "s1", "split": "direction_estimation"}]
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps([{"prompt": "p1", "quadrant": "A", "source": "s1", "split": "direction_estimation"}]))
    assert eval_set_matches_saved_metadata(meta_path, eval_rows) is False


def test_eval_set_matches_saved_metadata_false_when_split_reassigned(tmp_path):
    # Regression coverage for the held-out-split feature specifically: if
    # the split assignment changes (e.g. re-running build_eval_set.py with
    # a different seed/train_frac) but prompt/quadrant/source are otherwise
    # identical, this must still be treated as changed - direction
    # construction and causal/steering would silently use the wrong split
    # of prompts otherwise.
    eval_rows = [{"prompt": "p1", "quadrant": "A", "source": "s1", "split": "held_out_behavioral"}]
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps([{"prompt": "p1", "quadrant": "A", "source": "s1", "split": "direction_estimation"}]))
    assert eval_set_matches_saved_metadata(meta_path, eval_rows) is False
