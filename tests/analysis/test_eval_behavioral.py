import json

from src.analysis.eval_behavioral import eval_set_matches_saved_metadata


def _row(prompt, quadrant, source="src", split=None):
    r = {"prompt": prompt, "quadrant": quadrant, "source": source}
    if split is not None:
        r["split"] = split
    return r


def test_matches_when_saved_snapshot_is_identical(tmp_path):
    """Regression test for the real bug: eval_behavioral.py's resumability
    used to only check `stage_name in all_raw` -- no freshness check at
    all -- so a stage's results committed from a much older, smaller eval
    set were treated as "already completed" forever, even after
    controlled_eval.jsonl grew from 370 to 654 prompts. This is the same
    check eval_extract_activations.py already uses, applied here too."""
    eval_rows = [_row("p1", "A", split="direction_estimation"), _row("p2", "B")]
    meta_path = tmp_path / "_eval_set_metadata.json"
    snapshot = [{"prompt": "p1", "quadrant": "A", "source": "src", "split": "direction_estimation"},
                {"prompt": "p2", "quadrant": "B", "source": "src", "split": None}]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"M0": {"eval_rows": snapshot}}, f)

    assert eval_set_matches_saved_metadata(meta_path, "M0", eval_rows) is True


def test_does_not_match_when_saved_snapshot_is_smaller_stale_set(tmp_path):
    """The actual production bug shape: saved metadata reflects an older,
    smaller eval set (e.g. 370 prompts) than the live one (654) -- must be
    detected as stale, not silently accepted."""
    eval_rows = [_row("p1", "A"), _row("p2", "B"), _row("p3", "C")]  # live: 3 rows
    meta_path = tmp_path / "_eval_set_metadata.json"
    old_snapshot = [{"prompt": "p1", "quadrant": "A", "source": "src", "split": None}]  # old: 1 row
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"M0": {"eval_rows": old_snapshot}}, f)

    assert eval_set_matches_saved_metadata(meta_path, "M0", eval_rows) is False


def test_does_not_match_when_metadata_file_missing(tmp_path):
    """A raw.json committed from before this fix existed has no metadata
    sidecar at all -- must be treated as stale (force a re-run), not
    silently trusted just because the metadata file is absent."""
    eval_rows = [_row("p1", "A")]
    meta_path = tmp_path / "_eval_set_metadata.json"  # never created

    assert eval_set_matches_saved_metadata(meta_path, "M0", eval_rows) is False


def test_does_not_match_when_stage_absent_from_saved_metadata(tmp_path):
    """The combined metadata file exists and has OTHER stages recorded, but
    not this one (e.g. M1 was extracted fresh but M0 never got a metadata
    entry, from before this fix existed) -- must be stale for M0, not
    accidentally match on an unrelated stage's snapshot."""
    eval_rows = [_row("p1", "A")]
    meta_path = tmp_path / "_eval_set_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"M1": {"eval_rows": [{"prompt": "p1", "quadrant": "A", "source": "src", "split": None}]}}, f)

    assert eval_set_matches_saved_metadata(meta_path, "M0", eval_rows) is False


def test_split_field_is_part_of_the_comparison(tmp_path):
    """Same prompts/quadrant/source but a different split assignment (e.g.
    assign_direction_split re-run with a different seed) must count as a
    mismatch, not just a prompt-text comparison."""
    eval_rows = [_row("p1", "A", split="direction_estimation")]
    meta_path = tmp_path / "_eval_set_metadata.json"
    snapshot_wrong_split = [{"prompt": "p1", "quadrant": "A", "source": "src", "split": "held_out_behavioral"}]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"M0": {"eval_rows": snapshot_wrong_split}}, f)

    assert eval_set_matches_saved_metadata(meta_path, "M0", eval_rows) is False


def test_multiple_stages_in_one_metadata_file_checked_independently(tmp_path):
    """The real on-disk format is one combined file keyed by stage (unlike
    eval_extract_activations.py's one-file-per-stage) -- confirm a fresh
    stage and a stale stage in the SAME file are each judged correctly."""
    eval_rows = [_row("p1", "A")]
    current = [{"prompt": "p1", "quadrant": "A", "source": "src", "split": None}]
    stale = [{"prompt": "OLD", "quadrant": "A", "source": "src", "split": None}]
    meta_path = tmp_path / "_eval_set_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"M0": {"eval_rows": current}, "M1": {"eval_rows": stale}}, f)

    assert eval_set_matches_saved_metadata(meta_path, "M0", eval_rows) is True
    assert eval_set_matches_saved_metadata(meta_path, "M1", eval_rows) is False
