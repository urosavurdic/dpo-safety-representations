import json

import numpy as np

from src.analysis.eval_probes import probe_metadata_is_fresh, probe_layer, split_b_train_holdout, split_by_quadrant


def test_split_by_quadrant():
    arr = np.arange(8).reshape(4, 1, 2)
    meta = [{"quadrant": "A"}, {"quadrant": "B"}, {"quadrant": "C"}, {"quadrant": "D"}]
    result = split_by_quadrant(arr, meta)
    assert result["A"].shape == (1, 1, 2)
    assert result["D"].shape == (1, 1, 2)


def test_split_b_train_holdout_sizes():
    b = np.arange(100).reshape(50, 1, 2)
    train, holdout = split_b_train_holdout(b, train_size=30)
    assert len(train) == 30
    assert len(holdout) == 20


def test_probe_layer_runs_on_synthetic_separable_data():
    rng = np.random.RandomState(0)
    unsafe = rng.normal(loc=5, scale=0.1, size=(20, 1, 3))
    safe_train = rng.normal(loc=-5, scale=0.1, size=(20, 1, 3))
    safe_holdout = rng.normal(loc=-5, scale=0.1, size=(5, 1, 3))
    test_c = rng.normal(loc=5, scale=0.1, size=(5, 1, 3))
    test_d = rng.normal(loc=-5, scale=0.1, size=(5, 1, 3))

    result = probe_layer(unsafe, safe_train, safe_holdout, test_c, test_d, layer_idx=0, cv_folds=3)
    assert result["cv_accuracy_mean"] > 0.9
    assert result["holdout_b_flagged_unsafe_frac"] < 0.5
    assert result["quadrant_c_flagged_unsafe_frac"] > 0.5
    assert result["quadrant_d_flagged_unsafe_frac"] < 0.5

def test_probe_metadata_is_fresh_true_when_snapshot_matches_live_activation_metadata(tmp_path):
    """Regression test for the real bug: eval_probes.py's resumability used
    to only check `result_path.exists()` -- no freshness check against the
    live eval set at all -- so a stale results/probes/{stage}_probe_results.json
    (e.g. committed from a much smaller/older eval set) was treated as
    "already computed" forever. Same bug class as eval_behavioral.py and
    eval_extract_activations.py, found the same way: reading the actual
    resumability logic instead of assuming it was already handled."""
    act_dir = tmp_path / "results" / "activations"
    probes_dir = tmp_path / "results" / "probes"
    act_dir.mkdir(parents=True)
    probes_dir.mkdir(parents=True)

    live_metadata = [{"prompt": "p1", "quadrant": "A", "source": "src", "split": None}]
    with open(act_dir / "M0_metadata.json", "w", encoding="utf-8") as f:
        json.dump(live_metadata, f)
    with open(probes_dir / "M0_metadata.json", "w", encoding="utf-8") as f:
        json.dump(live_metadata, f)  # identical snapshot

    import src.analysis.eval_probes as eval_probes_module
    orig_act_dir = eval_probes_module.ACT_DIR
    eval_probes_module.ACT_DIR = act_dir
    try:
        assert probe_metadata_is_fresh("M0", probes_dir) is True
    finally:
        eval_probes_module.ACT_DIR = orig_act_dir


def test_probe_metadata_is_fresh_false_when_snapshot_is_stale(tmp_path):
    """The actual production bug shape: the saved snapshot reflects an
    older/smaller eval set than the live activation metadata."""
    act_dir = tmp_path / "results" / "activations"
    probes_dir = tmp_path / "results" / "probes"
    act_dir.mkdir(parents=True)
    probes_dir.mkdir(parents=True)

    live_metadata = [
        {"prompt": "p1", "quadrant": "A", "source": "src", "split": None},
        {"prompt": "p2", "quadrant": "B", "source": "src", "split": None},
    ]
    stale_snapshot = [{"prompt": "p1", "quadrant": "A", "source": "src", "split": None}]  # missing p2
    with open(act_dir / "M0_metadata.json", "w", encoding="utf-8") as f:
        json.dump(live_metadata, f)
    with open(probes_dir / "M0_metadata.json", "w", encoding="utf-8") as f:
        json.dump(stale_snapshot, f)

    import src.analysis.eval_probes as eval_probes_module
    orig_act_dir = eval_probes_module.ACT_DIR
    eval_probes_module.ACT_DIR = act_dir
    try:
        assert probe_metadata_is_fresh("M0", probes_dir) is False
    finally:
        eval_probes_module.ACT_DIR = orig_act_dir


def test_probe_metadata_is_fresh_false_when_snapshot_missing(tmp_path):
    """A probe_results.json committed from before this fix existed has no
    metadata snapshot alongside it at all -- must be treated as stale (force
    a re-run), not silently trusted just because the snapshot is absent."""
    act_dir = tmp_path / "results" / "activations"
    probes_dir = tmp_path / "results" / "probes"
    act_dir.mkdir(parents=True)
    probes_dir.mkdir(parents=True)
    with open(act_dir / "M0_metadata.json", "w", encoding="utf-8") as f:
        json.dump([{"prompt": "p1", "quadrant": "A", "source": "src", "split": None}], f)
    # no probes_dir / "M0_metadata.json" written

    import src.analysis.eval_probes as eval_probes_module
    orig_act_dir = eval_probes_module.ACT_DIR
    eval_probes_module.ACT_DIR = act_dir
    try:
        assert probe_metadata_is_fresh("M0", probes_dir) is False
    finally:
        eval_probes_module.ACT_DIR = orig_act_dir


def test_probe_metadata_is_fresh_false_when_live_activation_metadata_missing(tmp_path):
    """Live activations were deleted/never extracted for this stage -- can't
    confirm freshness against nothing, so must not claim "fresh"."""
    act_dir = tmp_path / "results" / "activations"
    probes_dir = tmp_path / "results" / "probes"
    act_dir.mkdir(parents=True)
    probes_dir.mkdir(parents=True)
    with open(probes_dir / "M0_metadata.json", "w", encoding="utf-8") as f:
        json.dump([{"prompt": "p1", "quadrant": "A", "source": "src", "split": None}], f)
    # no act_dir / "M0_metadata.json"

    import src.analysis.eval_probes as eval_probes_module
    orig_act_dir = eval_probes_module.ACT_DIR
    eval_probes_module.ACT_DIR = act_dir
    try:
        assert probe_metadata_is_fresh("M0", probes_dir) is False
    finally:
        eval_probes_module.ACT_DIR = orig_act_dir
