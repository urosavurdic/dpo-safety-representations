import numpy as np

from src.eval_probes import split_by_quadrant, split_b_train_holdout, probe_layer


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