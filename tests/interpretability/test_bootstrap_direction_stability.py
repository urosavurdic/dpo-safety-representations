import json

import numpy as np

from src.interpretability.bootstrap_direction_stability import (
    DEEP_LAYERS,
    bootstrap_directions,
    stability_sims,
    summarize_stability,
    summarize_stability_full,
)
from src.analysis.eval_refusal_direction import diff_in_means_direction


def _toy_data():
    pooled = np.array([
        [[5.0, 0.0]], [[5.0, 0.0]],   # A x2, identical
        [[-5.0, 0.0]], [[-5.0, 0.0]], # D x2, identical
    ])
    quadrants = np.array(["A", "A", "D", "D"])
    return pooled, quadrants


def test_bootstrap_directions_shape():
    pooled, quadrants = _toy_data()
    boots = bootstrap_directions(pooled, quadrants, n_bootstrap=10, seed=1)
    assert boots.shape == (10, 1, 2)


def test_stable_direction_gives_near_perfect_bootstrap_agreement():
    # All A rows identical, all D rows identical -> every resample gives
    # EXACTLY the same direction; cosine sim should be ~1.0 always.
    pooled, quadrants = _toy_data()
    original = diff_in_means_direction(pooled, quadrants)
    boots = bootstrap_directions(pooled, quadrants, n_bootstrap=20, seed=2)
    mean_sim, std_sim = summarize_stability(boots, original, layer=0)
    assert abs(mean_sim - 1.0) < 1e-6
    assert std_sim < 1e-6


def test_summarize_stability_full_reports_median_and_percentiles():
    """PROJECT_CONTEXT.md's bootstrap spec asks for mean/median/std/2.5%/97.5%,
    not just mean+std."""
    pooled, quadrants = _toy_data()
    original = diff_in_means_direction(pooled, quadrants)
    boots = bootstrap_directions(pooled, quadrants, n_bootstrap=20, seed=2)
    stats = summarize_stability_full(boots, original, layer=0)
    assert set(stats.keys()) == {"mean", "median", "std", "ci_low_2.5pct", "ci_high_97.5pct"}
    assert abs(stats["mean"] - 1.0) < 1e-6
    assert abs(stats["median"] - 1.0) < 1e-6
    assert stats["ci_low_2.5pct"] <= stats["mean"] <= stats["ci_high_97.5pct"]


def test_bootstrap_deterministic_given_fixed_seed():
    pooled, quadrants = _toy_data()
    boots_a = bootstrap_directions(pooled, quadrants, n_bootstrap=15, seed=7)
    boots_b = bootstrap_directions(pooled, quadrants, n_bootstrap=15, seed=7)
    assert np.allclose(boots_a, boots_b)


def test_stability_sims_matches_summarize_stability_mean_and_std():
    # stability_sims is the shared core both summarize_stability and
    # summarize_stability_full were refactored to use - their outputs must
    # still agree with directly computing mean/std on its raw array.
    pooled, quadrants = _toy_data()
    original = diff_in_means_direction(pooled, quadrants)
    boots = bootstrap_directions(pooled, quadrants, n_bootstrap=20, seed=2)
    sims = stability_sims(boots, original, layer=0)
    mean_sim, std_sim = summarize_stability(boots, original, layer=0)
    assert sims.shape == (20,)
    assert abs(float(sims.mean()) - mean_sim) < 1e-9
    assert abs(float(sims.std()) - std_sim) < 1e-9


def test_deep_layers_constant_matches_readme_finding_3():
    # README Finding 3: "layers 16-28" is the tight direct-DPO band.
    assert DEEP_LAYERS == list(range(16, 29))


def test_main_persists_raw_sims_only_for_deep_layers(tmp_path, monkeypatch):
    import src.interpretability.bootstrap_direction_stability as bds

    act_dir = tmp_path / "results" / "activations"
    act_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    n_layers, n_prompts = 29, 8  # matches the real pipeline's 0-28 layer range
    pooled = rng.normal(size=(n_prompts, n_layers, 4))
    quadrants = np.array(["A"] * 4 + ["D"] * 4)
    np.save(act_dir / "M3_pooled.npy", pooled)
    meta = [{"prompt": f"p{i}", "quadrant": q, "source": "toy", "split": "direction_estimation"}
            for i, q in enumerate(quadrants)]
    with open(act_dir / "M3_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    monkeypatch.chdir(tmp_path)  # both erd.ACT_DIR and load_stage's default act_dir are relative
    monkeypatch.setattr(bds, "STAGES", ["M3"])
    # NOTE: bootstrap_directions' n_bootstrap default is bound to N_BOOTSTRAP
    # at def-time, so monkeypatching bds.N_BOOTSTRAP here would NOT change
    # what main() actually uses - leave it at the real 1000 (toy data is
    # tiny, still runs in well under a second).

    bds.main()

    out_path = tmp_path / "results" / "interpretability" / "bootstrap_direction_stability.json"
    with open(out_path, encoding="utf-8") as f:
        result = json.load(f)

    assert "raw_sims" not in result["M3"]["0"]  # shallow layer - stats only, no raw array
    assert "raw_sims" in result["M3"]["16"]  # first deep layer
    assert len(result["M3"]["16"]["raw_sims"]) == 1000
    assert "raw_sims" in result["M3"]["28"]  # last layer