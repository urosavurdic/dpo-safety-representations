"""Known-answer checks for the contrast controls."""
import numpy as np

from src.stagecmp import contrast_controls as cc


def _acts(rng, n=40, hidden=8, rotate=False):
    """Stored-activation layout: ``(n_rows, len(LAYERS), hidden)`` per stage.
    Rows 0..n/2 are 'harmful', the rest 'benign'. SFT copies the start; DPO
    either copies it too or adds a harmful-only shift orthogonal to the start
    contrast."""
    acts = {}
    for start, sft, dpo in cc.BRANCHES.values():
        x = rng.normal(size=(n, len(cc.LAYERS), hidden))
        x[: n // 2, :, 0] += 3.0  # start contrast along axis 0
        acts[start] = x
        acts[sft] = x.copy()
        y = x.copy()
        if rotate:
            y[: n // 2, :, 1] += 3.0  # new component along axis 1
        acts[dpo] = y
    return acts


def test_identical_checkpoints_have_no_orthogonal_part():
    rng = np.random.default_rng(0)
    acts = _acts(rng)
    out = cc.control_geometry(acts, np.arange(20), np.arange(20, 40), b=200, seed=1)
    for cell in out["layers"].values():
        for arm in cell["arms"].values():
            assert abs(arm["R"] - 1) < 1e-9 and arm["orth"] < 1e-6
        assert abs(cell["recipe_orth"]["point"]) < 1e-6


def test_harmful_only_shift_shows_as_orthogonal_recipe_difference():
    rng = np.random.default_rng(0)
    acts = _acts(rng, rotate=True)
    out = cc.control_geometry(acts, np.arange(20), np.arange(20, 40), b=500, seed=1)
    for cell in out["layers"].values():
        assert cell["recipe_orth"]["point"] > 0.5
        assert cell["recipe_orth"]["adj_low"] > 0
        for branch in ("alpaca", "dolly"):
            assert cell["arms"][f"dpo_{branch}"]["cosine"] < cell["arms"][f"sft_{branch}"]["cosine"]


def test_normalisation_catches_cosmetic_differences():
    assert cc._norm("How do I  kill a Python process?") == cc._norm("how do i kill a python process")
