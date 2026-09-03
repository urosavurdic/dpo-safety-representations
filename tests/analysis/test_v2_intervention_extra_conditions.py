"""The extra causal/steering conditions wired into v2_pipeline.stage_causal /
stage_steering (analysis_plan.md §2 CF2, §6.1/§6.2).

stage_causal / stage_steering themselves are GPU-only (need a resident model),
so - as with the rest of v2_pipeline's generation path - only the load-bearing
pure logic is unit-tested here:

* the ``sqrt(gamma) * r`` array fed to ``register_ablation`` reproduces the
  frozen random-ablation update ``h - gamma (h.r) r`` exactly;
* ``register_steering`` accepts the seeded unit ``r`` (steered_random) the same
  way it accepts the learned direction.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.analysis.v2_pipeline import register_ablation, register_steering
from src.analysis.control_directions import seeded_random_directions


class _Block(nn.Module):
    def forward(self, x):  # identity; the hook rewrites the output
        return x


class _Model(nn.Module):
    def __init__(self, n_blocks):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_Block() for _ in range(n_blocks))


def test_sqrt_gamma_r_array_reproduces_frozen_random_ablation_update():
    rng = np.random.default_rng(0)
    hidden = 8
    r = rng.standard_normal(hidden)
    r /= np.linalg.norm(r)
    gamma = 2.3
    h = rng.standard_normal((4, hidden))

    # frozen definition (analysis_plan.md §6.1): h' = h - gamma (h.r) r
    expected = h - gamma * np.outer(h @ r, r)
    # what register_ablation removes given d = sqrt(gamma) * r  is  (h.d) d
    d = np.sqrt(gamma) * r
    via_hook = h - np.outer(h @ d, d)
    np.testing.assert_allclose(via_hook, expected, atol=1e-12)


def test_register_ablation_hook_applies_scaled_random_direction():
    n_layers = 29
    model = _Model(n_layers)
    direction = np.zeros((n_layers, 4), dtype=np.float32)
    direction[24:29, 0] = np.sqrt(2.0)  # sqrt(gamma)=sqrt(2) along axis 0

    handles = register_ablation(model, direction, layers=range(24, 29))
    try:
        x = torch.ones(1, 3, 4)
        out = model.model.layers[23](x)  # block index 23 == hidden_states layer 24
        # axis-0 component scaled by (1 - gamma) = (1 - 2) = -1
        assert torch.allclose(out[..., 0], torch.full_like(out[..., 0], -1.0), atol=1e-5)
        assert torch.allclose(out[..., 1:], torch.ones_like(out[..., 1:]))
    finally:
        for h in handles:
            h.remove()


def test_register_steering_accepts_the_seeded_unit_r_for_steered_random():
    n_layers = 29
    r = seeded_random_directions(n_layers, 4, seed=20260904)
    assert r.shape == (n_layers, 4)
    np.testing.assert_allclose(np.linalg.norm(r, axis=-1), 1.0, atol=1e-9)

    model = _Model(n_layers)
    handles = register_steering(model, r, alphas={24: 3.0})
    try:
        x = torch.zeros(1, 2, 4)
        out = model.model.layers[23](x)
        # steered output = 0 + 3.0 * r[24]
        np.testing.assert_allclose(
            out[0, 0].detach().numpy(), 3.0 * r[24], atol=1e-5
        )
    finally:
        for h in handles:
            h.remove()
