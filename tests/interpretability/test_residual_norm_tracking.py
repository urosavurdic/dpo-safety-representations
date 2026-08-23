import numpy as np
import pytest
import torch
import torch.nn as nn

from src.interpretability.residual_norm_tracking import (
    ResidualNormTracker,
    compare_to_baseline,
    compute_baseline_range,
    first_step_exceeding_p99,
    make_norm_clipped_steering_hook,
    make_norm_preserving_steering_hook,
    steer_direction,
)


class _FakeDecoderLayer(nn.Module):
    """Mimics a HF decoder layer returning (hidden_states,) as output --
    same pattern as tests/analysis/test_eval_causal_ablation.py."""
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        return (x,)


def test_steer_direction_matches_eval_steering_v2_semantics():
    direction = torch.tensor([1.0, 0.0, 0.0])
    h = torch.tensor([[3.0, 5.0, -2.0]])
    out = steer_direction(h, direction, alpha=2.0)
    assert torch.allclose(out, torch.tensor([[5.0, 5.0, -2.0]]))


def test_tracker_records_one_entry_per_forward_call_per_layer():
    layers = [_FakeDecoderLayer(), _FakeDecoderLayer()]
    tracker = ResidualNormTracker()
    tracker.register(model=None, decoder_layers=layers)

    x1 = torch.tensor([[[3.0, 4.0]]])  # norm 5, batch=1, seq=1
    for layer in layers:
        layer(x1)
    x2 = torch.tensor([[[6.0, 8.0]]])  # norm 10
    for layer in layers:
        layer(x2)

    records = tracker.collect()
    assert records["0"] == [[5.0], [10.0]]
    assert records["1"] == [[5.0], [10.0]]
    tracker.remove()


def test_tracker_uses_last_token_position_only():
    layer = _FakeDecoderLayer()
    tracker = ResidualNormTracker()
    tracker.register(model=None, decoder_layers=[layer])

    # seq_len=3 (a "prefill" call): only the LAST token's norm should be recorded.
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]]])  # last token norm = 5
    layer(x)

    records = tracker.collect()
    assert records["0"] == [[5.0]]
    tracker.remove()


def test_tracker_handles_batch_dimension():
    layer = _FakeDecoderLayer()
    tracker = ResidualNormTracker()
    tracker.register(model=None, decoder_layers=[layer])

    x = torch.tensor([[[3.0, 4.0]], [[6.0, 8.0]]])  # batch=2: norms 5 and 10
    layer(x)

    records = tracker.collect()
    assert records["0"] == [[5.0, 10.0]]
    tracker.remove()


def test_tracker_layer_indices_subset():
    layers = [_FakeDecoderLayer(), _FakeDecoderLayer(), _FakeDecoderLayer()]
    tracker = ResidualNormTracker()
    tracker.register(model=None, decoder_layers=layers, layer_indices=[0, 2])

    x = torch.tensor([[[3.0, 4.0]]])
    for layer in layers:
        layer(x)

    records = tracker.collect()
    assert set(records.keys()) == {"0", "2"}
    tracker.remove()


def test_tracker_reset_clears_records_but_keeps_hooks():
    layer = _FakeDecoderLayer()
    tracker = ResidualNormTracker()
    tracker.register(model=None, decoder_layers=[layer])
    layer(torch.tensor([[[3.0, 4.0]]]))
    tracker.reset()
    layer(torch.tensor([[[6.0, 8.0]]]))
    assert tracker.collect()["0"] == [[10.0]]
    tracker.remove()


def test_tracker_remove_stops_recording():
    layer = _FakeDecoderLayer()
    tracker = ResidualNormTracker()
    tracker.register(model=None, decoder_layers=[layer])
    layer(torch.tensor([[[3.0, 4.0]]]))
    tracker.remove()
    layer(torch.tensor([[[6.0, 8.0]]]))
    assert tracker.collect()["0"] == [[5.0]]  # second call not recorded


def test_norm_preserving_hook_leaves_norm_unchanged_but_direction_injected():
    direction = torch.tensor([1.0, 0.0])
    layer = _FakeDecoderLayer(transform=lambda x: x)
    handle = layer.register_forward_hook(make_norm_preserving_steering_hook(direction, alpha=100.0))

    h = torch.tensor([[[0.0, 5.0]]])  # norm 5, orthogonal to direction
    out = layer(h)[0]

    assert torch.allclose(out.norm(dim=-1), h.norm(dim=-1), atol=1e-5)
    # direction was injected: output should have a nonzero component along [1, 0]
    assert out[0, 0, 0].item() > 0
    handle.remove()


def test_norm_preserving_hook_is_noop_on_zero_vector():
    direction = torch.tensor([1.0, 0.0])
    layer = _FakeDecoderLayer(transform=lambda x: x)
    handle = layer.register_forward_hook(make_norm_preserving_steering_hook(direction, alpha=5.0))

    h = torch.zeros(1, 1, 2)
    out = layer(h)[0]
    # Falls back to the raw (unscaled) steered vector when there's nothing to preserve.
    assert torch.allclose(out, torch.tensor([[[5.0, 0.0]]]))
    handle.remove()


def test_norm_clipped_hook_leaves_small_vectors_untouched():
    direction = torch.tensor([1.0, 0.0])
    layer = _FakeDecoderLayer(transform=lambda x: x)
    handle = layer.register_forward_hook(
        make_norm_clipped_steering_hook(direction, alpha=0.1, max_norm=torch.tensor(1000.0))
    )
    h = torch.tensor([[[0.0, 5.0]]])
    out = layer(h)[0]
    expected = h + 0.1 * direction
    assert torch.allclose(out, expected, atol=1e-5)
    handle.remove()


def test_norm_clipped_hook_clips_vectors_exceeding_max_norm():
    direction = torch.tensor([1.0, 0.0])
    layer = _FakeDecoderLayer(transform=lambda x: x)
    handle = layer.register_forward_hook(
        make_norm_clipped_steering_hook(direction, alpha=100.0, max_norm=torch.tensor(10.0))
    )
    h = torch.tensor([[[0.0, 5.0]]])  # steered = [100, 5], norm ~100.12, should clip to 10
    out = layer(h)[0]
    assert torch.allclose(out.norm(dim=-1), torch.tensor([10.0]), atol=1e-4)
    handle.remove()


def test_compute_baseline_range_pools_across_steps_and_batch():
    records = {"0": [[1.0, 2.0], [3.0, 4.0]]}  # 4 values: 1,2,3,4
    stats = compute_baseline_range(records)
    assert stats["0"]["n"] == 4
    assert stats["0"]["mean"] == 2.5
    assert stats["0"]["p50"] == pytest.approx(2.5)


def test_compare_to_baseline_flags_exceeds_p99():
    baseline_range = {"0": {"mean": 10.0, "std": 1.0, "p50": 10.0, "p95": 11.5, "p99": 12.0, "n": 100}}
    steered_records = {"0": [[10.5], [50.0]]}  # step 0 normal, step 1 way out of range
    comparison = compare_to_baseline(steered_records, baseline_range)
    assert comparison["0"][0]["exceeds_p99"] is False
    assert comparison["0"][1]["exceeds_p99"] is True
    assert comparison["0"][1]["z_score"] == 40.0  # (50-10)/1


def test_compare_to_baseline_skips_layers_absent_from_baseline():
    baseline_range = {"0": {"mean": 10.0, "std": 1.0, "p50": 10.0, "p95": 11.0, "p99": 12.0, "n": 10}}
    steered_records = {"0": [[10.0]], "5": [[999.0]]}  # layer 5 never seen in baseline
    comparison = compare_to_baseline(steered_records, baseline_range)
    assert "5" not in comparison
    assert "0" in comparison


def test_first_step_exceeding_p99_returns_correct_step():
    comparison = [
        {"step": 0, "mean_norm": 10.0, "z_score": 0.0, "exceeds_p99": False},
        {"step": 1, "mean_norm": 11.0, "z_score": 1.0, "exceeds_p99": False},
        {"step": 2, "mean_norm": 50.0, "z_score": 40.0, "exceeds_p99": True},
    ]
    assert first_step_exceeding_p99(comparison) == 2


def test_first_step_exceeding_p99_returns_none_when_never_exceeded():
    comparison = [{"step": 0, "mean_norm": 10.0, "z_score": 0.0, "exceeds_p99": False}]
    assert first_step_exceeding_p99(comparison) is None


def test_realistic_end_to_end_baseline_vs_collapsing_pattern():
    """Simulates the qualitative shape of the real deprecated-run finding:
    a baseline run with roughly-constant per-layer norms across
    generation, vs a 'steered' run whose norm grows step-over-step at a
    deep layer (the compounding-magnitude hypothesis's predicted
    signature) -- checks the pipeline correctly flags growing divergence,
    not just a single unusual spike."""
    rng = np.random.default_rng(0)
    baseline_steps = [[float(v) for v in rng.normal(20.0, 1.0, size=8)] for _ in range(50)]
    baseline_records = {"27": baseline_steps}
    baseline_range = compute_baseline_range(baseline_records)

    # Steered: norm creeps up roughly linearly with generation step.
    steered_steps = [[20.0 + 3.0 * step] for step in range(50)]
    steered_records = {"27": steered_steps}
    comparison = compare_to_baseline(steered_records, baseline_range)["27"]

    first_exceed = first_step_exceeding_p99(comparison)
    assert first_exceed is not None
    assert first_exceed > 0  # shouldn't be flagged at step 0 given this growth pattern
    # Later steps should be MORE anomalous than earlier ones (monotonic z-score growth).
    z_scores = [e["z_score"] for e in comparison]
    assert z_scores == sorted(z_scores)
