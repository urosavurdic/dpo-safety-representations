import torch

from src.analysis.eval_residual_norm_diagnostic import (
    build_norm_summary,
    build_steering_hooks,
    summarize_config,
)
from src.interpretability.residual_norm_tracking import compute_baseline_range


def test_build_steering_hooks_maps_hidden_states_index_to_decoder_index():
    directions = {24: torch.tensor([1.0, 0.0]), 28: torch.tensor([0.0, 1.0])}
    alphas = {24: 2.0, 28: 3.0}
    hooks = build_steering_hooks(directions, alphas, norm_preserving=False)
    # hidden_states[i] = output of decoder_layers[i-1] -- same convention eval_steering_v2 uses
    assert set(hooks.keys()) == {23, 27}


def test_build_steering_hooks_applies_correct_alpha_when_called():
    directions = {24: torch.tensor([1.0, 0.0])}
    alphas = {24: 5.0}
    hooks = build_steering_hooks(directions, alphas, norm_preserving=False)
    hook_fn = hooks[23]

    class _Module:
        pass

    out = hook_fn(_Module(), None, torch.tensor([[[0.0, 3.0]]]))
    h = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(h, torch.tensor([[[5.0, 3.0]]]))


def test_build_steering_hooks_norm_preserving_variant_preserves_norm():
    directions = {24: torch.tensor([1.0, 0.0])}
    alphas = {24: 100.0}
    hooks = build_steering_hooks(directions, alphas, norm_preserving=True)
    hook_fn = hooks[23]

    class _Module:
        pass

    h_in = torch.tensor([[[0.0, 4.0]]])
    out = hook_fn(_Module(), None, h_in)
    h_out = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(h_out.norm(dim=-1), h_in.norm(dim=-1), atol=1e-5)


def test_summarize_config_computes_degenerate_rate():
    records = [
        {"response": "a", "is_degenerate": True},
        {"response": "b", "is_degenerate": False},
        {"response": "c", "is_degenerate": True},
        {"response": "d", "is_degenerate": False},
    ]
    summary = summarize_config(records)
    assert summary == {"n_prompts": 4, "n_degenerate": 2, "degenerate_rate": 0.5}


def test_summarize_config_handles_empty_list():
    summary = summarize_config([])
    assert summary["n_prompts"] == 0
    assert summary["degenerate_rate"] is None


def test_build_norm_summary_reports_first_exceeding_step_and_max_z():
    baseline_range = compute_baseline_range({"27": [[10.0, 15.0, 20.0, 25.0, 30.0]] * 5})
    condition_records = {"27": [[20.0], [22.0], [200.0]]}  # step 2 blows way up, step 1 well within normal range
    summary = build_norm_summary(condition_records, baseline_range)
    assert summary["27"]["first_step_exceeding_p99"] == 2
    assert summary["27"]["max_z_score"] > 0


def test_build_norm_summary_none_when_never_exceeds():
    baseline_range = compute_baseline_range({"27": [[20.0, 21.0, 19.0, 20.5]] * 10})
    condition_records = {"27": [[20.0], [20.1], [19.9]]}
    summary = build_norm_summary(condition_records, baseline_range)
    assert summary["27"]["first_step_exceeding_p99"] is None
