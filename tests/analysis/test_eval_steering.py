import json

import torch
import torch.nn as nn

from src.analysis.eval_steering import steer_direction, register_steering_hooks, load_existing_baseline_D


def test_steer_direction_adds_scaled_component():
    direction = torch.tensor([1.0, 0.0, 0.0])
    h = torch.tensor([[3.0, 5.0, -2.0]])
    out = steer_direction(h, direction, alpha=10.0)
    assert torch.allclose(out, torch.tensor([[13.0, 5.0, -2.0]]), atol=1e-6)


def test_steer_direction_zero_alpha_is_noop():
    direction = torch.tensor([0.0, 1.0, 0.0])
    h = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(steer_direction(h, direction, alpha=0.0), h, atol=1e-6)


class _FakeDecoderLayer(nn.Module):
    def forward(self, x):
        return (x,)


class _FakeInnerModel(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_FakeDecoderLayer() for _ in range(n_layers)])


class _FakeCausalLM(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.model = _FakeInnerModel(n_layers)


def test_steering_hook_adds_then_can_be_removed():
    fake_model = _FakeCausalLM(n_layers=2)
    handles = register_steering_hooks(fake_model, {1: torch.tensor([1.0, 0.0, 0.0])}, {1: 5.0})
    x = torch.tensor([[0.0, 2.0, 2.0]])
    out = fake_model.model.layers[0](x)
    assert torch.allclose(out[0], torch.tensor([[5.0, 2.0, 2.0]]), atol=1e-6)
    for h in handles:
        h.remove()
    assert torch.allclose(fake_model.model.layers[0](x)[0], x, atol=1e-6)


def test_load_existing_baseline_D_filters_correctly(tmp_path):
    data = [
        {"stage": "M3_baseline", "quadrant": "D", "prompt": "p1"},
        {"stage": "M3_baseline", "quadrant": "A", "prompt": "p2"},
        {"stage": "M3_ablated", "quadrant": "D", "prompt": "p3"},
    ]
    path = tmp_path / "causal_ablation_raw_wide.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    rows = load_existing_baseline_D(str(path))
    assert len(rows) == 1 and rows[0]["prompt"] == "p1"