import torch
import torch.nn as nn

from src.analysis.eval_causal_ablation import (
    ablate_direction,
    filter_to_held_out_behavioral_split,
    get_decoder_layers,
    register_ablation_hooks,
)


def test_ablate_direction_removes_only_the_parallel_component():
    direction = torch.tensor([1.0, 0.0, 0.0])
    h = torch.tensor([[3.0, 5.0, -2.0]])
    out = ablate_direction(h, direction)
    assert torch.allclose(out, torch.tensor([[0.0, 5.0, -2.0]]), atol=1e-6)


def test_ablate_direction_leaves_orthogonal_vectors_unchanged():
    direction = torch.tensor([1.0, 0.0, 0.0])
    h = torch.tensor([[0.0, 4.0, -1.0]])
    out = ablate_direction(h, direction)
    assert torch.allclose(out, h, atol=1e-6)


class _FakeDecoderLayer(nn.Module):
    """Mimics a HF decoder layer returning (hidden_states,) as output."""
    def forward(self, x):
        return (x,)


class _FakeInnerModel(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([_FakeDecoderLayer() for _ in range(n_layers)])


class _FakeCausalLM(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.model = _FakeInnerModel(n_layers)


def test_get_decoder_layers_finds_standard_qwen_llama_path():
    layers = get_decoder_layers(_FakeCausalLM())
    assert len(layers) == 3


def test_get_decoder_layers_raises_clear_error_when_path_missing():
    class _WeirdModel(nn.Module):
        pass
    try:
        get_decoder_layers(_WeirdModel())
        assert False, "expected AttributeError"
    except AttributeError as e:
        assert "model.model.layers" in str(e)


def test_hook_modifies_output_and_can_be_removed():
    fake_model = _FakeCausalLM(n_layers=2)
    direction = torch.tensor([1.0, 0.0, 0.0])
    directions_by_layer = {1: direction}  # hidden_states index 1 -> decoder_layers[0]

    handles = register_ablation_hooks(fake_model, directions_by_layer)
    x = torch.tensor([[5.0, 2.0, 2.0]])
    out = fake_model.model.layers[0](x)
    assert torch.allclose(out[0], torch.tensor([[0.0, 2.0, 2.0]]), atol=1e-6)

    for h in handles:
        h.remove()
    out_after_removal = fake_model.model.layers[0](x)
    assert torch.allclose(out_after_removal[0], x, atol=1e-6)


def test_filter_to_held_out_behavioral_split_keeps_only_held_out_a_and_d():
    rows = [
        {"prompt": "a1", "quadrant": "A", "split": "direction_estimation"},
        {"prompt": "a2", "quadrant": "A", "split": "held_out_behavioral"},
        {"prompt": "d1", "quadrant": "D", "split": "direction_estimation"},
        {"prompt": "d2", "quadrant": "D", "split": "held_out_behavioral"},
        {"prompt": "b1", "quadrant": "B", "split": None},
        {"prompt": "c1", "quadrant": "C", "split": None},
    ]
    kept = filter_to_held_out_behavioral_split(rows)
    assert {r["prompt"] for r in kept} == {"a2", "d2", "b1", "c1"}


def test_filter_to_held_out_behavioral_split_drops_a_d_rows_missing_split():
    # Activations/eval rows from before the split existed - drop rather than
    # silently include, since we can't tell if they'd have been held-out.
    rows = [
        {"prompt": "a1", "quadrant": "A"},  # no "split" key at all
        {"prompt": "b1", "quadrant": "B"},
    ]
    kept = filter_to_held_out_behavioral_split(rows)
    assert {r["prompt"] for r in kept} == {"b1"}