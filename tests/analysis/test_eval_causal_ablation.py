import pytest
import torch
import torch.nn as nn

import src.analysis.eval_causal_ablation as eca
from src.analysis.eval_causal_ablation import (
    ablate_direction,
    filter_to_held_out_behavioral_split,
    get_decoder_layers,
    main,
    register_ablation_hooks,
    run_condition,
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


def test_main_refuses_to_run_without_allow_legacy(monkeypatch, capsys):
    # This deprecated standalone path is not part of the frozen-v2 T4 run - it
    # must fail closed (before any model/tokenizer load) unless explicitly opted
    # into with --allow-legacy.
    monkeypatch.setattr("sys.argv", ["eval_causal_ablation"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "REFUSING TO RUN" in err
    assert "v2_pipeline" in err


def test_main_refuses_even_with_limit_flag(monkeypatch):
    # A dry-run-style invocation is still the deprecated path - still refused.
    monkeypatch.setattr("sys.argv", ["eval_causal_ablation", "--limit", "1"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_run_condition_rows_carry_both_stage_and_model_stage(monkeypatch):
    # Consumers (summarize_/mcnemar_/bootstrap_causal_ablation.py) read row["stage"];
    # v2_pipeline.result_row's convention is stage=condition, model_stage=checkpoint.
    # The legacy generator must emit both so its opt-in output is consumable.
    monkeypatch.setattr(
        eca, "generate_batch",
        lambda model, tokenizer, prompts, device: ["resp" for _ in prompts],
    )
    rows = [
        {"prompt": "p1", "quadrant": "A", "source": "harmbench"},
        {"prompt": "p2", "quadrant": "D", "source": "alpaca"},
    ]
    out = []
    run_condition(None, None, rows, "cpu", "M3_ablated", out)
    assert len(out) == 2
    for row in out:
        assert row["stage"] == "M3_ablated"
        assert row["model_stage"] == "M3_ablated"
        assert set(row) == {"prompt", "quadrant", "source", "stage", "model_stage", "response"}


@pytest.mark.parametrize("stage", ["M3", "M3_direct", "M3_alt", "M3_direct_alt"])
def test_main_parser_accepts_the_four_dpo_endpoint_stages(stage, monkeypatch, capsys):
    # Matches v2_pipeline.INTERVENTION_STAGES. argparse would reject an invalid
    # --stage choice with "invalid choice"; instead we should reach the
    # deprecation guard ("REFUSING TO RUN"), proving the choice was accepted.
    monkeypatch.setattr("sys.argv", ["eval_causal_ablation", "--stage", stage])
    with pytest.raises(SystemExit):
        main()
    err = capsys.readouterr().err
    assert "REFUSING TO RUN" in err
    assert "invalid choice" not in err