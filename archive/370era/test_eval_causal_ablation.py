"""Contract of the deprecated standalone causal-ablation entry point.

The module is superseded by the benchmark-bound pipeline and refuses to run
without --allow-legacy. These tests pin that refusal and the row-key contract.
Its generation helpers now live in src/common/generation.py, tested there.
"""

import pytest

import archive.370era.eval_causal_ablation as eca
from archive.370era.eval_causal_ablation import main, run_condition


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
