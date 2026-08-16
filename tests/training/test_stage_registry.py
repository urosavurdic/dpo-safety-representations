import pytest

from src.training.stage_registry import (
    TRAINING_STAGES,
    resolve_run_order,
    data_prep_commands_for,
)


def test_every_stage_has_a_valid_yaml_and_dryrun_config_path_pattern():
    for stage, spec in TRAINING_STAGES.items():
        assert spec["kind"] in ("sft", "dpo")
        assert spec["config"].startswith("configs/") and spec["config"].endswith(".yaml")
        assert spec["dryrun_config"].startswith("configs/") and spec["dryrun_config"].endswith(".yaml")
        assert spec["data_prep"] in ("alpaca", "dolly", "pku_safe_rlhf")


def test_resolve_run_order_pulls_in_missing_prerequisites():
    # Asking for M3 alone must also train M1 then M2 first, in that order.
    order = resolve_run_order(["M3"])
    assert order == ["M1", "M2", "M3"]


def test_resolve_run_order_m3_direct_only_needs_m1_not_m2():
    order = resolve_run_order(["M3_direct"])
    assert order == ["M1", "M3_direct"]
    assert "M2" not in order


def test_resolve_run_order_alt_branch_is_independent_of_original_branch():
    order = resolve_run_order(["M3_alt"])
    assert order == ["M1_alt", "M2_alt", "M3_alt"]
    assert "M1" not in order and "M2" not in order


def test_resolve_run_order_dedupes_shared_prerequisites_across_multiple_targets():
    # M3 and M3_direct both need M1; must appear once, before both.
    order = resolve_run_order(["M3", "M3_direct"])
    assert order.count("M1") == 1
    assert order.index("M1") < order.index("M2")
    assert order.index("M1") < order.index("M3_direct")
    assert order.index("M2") < order.index("M3")


def test_resolve_run_order_full_eight_stage_run():
    order = resolve_run_order(list(TRAINING_STAGES.keys()))
    assert set(order) == set(TRAINING_STAGES.keys())
    assert order.index("M1") < order.index("M2") < order.index("M3")
    assert order.index("M1") < order.index("M3_direct")
    assert order.index("M1_alt") < order.index("M2_alt") < order.index("M3_alt")
    assert order.index("M1_alt") < order.index("M3_direct_alt")


def test_resolve_run_order_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Unknown stage"):
        resolve_run_order(["M99"])


def test_resolve_run_order_single_prerequisite_free_stage():
    assert resolve_run_order(["M1"]) == ["M1"]
    assert resolve_run_order(["M1_alt"]) == ["M1_alt"]


def test_data_prep_commands_for_m1_only_needs_eval_set_and_alpaca():
    commands = data_prep_commands_for(["M1"])
    tags = [c[0] for c in commands]
    assert tags == ["controlled eval set + reserved prompts", "alpaca"]


def test_data_prep_commands_for_full_alt_chain_needs_dolly_and_pku_but_not_alpaca():
    commands = data_prep_commands_for(["M1_alt", "M2_alt", "M3_alt"])
    tags = {c[0] for c in commands}
    assert tags == {"controlled eval set + reserved prompts", "dolly", "pku_safe_rlhf"}
    assert "alpaca" not in tags


def test_data_prep_commands_for_deduplicates_across_stages_needing_same_prep():
    # M2 and M3 both need pku_safe_rlhf - must appear once, not twice.
    commands = data_prep_commands_for(["M2", "M3"])
    tags = [c[0] for c in commands]
    assert tags.count("pku_safe_rlhf") == 1


def test_data_prep_commands_for_m3_direct_skips_eval_set_and_alpaca_m1_data():
    # M3_direct's OWN data_prep tag is pku_safe_rlhf; it doesn't itself need
    # alpaca/eval-set prep (that's M1's prerequisite, resolved separately by
    # resolve_run_order, not by data_prep_commands_for on M3_direct alone).
    commands = data_prep_commands_for(["M3_direct"])
    tags = {c[0] for c in commands}
    assert tags == {"pku_safe_rlhf"}
