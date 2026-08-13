import yaml

from src.training.train_dpo import build_dpo_config

def _load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_build_dpo_config_from_m3_dpo_yaml(tmp_path):
    cfg = _load_cfg("configs/m3_dpo.yaml")
    dpo_args = build_dpo_config(cfg, tmp_path)

    # Would have caught bug #1 (max_prompt_length removed from DPOConfig):
    # this line raises TypeError immediately if build_dpo_config still
    # passes an unsupported kwarg.
    assert dpo_args.max_length == cfg["training"]["max_seq_length"]

    # Would have caught bug #2 (bf16 defaults True unless fp16 is also
    # explicitly set): asserts the real config's precision flags actually
    # took effect, not silently fell back to a default.
    assert dpo_args.bf16 is False
    assert dpo_args.fp16 is True

    assert dpo_args.beta == cfg["dpo"]["beta"]
    assert dpo_args.loss_type == [cfg["dpo"]["loss_type"]]


def test_build_dpo_config_from_m3_gpu_dryrun_yaml(tmp_path):
    cfg = _load_cfg("configs/m3_gpu_dryrun.yaml")
    dpo_args = build_dpo_config(cfg, tmp_path)

    assert dpo_args.max_length == cfg["training"]["max_seq_length"]
    assert dpo_args.bf16 is False
    assert dpo_args.fp16 is True
    assert dpo_args.max_steps == cfg["training"]["max_steps"]


def test_build_dpo_config_report_to_empty_without_wandb_project(tmp_path):
    cfg = _load_cfg("configs/m3_gpu_dryrun.yaml")
    assert cfg["wandb"]["project"] is None  # confirms this config exercises the "no wandb" branch
    dpo_args = build_dpo_config(cfg, tmp_path)
    assert dpo_args.report_to == []