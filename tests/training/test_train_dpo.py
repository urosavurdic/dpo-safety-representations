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

# --- version robustness ----------------------------------------------------
# TRL removes DPOConfig fields between releases (max_prompt_length, then
# warmup_ratio). CI installs unpinned ranges, so it hits newer TRL than a
# developer machine and these were a hard TypeError there while passing locally.


class _ConfigMissingWarmupRatio:
    """A TRL release that no longer accepts warmup_ratio."""

    def __init__(self, output_dir=None, beta=None, seed=None):
        self.output_dir = output_dir
        self.beta = beta
        self.seed = seed


def test_unsupported_keys_are_dropped_not_raised():
    from src.training.train_dpo import _build_supported

    cfg = _build_supported(
        _ConfigMissingWarmupRatio,
        dict(output_dir="/tmp/x", beta=0.1, seed=7, warmup_ratio=0.03),
    )
    assert (cfg.output_dir, cfg.beta, cfg.seed) == ("/tmp/x", 0.1, 7)


def test_dropped_keys_are_reported_loudly():
    """Dropping a training hyperparameter silently would change a run without
    saying so, which is worse than crashing."""
    import warnings

    from src.training.train_dpo import _build_supported

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build_supported(
            _ConfigMissingWarmupRatio,
            dict(output_dir="/tmp/x", beta=0.1, seed=7, warmup_ratio=0.03),
        )

    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert "warmup_ratio" in str(caught[0].message)


def test_supported_keys_are_all_passed_through():
    from src.training.train_dpo import _build_supported

    cfg = _build_supported(
        _ConfigMissingWarmupRatio, dict(output_dir="/tmp/y", beta=0.2, seed=1)
    )
    assert (cfg.output_dir, cfg.beta, cfg.seed) == ("/tmp/y", 0.2, 1)
