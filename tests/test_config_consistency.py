"""
Regression test for a real bug: configs/m2_alt_gpu_dryrun.yaml (and the two
other alt-branch dry-run configs) pointed init_from_adapter at the REAL HF
repo (urosavurdic/qwen2.5-1.5b-m1-alt-helpful), which doesn't exist until
M1_alt is actually really trained and pushed (push_to_hub: false in a dry
run, by design). This made a full dry-run chain of a brand-new branch
(M1_alt -> M2_alt -> M3_alt/M3_direct_alt) impossible to smoke-test end to
end - caught only by actually running it in Colab, not by any prior test.

Fixed by pointing those three configs' init_from_adapter at the
PREREQUISITE stage's own LOCAL dry-run output directory instead. This test
checks that invariant holds for every dry-run config that uses a local path
(starts with "/") - NOT that every dry-run config must use one, since
M2/M3/M3_direct's dry-run configs intentionally still point at the real
M1/M2 HF repos (those already exist for real, and switching them to a local
path could break currently-working behavior if this session never happened
to run M1's own dry run first - see PR discussion / commit message).

Uses posixpath, NOT pathlib.Path, to build the expected path string. These
config values are Colab/Linux runtime paths (data - opaque strings sent to
a Colab kernel), not paths on whatever machine runs this test. pathlib.Path
renders with the HOST OS's separator on str() - on Windows that's
backslashes, which would never match the (correctly) forward-slash config
values and made this exact test fail on a real Windows dev machine the
first time it ran.
"""
import posixpath

import yaml

from src.training.stage_registry import TRAINING_STAGES


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_local_path_init_from_adapter_matches_prerequisite_stage_own_dryrun_output():
    checked_any_local_path = False
    for stage, spec in TRAINING_STAGES.items():
        dep = spec["depends_on"]
        if dep is None:
            continue
        dryrun_cfg = _load_yaml(spec["dryrun_config"])
        init_from = dryrun_cfg.get("model", {}).get("init_from_adapter")
        if init_from is None or not init_from.startswith("/"):
            continue  # points at a real HF repo - not this test's concern (see docstring)

        checked_any_local_path = True
        dep_dryrun_cfg = _load_yaml(TRAINING_STAGES[dep]["dryrun_config"])
        expected = posixpath.join(dep_dryrun_cfg["output"]["base_dir"], "final")
        assert init_from == expected, (
            f"{spec['dryrun_config']}'s init_from_adapter ({init_from}) doesn't match "
            f"its prerequisite {dep}'s own dry-run final output ({expected}) - this is "
            f"exactly the bug that broke the M1_alt->M2_alt dry-run chain."
        )

    # Sanity: the fix actually added local-path configs for this test to check -
    # if this were 0, the test above would be vacuously (uselessly) passing.
    assert checked_any_local_path, "expected at least one dry-run config to use a local path by now"


def test_every_dryrun_config_output_base_dir_is_unique():
    """Two stages accidentally sharing an output.base_dir would make one
    silently overwrite the other's checkpoints/final adapter."""
    base_dirs = []
    for spec in TRAINING_STAGES.values():
        cfg = _load_yaml(spec["dryrun_config"])
        base_dirs.append(cfg["output"]["base_dir"])
    assert len(base_dirs) == len(set(base_dirs)), f"duplicate output.base_dir found: {base_dirs}"