"""
Single source of truth for what a training stage IS: which trainer it uses
(SFT vs DPO), which config drives it, what it depends on having already
been pushed to HF first, and which data-prep step it needs.

Both the unified training Colab notebook and the local reproduction script
(src/reproduce.py) import TRAINING_STAGES from here rather than each
hardcoding their own copy of this list - this is the thing that would have
silently drifted out of sync (e.g. a new alt-branch stage added to configs/
but forgotten in one of the two orchestrators) if duplicated.

This does NOT duplicate src.training.model.STAGE_ADAPTER_CHAINS (that's
runtime adapter-merge order for LOADING a trained stage's weights); this is
training-time orchestration metadata (which config to run, in what order,
needing what data first). The two are related but serve different callers.
"""
from pathlib import Path

TRAINING_STAGES = {
    "M1": {
        "kind": "sft",
        "depends_on": None,
        "config": "configs/m1_sft_helpful.yaml",
        "dryrun_config": "configs/m1_gpu_dryrun.yaml",
        "data_prep": "alpaca",
    },
    "M2": {
        "kind": "sft",
        "depends_on": "M1",
        "config": "configs/m2_sft_safety.yaml",
        "dryrun_config": "configs/m2_gpu_dryrun.yaml",
        "data_prep": "pku_safe_rlhf",
    },
    "M3": {
        "kind": "dpo",
        "depends_on": "M2",
        "config": "configs/m3_dpo.yaml",
        "dryrun_config": "configs/m3_gpu_dryrun.yaml",
        "data_prep": "pku_safe_rlhf",
    },
    "M3_direct": {
        "kind": "dpo",
        "depends_on": "M1",
        "config": "configs/m3_direct_dpo.yaml",
        "dryrun_config": "configs/m3_direct_gpu_dryrun.yaml",
        "data_prep": "pku_safe_rlhf",
    },
    "M1_alt": {
        "kind": "sft",
        "depends_on": None,
        "config": "configs/m1_alt_sft_helpful.yaml",
        "dryrun_config": "configs/m1_alt_gpu_dryrun.yaml",
        "data_prep": "dolly",
    },
    "M2_alt": {
        "kind": "sft",
        "depends_on": "M1_alt",
        "config": "configs/m2_alt_sft_safety.yaml",
        "dryrun_config": "configs/m2_alt_gpu_dryrun.yaml",
        "data_prep": "pku_safe_rlhf",
    },
    "M3_alt": {
        "kind": "dpo",
        "depends_on": "M2_alt",
        "config": "configs/m3_alt_dpo.yaml",
        "dryrun_config": "configs/m3_alt_gpu_dryrun.yaml",
        "data_prep": "pku_safe_rlhf",
    },
    "M3_direct_alt": {
        "kind": "dpo",
        "depends_on": "M1_alt",
        "config": "configs/m3_direct_alt_dpo.yaml",
        "dryrun_config": "configs/m3_direct_alt_gpu_dryrun.yaml",
        "data_prep": "pku_safe_rlhf",
    },
}

# Where each data_prep tag's OUTPUT file lands, and how to build it if missing.
# "pku_safe_rlhf" produces two files (dpo_pairs.jsonl consumed by dpo-kind
# stages, sft_safety.jsonl consumed by M2/M2_alt) from ONE command, since
# data_prep.py builds both from a single matched pass - see its docstring.
DATA_PREP = {
    "alpaca": {
        "output": "data/processed/sft_helpful.jsonl",
        "command": "python -m src.data_pipeline.build_m1_data --dataset alpaca",
    },
    "dolly": {
        "output": "data/processed/sft_helpful_alt.jsonl",
        "command": "python -m src.data_pipeline.build_m1_data --dataset dolly",
    },
    "pku_safe_rlhf": {
        "output": "data/processed/dpo_pairs.jsonl",  # sft_safety.jsonl built alongside it
        "command": "python -m src.data_pipeline.data_prep",
    },
}

# build_m1_data.py needs alpaca_reserved_for_eval.json (built by
# build_eval_set.py) to exclude quadrant D's reserved prompts, regardless of
# which --dataset is selected - this is a prerequisite of BOTH alpaca and
# dolly, not something either data_prep entry above should duplicate.
EVAL_SET_PREREQUISITE = {
    "output": "data/processed/controlled_eval.jsonl",
    "command": "python -m src.data_pipeline.build_eval_set",
}


def resolve_run_order(selected_stages):
    """Topologically sorts `selected_stages` by depends_on, and pulls in any
    unselected prerequisite stages automatically (you can't train M3 without
    M2 already having been trained and pushed, whether or not you also asked
    for M2 in this run). Raises on an unknown stage name or a cycle (the
    latter can't actually happen given TRAINING_STAGES' fixed shape, but
    guards against a future bad edit)."""
    unknown = set(selected_stages) - set(TRAINING_STAGES)
    if unknown:
        raise ValueError(f"Unknown stage(s): {sorted(unknown)}. Valid: {sorted(TRAINING_STAGES)}")

    needed = set(selected_stages)
    frontier = list(selected_stages)
    while frontier:
        stage = frontier.pop()
        dep = TRAINING_STAGES[stage]["depends_on"]
        if dep and dep not in needed:
            needed.add(dep)
            frontier.append(dep)

    ordered = []
    visited = set()

    def visit(stage, path):
        if stage in visited:
            return
        if stage in path:
            raise ValueError(f"Dependency cycle detected involving {stage}")
        dep = TRAINING_STAGES[stage]["depends_on"]
        if dep:
            visit(dep, path | {stage})
        visited.add(stage)
        ordered.append(stage)

    for stage in sorted(needed):  # sorted() only for deterministic traversal start
        visit(stage, set())

    return ordered


def data_prep_commands_for(selected_stages):
    """Returns the ordered list of (description, output_path, command) needed
    to prepare data for `selected_stages`, deduplicated, eval-set prerequisite
    first. Caller is responsible for skipping any whose output already
    exists - this just says what WOULD need to run, doesn't run anything."""
    tags = {TRAINING_STAGES[s]["data_prep"] for s in selected_stages}
    needs_m1_data = "alpaca" in tags or "dolly" in tags

    commands = []
    if needs_m1_data:
        commands.append(("controlled eval set + reserved prompts",
                          EVAL_SET_PREREQUISITE["output"], EVAL_SET_PREREQUISITE["command"]))
    for tag in sorted(tags):
        spec = DATA_PREP[tag]
        commands.append((tag, spec["output"], spec["command"]))
    return commands
