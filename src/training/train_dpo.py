"""
DPO trainer for M3, initialized from M2. Reuses src/training/{utils,
callbacks,model}.py exactly as train_sft.py does.

Key difference from train_sft.py: model is NOT pre-wrapped with
get_peft_model here. DPOTrainer is given the plain dense model plus
peft_config directly, with ref_model=None - TRL manages the
adapter-disable-for-reference-logprobs trick internally instead of loading
a second full model into VRAM.
"""
import argparse
import os
from pathlib import Path

from trl import DPOConfig, DPOTrainer

from src.training.utils import load_config, ensure_dir, get_git_commit, save_reproducibility_artifacts
from src.training.callbacks import ResumeCallback, GPUMemoryCallback, latest_checkpoint
from src.training.model import load_tokenizer, load_model, create_lora_config
from src.training.dpo_data import load_dpo_dataset, format_dpo_example


def build_dpo_config(cfg: dict, output_dir) -> DPOConfig:
    """
    Build the DPOConfig for a run from the loaded YAML config. Extracted out
    of main() so it can be unit-tested without a network, model, or GPU.

    This exact mapping is what broke twice in a row against real TRL/
    transformers upgrades: `max_prompt_length` was removed from DPOConfig,
    and DPOConfig defaults `bf16=True` whenever `fp16` isn't also passed
    explicitly (see PROJECT_CONTEXT.md experiment log). A test that actually
    constructs this object from configs/m3_dpo.yaml and
    configs/m3_gpu_dryrun.yaml catches that whole class of breakage in under
    a second, before a Colab GPU session is ever spent on it.
    """
    report_to = ["wandb"] if cfg.get("wandb", {}).get("project") else []

    return DPOConfig(
        output_dir=str(output_dir),
        beta=cfg["dpo"]["beta"],
        loss_type=cfg["dpo"]["loss_type"],
        max_length=cfg["training"]["max_seq_length"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        max_steps=cfg["training"].get("max_steps", -1),
        learning_rate=cfg["training"]["learning_rate"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"]["fp16"],
        report_to=report_to,
        run_name=cfg.get("wandb", {}).get("run_name"),
        seed=cfg["seed"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    base_dir = Path(cfg["output"]["base_dir"])
    checkpoints_dir = base_dir / "checkpoints"
    final_dir = base_dir / "final"
    ensure_dir(checkpoints_dir)
    ensure_dir(final_dir)

    print(f"[{cfg['experiment_name']}] loading dataset from {cfg['dataset']['path']}")
    raw_dataset = load_dpo_dataset(cfg["dataset"]["path"])
    if cfg["dataset"].get("max_samples"):
        raw_dataset = raw_dataset.select(range(cfg["dataset"]["max_samples"]))
    print(f"Loaded {len(raw_dataset)} preference pairs.")

    dataset = raw_dataset.map(format_dpo_example, remove_columns=raw_dataset.column_names)

    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg)  # plain dense model - NOT get_peft_model-wrapped
    lora_config = create_lora_config(cfg)

    if cfg.get("wandb", {}).get("project"):
        os.environ["WANDB_PROJECT"] = cfg["wandb"]["project"]

    dpo_args = build_dpo_config(cfg, checkpoints_dir)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[ResumeCallback(), GPUMemoryCallback()],
    )

    resume_path = latest_checkpoint(checkpoints_dir)
    if resume_path:
        print(f"Found existing checkpoint: {resume_path}")
    trainer.train(resume_from_checkpoint=resume_path)

    print(f"Training finished. Saving final adapter to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    save_reproducibility_artifacts(cfg, final_dir)
    print("Saved config_used.yaml, git_commit.txt, requirements.txt alongside the final adapter.")

    if cfg["output"].get("push_to_hub") and cfg["output"].get("hf_repo"):
        print(f"Pushing to HF Hub: {cfg['output']['hf_repo']}")
        trainer.model.push_to_hub(cfg["output"]["hf_repo"])
        tokenizer.push_to_hub(cfg["output"]["hf_repo"])

    print("Done.")


if __name__ == "__main__":
    main()