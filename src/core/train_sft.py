"""
SFT trainer for M1 (SFT-Helpful) and M2 (SFT-Safety) - only the config
differs between the two runs. Built on src/training/{utils,callbacks,model,
data,formatting}.py so train_dpo.py (M3) can reuse the same tokenizer/model/
LoRA construction and the same chat-formatting function.

Formatting: uses tokenizer.apply_chat_template (ChatML), not a custom plain
template - Qwen2.5's tokenizer ships real <|im_start|>/<|im_end|> special
tokens and a working chat_template even on the base checkpoint, so the SAME
call can be used on M0 later during activation extraction. See
PROJECT_CONTEXT.md, design decision #4.

Checkpoint layout:
  {output.base_dir}/checkpoints/   <- resumable training state, auto-pruned
  {output.base_dir}/final/         <- completed adapter, written once at the
                                       end; never resumed from, never overwritten.
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import yaml
from peft import get_peft_model
from trl import SFTConfig, SFTTrainer

from src.training.callbacks import ResumeCallback, GPUMemoryCallback, latest_checkpoint
from src.training.model import load_tokenizer, load_model, create_lora_config
from src.training.data import load_sft_dataset
from src.training.formatting import format_chat_example
from src.training.utils import load_config, ensure_dir, get_git_commit, save_reproducibility_artifacts


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown (not a git repo, or git unavailable)"


def save_reproducibility_artifacts(cfg, output_dir):
    ensure_dir(output_dir)
    with open(Path(output_dir) / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    with open(Path(output_dir) / "git_commit.txt", "w", encoding="utf-8") as f:
        f.write(get_git_commit() + "\n")
    if Path("requirements.txt").exists():
        shutil.copy("requirements.txt", Path(output_dir) / "requirements.txt")


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
    raw_dataset = load_sft_dataset(cfg["dataset"]["path"])
    if cfg["dataset"].get("max_samples"):
        raw_dataset = raw_dataset.select(range(cfg["dataset"]["max_samples"]))
    print(f"Loaded {len(raw_dataset)} training examples.")

    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg)
    lora_config = create_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = raw_dataset.map(
        lambda ex: format_chat_example(ex, tokenizer),
        remove_columns=raw_dataset.column_names,
    )
    report_to = []
    if cfg.get("wandb", {}).get("project"):
        report_to = ["wandb"]
        os.environ["WANDB_PROJECT"] = cfg["wandb"]["project"]

    training_args = SFTConfig(
        output_dir=str(checkpoints_dir),
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
        max_length=cfg["training"]["max_seq_length"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"]["fp16"],
        dataset_text_field="text",
        report_to=report_to,
        run_name=cfg.get("wandb", {}).get("run_name"),
        seed=cfg["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
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
        model.push_to_hub(cfg["output"]["hf_repo"])
        tokenizer.push_to_hub(cfg["output"]["hf_repo"])

    print("Done.")


if __name__ == "__main__":
    main()