"""Re-run one arm under a new seed, checkpointed so any interruption resumes.

The recipe is read from the **original** YAML config of that arm (so the
hyperparameters have one source of truth) and changed in only these ways:

* ``seed`` -- and ``transformers.set_seed`` is called *before* the model is
  built, so the LoRA initialisation is seeded too (``train_sft.py`` built the
  adapter before the trainer seeded anything);
* no Hub push, no W&B, local output paths;
* the trainer's own checkpoints become *resume* checkpoints: every
  ``resume_every`` optimizer steps, latest one only, banked as soon as written;
* at each step in ``dose_steps`` the adapter alone is saved (fp16) and banked,
  for the dose curves;
* ``sft_aol`` only: prompt/completion format, so the loss covers the response
  tokens alone (the original SFT scored the whole rendered conversation).

Training itself is the original code path: the same tokenizer, chat
rendering, LoRA config, ``SFTTrainer`` / ``DPOTrainer`` (reference = the
start, adapter disabled), batch 32 on one device.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

from src.seedrep.rows import DPO_DATA, ROOT, SFT_DATA

RECIPE_CONFIG = {
    ("sft", "alpaca"): "configs/m2_sft_safety.yaml",
    ("dpo", "alpaca"): "configs/m3_direct_dpo.yaml",
    ("sft", "dolly"): "configs/m2_alt_sft_safety.yaml",
    ("dpo", "dolly"): "configs/m3_direct_alt_dpo.yaml",
}
FAULT_ENV = "SEEDREP_FAULT_AT"


class InjectedFault(RuntimeError):
    """Raised by the fault-injection hook; tests and dry runs only."""


def fault_point(kind: str, index: int, scratch: Path) -> None:
    """Raise once at ``SEEDREP_FAULT_AT=kind:index``; a marker file makes the
    rerun pass, which is what the interruption tests rely on."""
    spec = os.environ.get(FAULT_ENV, "")
    if spec != f"{kind}:{index}":
        return
    marker = Path(scratch) / f".fault_{kind}_{index}"
    if marker.exists():
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("fired", encoding="utf-8")
    raise InjectedFault(f"injected fault at {kind}:{index}")


def recipe_config(recipe: str, branch: str, seed: int, overrides: dict | None = None) -> dict:
    from src.training.utils import load_config

    base = "dpo" if recipe == "dpo" else "sft"
    cfg = copy.deepcopy(load_config(str(ROOT / RECIPE_CONFIG[(base, branch)])))
    cfg["seed"] = int(seed)
    cfg["experiment_name"] = f"seedrep_{recipe}_{branch}_s{seed}"
    cfg["output"] = {"push_to_hub": False}
    cfg.pop("wandb", None)
    cfg["dataset"]["path"] = str(DPO_DATA if base == "dpo" else SFT_DATA)
    cfg["recipe"] = recipe
    for section, values in (overrides or {}).items():
        if isinstance(values, dict):
            cfg.setdefault(section, {}).update(values)
        else:
            cfg[section] = values
    return cfg


def _load_start_model(cfg: dict):
    """``src.training.model.load_model`` with the dtype chosen by device: fp16
    on a GPU exactly as the originals, fp32 on CPU (smoke tests only)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], dtype=dtype,
        trust_remote_code=cfg["model"].get("trust_remote_code", True))
    adapter = cfg["model"].get("init_from_adapter")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    return model


def _to_fp16_adapter(adapter_dir: Path) -> None:
    from safetensors.numpy import load_file, save_file

    path = adapter_dir / "adapter_model.safetensors"
    tensors = {k: v.astype("float16") for k, v in load_file(str(path)).items()}
    save_file(tensors, str(path))


def _callbacks(run_id, bank, scratch, dose_steps, deadline, margin_seconds, log, fault_dir):
    from transformers import TrainerCallback

    from src.training.callbacks import latest_checkpoint

    class DoseSaver(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            step = state.global_step
            if step not in dose_steps:
                return
            remote = f"adapters/{run_id}/step_{step:04d}"
            if bank.exists(f"{remote}/adapter_model.safetensors"):
                return
            out = scratch / "dose" / f"step_{step:04d}"
            model.save_pretrained(str(out))
            _to_fp16_adapter(out)
            (out / "step.json").write_text(json.dumps({
                "step": step, "epoch": state.epoch,
                "log_tail": state.log_history[-3:],
            }), encoding="utf-8")
            bank.put_dir(out, remote, f"{run_id}: dose adapter step {step}")
            log(f"  [{run_id}] banked dose adapter at step {step}")

    class ResumeBanker(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            ckpt = latest_checkpoint(args.output_dir)
            if ckpt is None:
                return
            ckpt = Path(ckpt)
            prefix = f"resume/{run_id}"
            old = [f for f in bank.list(prefix + "/") if not f.startswith(f"{prefix}/{ckpt.name}/")]
            # one commit: the new checkpoint in, the previous one out
            bank.put_dir(ckpt, f"{prefix}/{ckpt.name}", f"{run_id}: resume checkpoint {ckpt.name}",
                         delete=old)
            log(f"  [{run_id}] banked resume checkpoint {ckpt.name}")

    class FaultAndDeadline(TrainerCallback):
        stopped = False

        def on_step_end(self, args, state, control, **kwargs):
            fault_point("train", state.global_step, fault_dir)
            if deadline is not None and deadline.remaining_seconds < margin_seconds:
                control.should_save = True
                control.should_training_stop = True
                FaultAndDeadline.stopped = True

    return [DoseSaver(), ResumeBanker(), FaultAndDeadline()], FaultAndDeadline


def train(run_id: str, cfg: dict, bank, scratch_root: Path, *, dose_steps=(), resume_every=50,
          deadline=None, margin_seconds=900, log=print) -> dict:
    """Train (or resume) one run. Returns ``{"status": "done"|"deadline", ...}``."""
    import torch
    from transformers import set_seed

    from src.training.callbacks import ResumeCallback, latest_checkpoint
    from src.training.model import create_lora_config, load_tokenizer

    if torch.cuda.device_count() > 1:
        # the Trainer would wrap the model in DataParallel: per-device batch x
        # GPUs changes the recipe's effective batch (and breaks on merged LoRA)
        raise RuntimeError(
            f"training sees {torch.cuda.device_count()} GPUs; run it with exactly one "
            "visible (CUDA_VISIBLE_DEVICES=<n>), as src.seedrep.launch does per worker")

    scratch = Path(scratch_root) / run_id
    resume_dir, final_dir = scratch / "resume", scratch / "final"
    resume_dir.mkdir(parents=True, exist_ok=True)
    if latest_checkpoint(resume_dir) is None:
        names = bank.list(f"resume/{run_id}/")
        if names:
            ckpt_name = names[0].split("/")[2]
            bank.get_dir(f"resume/{run_id}/{ckpt_name}", resume_dir / ckpt_name)
            log(f"  [{run_id}] pulled resume checkpoint {ckpt_name} from the bank")
    resume_path = latest_checkpoint(resume_dir)

    set_seed(cfg["seed"])
    tokenizer = load_tokenizer(cfg)
    model = _load_start_model(cfg)
    lora = create_lora_config(cfg)
    cbs, stopper = _callbacks(run_id, bank, scratch, set(dose_steps), deadline, margin_seconds, log,
                              fault_dir=Path(scratch_root))
    t = cfg["training"]
    import torch

    on_gpu = torch.cuda.is_available()
    common = dict(
        output_dir=str(resume_dir),
        num_train_epochs=t["num_train_epochs"],
        max_steps=t.get("max_steps", -1),
        learning_rate=t["learning_rate"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        save_strategy="steps",
        save_steps=resume_every,
        save_total_limit=1,
        gradient_checkpointing=t["gradient_checkpointing"],
        bf16=False,
        fp16=bool(t["fp16"]) and on_gpu,
        report_to=[],
        seed=cfg["seed"],
    )
    recipe = cfg["recipe"]
    if recipe == "dpo":
        from trl import DPOConfig, DPOTrainer

        from src.training.dpo_data import format_dpo_example, load_dpo_dataset
        from src.training.train_dpo import _build_supported

        raw = load_dpo_dataset(cfg["dataset"]["path"])
        if cfg["dataset"].get("max_samples"):
            raw = raw.select(range(cfg["dataset"]["max_samples"]))
        dataset = raw.map(format_dpo_example, remove_columns=raw.column_names)
        args = _build_supported(DPOConfig, dict(
            common, beta=cfg["dpo"]["beta"], loss_type=cfg["dpo"]["loss_type"],
            max_length=t["max_seq_length"]))
        trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=dataset,
                             processing_class=tokenizer, peft_config=lora,
                             callbacks=[ResumeCallback(), *cbs])
    else:
        from peft import get_peft_model
        from trl import SFTConfig, SFTTrainer

        from src.training.data import load_sft_dataset
        from src.training.formatting import format_chat_example
        from src.training.train_dpo import _build_supported

        raw = load_sft_dataset(cfg["dataset"]["path"])
        if cfg["dataset"].get("max_samples"):
            raw = raw.select(range(cfg["dataset"]["max_samples"]))
        model = get_peft_model(model, lora)
        if recipe == "sft_aol":
            dataset = raw.map(lambda ex: {
                "prompt": [{"role": "user", "content": ex["prompt"]}],
                "completion": [{"role": "assistant", "content": ex["response"]}],
            }, remove_columns=raw.column_names)
            extra = dict(completion_only_loss=True)
        else:
            dataset = raw.map(lambda ex: format_chat_example(ex, tokenizer),
                              remove_columns=raw.column_names)
            extra = dict(dataset_text_field="text")
        args = _build_supported(SFTConfig, dict(common, max_length=t["max_seq_length"], **extra))
        trainer = SFTTrainer(model=model, args=args, train_dataset=dataset,
                             processing_class=tokenizer, callbacks=[ResumeCallback(), *cbs])

    if resume_path:
        log(f"  [{run_id}] resuming from {Path(resume_path).name}")
    trainer.train(resume_from_checkpoint=resume_path)
    state = trainer.state
    if stopper.stopped and state.global_step < state.max_steps:
        log(f"  [{run_id}] stopped at step {state.global_step} for the session deadline")
        return {"status": "deadline", "step": state.global_step}

    if final_dir.exists():
        shutil.rmtree(final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (final_dir / "train_log.json").write_text(json.dumps(state.log_history, indent=1),
                                              encoding="utf-8")
    (final_dir / "config_used.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    shas = bank.put_dir(final_dir, f"adapters/{run_id}/final", f"{run_id}: final adapter")
    leftovers = bank.list(f"resume/{run_id}/")
    if leftovers:
        bank.delete(leftovers, f"{run_id}: training finished, drop resume checkpoint")
    shutil.rmtree(resume_dir, ignore_errors=True)     # ~0.9 GB of optimizer state
    return {"status": "done", "steps": state.global_step, "sha256": shas,
            "resumed_from": Path(resume_path).name if resume_path else None}
