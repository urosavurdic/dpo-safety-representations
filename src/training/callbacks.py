from pathlib import Path

import torch
from transformers import TrainerCallback


class ResumeCallback(TrainerCallback):
    """
    Prints resume information at the start of training.
    """

    def on_train_begin(self, args, state, control, **kwargs):

        if state.global_step > 0:
            print(f"\nResuming from global step {state.global_step}\n")
        else:
            print("\nStarting fresh training\n")


class GPUMemoryCallback(TrainerCallback):
    """
    Logs GPU memory every logging step.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):

        if torch.cuda.is_available():

            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3

            print(
                f"[GPU] allocated={allocated:.2f} GB | reserved={reserved:.2f} GB"
            )


def latest_checkpoint(output_dir: str):
    """
    Return newest checkpoint folder or None.

    Looks for

    checkpoint-100
    checkpoint-200
    checkpoint-300
    ...

    inside output_dir.
    """

    output_dir = Path(output_dir)

    if not output_dir.exists():
        return None

    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )

    if len(checkpoints) == 0:
        return None

    return str(checkpoints[-1])