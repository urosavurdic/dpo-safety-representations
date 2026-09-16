"""Loading SFT training data from local JSONL.

Each row is a prompt/response pair. Kept separate from the DPO loader because
the two formats diverge: preference data carries chosen/rejected instead.
"""
from datasets import load_dataset


def load_sft_dataset(path: str):
    """
    Load a local JSONL SFT dataset.

    Expected columns

    prompt
    response
    """

    dataset = load_dataset(
        "json",
        data_files=path,
        split="train",
    )

    return dataset