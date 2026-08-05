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