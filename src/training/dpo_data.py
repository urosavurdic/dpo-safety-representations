from datasets import load_dataset


def load_dpo_dataset(path: str):
    return load_dataset("json", data_files=path, split="train")


def format_dpo_example(example):
    """
    Convert {"prompt": str, "chosen": str, "rejected": str} into TRL's
    conversational preference format. DPOTrainer applies the chat template
    internally from this - unlike format_chat_example for SFT, we don't call
    apply_chat_template ourselves, because DPO needs prompt and completions
    as separate fields (can't flatten to one "text" string the way SFT does).
    """
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "chosen": [{"role": "assistant", "content": example["chosen"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }