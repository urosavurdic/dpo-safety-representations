"""Rendering prompt/response pairs into the model's chat template.

The base model has no chat template while every fine-tuned stage does, so the
same literal template tokens are applied throughout. Otherwise a comparison
across stages would measure surface form rather than the model.
"""
from typing import Dict


def format_chat_example(example: Dict, tokenizer) -> Dict:
    """
    Convert a prompt/response pair into Qwen chat format.

    Input
    -----
    {
        "prompt": "...",
        "response": "..."
    }

    Output
    ------
    {
        "text": "<formatted conversation>"
    }
    """

    messages = [
        {
            "role": "user",
            "content": example["prompt"],
        },
        {
            "role": "assistant",
            "content": example["response"],
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}