"""
Shared generation utilities for qualitative and behavioral evaluation.
"""

import torch


def build_generation_prompt(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_generation_eos_ids(tokenizer):
    """
    Stop generation at either the tokenizer EOS token or the ChatML
    assistant-turn boundary.
    """
    eos_ids = {tokenizer.eos_token_id}

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    if (
        im_end_id is not None
        and im_end_id != tokenizer.unk_token_id
    ):
        eos_ids.add(im_end_id)

    return list(eos_ids)


def generate(model, tokenizer, prompt, max_new_tokens=150):
    text = build_generation_prompt(tokenizer, prompt)

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    eos_ids = get_generation_eos_ids(tokenizer)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()