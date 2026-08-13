"""
Shared generation utilities for eval_qualitative.py and eval_behavioral.py.
"""
import torch


def build_generation_prompt(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_batch(model, tokenizer, prompts, max_new_tokens=200):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # required for correct batch generation with a decoder-only model
    try:
        texts = [build_generation_prompt(tokenizer, p) for p in prompts]
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        eos_ids = get_generation_eos_ids(tokenizer)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )
        results = []
        for i in range(len(prompts)):
            generated = outputs[i][inputs["input_ids"].shape[1]:]
            results.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
        return results
    finally:
        tokenizer.padding_side = original_padding_side

def get_generation_eos_ids(tokenizer):
    """
    tokenizer.eos_token_id may not coincide with <|im_end|> for this
    checkpoint - confirmed true for Qwen2.5-1.5B (151643 vs 151645).
    Include both so generation reliably stops at a ChatML turn boundary
    for models that were trained to produce <|im_end|> (M1-M3). M0 may
    still not stop early even with this fix - it was never trained to
    emit <|im_end|> in response to this format, so there's no signal to
    catch even when we're listening for the right token.
    """
    eos_ids = {tokenizer.eos_token_id}
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_ids.add(im_end_id)
    return list(eos_ids)


def generate(model, tokenizer, prompt, max_new_tokens=200):
    text = build_generation_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt")
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
    return tokenizer.decode(generated, skip_special_tokens=True).strip()