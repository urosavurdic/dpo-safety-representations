from transformers import AutoTokenizer

from src.eval_generation import (
    build_generation_prompt,
    get_generation_eos_ids,
)


TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")


def test_get_generation_eos_ids_includes_im_end():
    eos_ids = get_generation_eos_ids(TOKENIZER)

    im_end_id = TOKENIZER.convert_tokens_to_ids("<|im_end|>")

    assert im_end_id in eos_ids


def test_get_generation_eos_ids_includes_tokenizer_eos():
    eos_ids = get_generation_eos_ids(TOKENIZER)

    assert TOKENIZER.eos_token_id in eos_ids


def test_build_generation_prompt_ready_for_assistant_turn():
    prompt = build_generation_prompt(
        TOKENIZER,
        "Hello",
    )

    assert "<|im_start|>assistant" in prompt