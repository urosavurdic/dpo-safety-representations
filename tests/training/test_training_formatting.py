from transformers import AutoTokenizer

from src.training.formatting import format_chat_example


def test_chat_formatting():

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B"
    )

    example = {
        "prompt": "Hello",
        "response": "Hi!"
    }

    formatted = format_chat_example(
        example,
        tokenizer,
    )

    assert "text" in formatted
    assert "Hello" in formatted["text"]
    assert "Hi!" in formatted["text"]