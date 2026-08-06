import json

from src.training.dpo_data import load_dpo_dataset, format_dpo_example


def test_format_dpo_example():
    example = {"prompt": "What is X?", "chosen": "X is Y.", "rejected": "I won't say."}
    formatted = format_dpo_example(example)
    assert formatted["prompt"] == [{"role": "user", "content": "What is X?"}]
    assert formatted["chosen"] == [{"role": "assistant", "content": "X is Y."}]
    assert formatted["rejected"] == [{"role": "assistant", "content": "I won't say."}]


def test_load_dpo_dataset(tmp_path):
    file = tmp_path / "sample.jsonl"
    with open(file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "A", "chosen": "B", "rejected": "C"}) + "\n")
    dataset = load_dpo_dataset(str(file))
    assert len(dataset) == 1
    assert dataset[0]["prompt"] == "A"