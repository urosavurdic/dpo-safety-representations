import json

from src.training.data import load_sft_dataset


def test_load_dataset(tmp_path):

    file = tmp_path / "sample.jsonl"

    with open(file, "w") as f:

        f.write(
            json.dumps(
                {
                    "prompt": "A",
                    "response": "B",
                }
            )
            + "\n"
        )

    dataset = load_sft_dataset(str(file))

    assert len(dataset) == 1
    assert dataset[0]["prompt"] == "A"