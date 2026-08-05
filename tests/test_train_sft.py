import json

from src.train_sft import (
    PROMPT_TEMPLATE,
    load_sft_dataset,
    save_reproducibility_artifacts,
    get_git_commit,
)


def test_prompt_template_formatting():
    formatted = PROMPT_TEMPLATE.format(prompt="What is X?", response="X is Y.")
    assert "### Instruction:" in formatted
    assert "What is X?" in formatted
    assert "X is Y." in formatted


def test_load_sft_dataset_respects_max_samples(tmp_path):
    p = tmp_path / "data.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"prompt": f"q{i}", "response": f"a{i}"}) + "\n")
    ds = load_sft_dataset(str(p), max_samples=2)
    assert len(ds) == 2
    assert "q0" in ds[0]["text"]


def test_save_reproducibility_artifacts(tmp_path):
    cfg = {"experiment_name": "test", "seed": 1}
    save_reproducibility_artifacts(cfg, tmp_path)
    assert (tmp_path / "config_used.yaml").exists()
    assert (tmp_path / "git_commit.txt").exists()


def test_get_git_commit_returns_string():
    commit = get_git_commit()
    assert isinstance(commit, str) and len(commit) > 0