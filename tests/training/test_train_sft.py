from src.training.train_sft import save_reproducibility_artifacts, get_git_commit

def test_save_reproducibility_artifacts(tmp_path):
    cfg = {"experiment_name": "test", "seed": 1}
    save_reproducibility_artifacts(cfg, tmp_path)
    assert (tmp_path / "config_used.yaml").exists()
    assert (tmp_path / "git_commit.txt").exists()


def test_get_git_commit_returns_string():
    commit = get_git_commit()
    assert isinstance(commit, str) and len(commit) > 0