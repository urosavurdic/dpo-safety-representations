from pathlib import Path

from src.training.callbacks import latest_checkpoint


def test_latest_checkpoint(tmp_path):

    (tmp_path / "checkpoint-100").mkdir()

    (tmp_path / "checkpoint-400").mkdir()

    (tmp_path / "checkpoint-200").mkdir()

    latest = latest_checkpoint(tmp_path)

    assert latest.endswith("checkpoint-400")


def test_no_checkpoint(tmp_path):

    assert latest_checkpoint(tmp_path) is None