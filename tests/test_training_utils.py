from src.training.utils import load_config


def test_load_config(tmp_path):

    cfg = tmp_path / "config.yaml"

    cfg.write_text(
        """
learning_rate: 0.0002
epochs: 2
""",
        encoding="utf-8",
    )

    loaded = load_config(cfg)

    assert loaded["learning_rate"] == 0.0002
    assert loaded["epochs"] == 2