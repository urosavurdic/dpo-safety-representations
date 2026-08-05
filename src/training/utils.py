from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to yaml config.

    Returns
    -------
    dict
    """

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path):
    """
    Create directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)