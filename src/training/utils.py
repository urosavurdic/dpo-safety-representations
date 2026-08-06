from pathlib import Path
import shutil
import subprocess

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown (not a git repo, or git unavailable)"


def save_reproducibility_artifacts(cfg, output_dir):
    ensure_dir(output_dir)
    with open(Path(output_dir) / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    with open(Path(output_dir) / "git_commit.txt", "w", encoding="utf-8") as f:
        f.write(get_git_commit() + "\n")
    if Path("requirements.txt").exists():
        shutil.copy("requirements.txt", Path(output_dir) / "requirements.txt")