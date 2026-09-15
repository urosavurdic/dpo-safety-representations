"""Run provenance: which commit produced an artifact."""

import subprocess

__all__ = ["get_git_commit"]


def get_git_commit() -> str:
    """Return the current HEAD sha, or ``"unknown"`` outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"
