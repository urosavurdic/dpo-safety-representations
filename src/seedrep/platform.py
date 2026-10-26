"""Where am I running? Paths, secrets and GPUs for Kaggle, Colab or a laptop.

Every other module asks this one rather than hard-coding ``/kaggle`` or
``/content``, so the same notebook and the same worker run unchanged on any of
the three, and a job started on one can be finished on another.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.seedrep.rows import ROOT


def detect() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content").exists() and _importable("google.colab"):
        return "colab"
    return "local"


def _importable(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@dataclass(frozen=True)
class Paths:
    #: large, disposable: training checkpoints before they are banked
    scratch: Path
    #: small working copies of banked files
    work: Path


def paths(kind: str | None = None) -> Paths:
    kind = kind or detect()
    override = os.environ.get("SEEDREP_SCRATCH")
    if kind == "kaggle":
        scratch, work = Path("/tmp/seedrep_scratch"), Path("/kaggle/working/seedrep")
    elif kind == "colab":
        scratch, work = Path("/tmp/seedrep_scratch"), Path("/content/seedrep")
    else:
        scratch, work = ROOT / ".seedrep_cache" / "scratch", ROOT / ".seedrep_cache" / "work"
    if override:
        scratch = Path(override)
    if os.environ.get("SEEDREP_WORK"):
        work = Path(os.environ["SEEDREP_WORK"])
    scratch.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    return Paths(scratch=scratch, work=work)


def secret(name: str, kind: str | None = None) -> str | None:
    """Environment first, then the platform's secret store; None if absent."""
    if os.environ.get(name):
        return os.environ[name]
    kind = kind or detect()
    try:
        if kind == "kaggle":
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret(name)
        if kind == "colab":
            from google.colab import userdata

            return userdata.get(name)
    except Exception:
        return None
    return None


def gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0
