"""Bind ``results/`` and the Hugging Face cache to a persistent Drive folder.

Colab VMs are ephemeral: anything under the repo checkout (``results/``, the
``~/.cache/huggingface`` weights) is lost when the runtime recycles. Every T4
session notebook calls :func:`bind` as its first real step, before any
extract / generate / analysis, so that:

* ``results/`` becomes a **symlink into Drive** - the pipeline writes straight
  to Drive by its normal relative path, no pipeline code changes;
* a disconnect mid-run resumes from the first unfinished shard on the next
  session (``v2_pipeline`` skips finished stages/shards);
* the ~3 GB base model + LoRA adapters download **once total** (via
  ``HF_HOME``), not once per session.

:func:`bind` is **idempotent** - safe to call again mid-session, and safe to
call after work has already been done on the ephemeral ``results/`` (it merges
that work into Drive before switching to the symlink, so nothing is lost).

Reproducibility / automation:
* ``drive_root`` defaults to ``/content/drive/MyDrive/dpo_v2`` but can be
  overridden by the ``DPO_DRIVE_ROOT`` env var (independent attempts, or a
  shared-folder path when switching Google accounts).
* ``persist_hf_cache=False`` keeps the HF cache on ephemeral disk (saves ~5 GB
  of Drive quota; costs one ~3 GB re-download per fresh session).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/dpo_v2"
_MOUNT_HINT = "/content/drive/MyDrive"


class DriveNotMountedError(RuntimeError):
    """Raised when the Drive mount point is absent (not Colab, or Drive not mounted)."""


def _hf_source_cache() -> Path:
    return Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))


def merge_into_drive(local: str | Path, drive_results: str | Path) -> bool:
    """Copy everything under ``local`` into ``drive_results`` (merge, never
    delete Drive-only content; local wins on name clashes). Returns True if the
    merge added any file/dir that Drive did not already have. Symlink-free, so
    it is unit-testable off Colab."""
    local = Path(local)
    drive_results = Path(drive_results)
    drive_results.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        return False
    before = {p.relative_to(drive_results).as_posix() for p in drive_results.rglob("*")}
    shutil.copytree(local, drive_results, dirs_exist_ok=True)
    after = {p.relative_to(drive_results).as_posix() for p in drive_results.rglob("*")}
    return bool(after - before)


HF_CACHE_MIN_FREE_GIB = 6.0


def bind(
    drive_root: str | None = None,
    *,
    persist_hf_cache: bool | None = None,
    results_dir: str | Path = "results",
    require_mount: bool = True,
) -> dict:
    """Point ``results_dir`` at ``<drive_root>/results`` (symlink) and, when
    persisting the HF cache, ``HF_HOME`` at ``<drive_root>/hf_cache``.

    ``persist_hf_cache``:
      * ``None`` (default) - auto: persist the ~3-5 GB HF cache to Drive only
        if the Drive volume has at least ``HF_CACHE_MIN_FREE_GIB`` free;
        otherwise keep it on ephemeral disk (weights re-download per session)
        and print a warning.
      * ``True`` / ``False`` - force it on / off.

    Returns a status dict: ``status`` is ``"bound"`` (just wired up),
    ``"already_bound"`` (was a symlink), or ``"merged_then_bound"`` (had local
    work that was copied into Drive first); ``hf_cache_note`` explains the
    auto-decision when one was made.
    """
    root = Path(drive_root or os.environ.get("DPO_DRIVE_ROOT") or DEFAULT_DRIVE_ROOT)

    if require_mount and not Path(_MOUNT_HINT).is_dir() and not root.parent.is_dir():
        raise DriveNotMountedError(
            f"Drive mount point {root.parent} is missing. On Colab run "
            "`from google.colab import drive; drive.mount('/content/drive')` "
            "first. Off Colab, pass require_mount=False with a local drive_root."
        )

    d_results = root / "results"
    d_hf = root / "hf_cache"
    d_results.mkdir(parents=True, exist_ok=True)

    hf_cache_note = None
    if persist_hf_cache is None:
        try:
            free_gib = shutil.disk_usage(root).free / (1024 ** 3)
        except OSError:
            free_gib = float("inf")
        persist_hf_cache = free_gib >= HF_CACHE_MIN_FREE_GIB
        hf_cache_note = (
            f"auto: {free_gib:.1f} GiB free on Drive -> "
            + ("persisting HF cache" if persist_hf_cache
               else f"HF cache kept EPHEMERAL (< {HF_CACHE_MIN_FREE_GIB} GiB free); "
                    "weights will re-download each fresh session")
        )
        if not persist_hf_cache:
            print(f"[colab_persist] WARNING: {hf_cache_note}")
    if persist_hf_cache:
        d_hf.mkdir(parents=True, exist_ok=True)

    lr = Path(results_dir)
    if lr.is_symlink():
        status = "already_bound"
    else:
        status = "bound"
        if lr.exists():
            # merge whatever this session/checkout produced INTO Drive before
            # switching, so no local work is ever lost regardless of what Drive
            # already holds.
            if merge_into_drive(lr, d_results):
                status = "merged_then_bound"
            shutil.rmtree(lr)
        lr.symlink_to(d_results, target_is_directory=True)

    hf_home = None
    if persist_hf_cache:
        hf_home = str(d_hf)
        os.environ["HF_HOME"] = hf_home
        src = _hf_source_cache()
        # one-time: seed the Drive cache from a session that already has weights
        if src.exists() and src.resolve() != d_hf.resolve() and not any(d_hf.iterdir()):
            shutil.copytree(src, d_hf, dirs_exist_ok=True)

    return {
        "status": status,
        "drive_root": str(root),
        "results": str(lr.resolve()),
        "hf_home": hf_home,
        "persist_hf_cache": bool(persist_hf_cache),
        "hf_cache_note": hf_cache_note,
    }


def status_line(info: dict) -> str:
    return (
        f"[colab_persist] {info['status']}: results/ -> {info['results']}"
        + (f" | HF_HOME -> {info['hf_home']}" if info.get("hf_home") else " | HF cache: ephemeral")
    )
