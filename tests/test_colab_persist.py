"""Tests for src/colab_persist.py (WP-NB persistence).

All under tmp_path with a fake "drive root" - no real Colab / Drive needed.
The symlink-creating path of bind() needs OS symlink privilege (fine on the
Colab Linux VM, not always on a Windows dev box), so those tests skip when it
is unavailable; the merge logic is tested symlink-free.
"""
import os
from pathlib import Path

import pytest

from src.colab_persist import (
    DriveNotMountedError,
    bind,
    merge_into_drive,
    status_line,
)


def _symlinks_ok(tmp_path) -> bool:
    probe = tmp_path / "_probe"
    try:
        (tmp_path / "_t").mkdir()
        probe.symlink_to(tmp_path / "_t", target_is_directory=True)
        probe.unlink()
        return True
    except (OSError, NotImplementedError):
        return False


# --- merge_into_drive (symlink-free) --------------------------------------
def test_merge_copies_local_and_reports_new_content(tmp_path):
    local = tmp_path / "results"
    (local / "activations").mkdir(parents=True)
    (local / "activations" / "M0_final.npy").write_bytes(b"x")
    drive = tmp_path / "drive" / "results"
    assert merge_into_drive(local, drive) is True
    assert (drive / "activations" / "M0_final.npy").read_bytes() == b"x"


def test_merge_preserves_drive_only_content(tmp_path):
    drive = tmp_path / "drive" / "results"
    (drive / "activations").mkdir(parents=True)
    (drive / "activations" / "M3_final.npy").write_bytes(b"prior")
    local = tmp_path / "results"
    (local).mkdir()
    (local / "README.md").write_text("seed", encoding="utf-8")
    merge_into_drive(local, drive)
    assert (drive / "activations" / "M3_final.npy").read_bytes() == b"prior"  # survives
    assert (drive / "README.md").exists()  # local seed merged in


def test_merge_drive_wins_on_name_clash_never_overwrites(tmp_path):
    # Drive holds a real 654-row session output; local is a fresh checkout
    # carrying a stale 370-era file of the same name. Drive MUST win.
    drive = tmp_path / "drive" / "results" / "activations"
    drive.mkdir(parents=True)
    (drive / "M3_metadata.json").write_text("654-row-real", encoding="utf-8")
    local = tmp_path / "results" / "activations"
    local.mkdir(parents=True)
    (local / "M3_metadata.json").write_text("370-row-stale", encoding="utf-8")
    added = merge_into_drive(tmp_path / "results", tmp_path / "drive" / "results")
    assert (drive / "M3_metadata.json").read_text() == "654-row-real"
    assert added is False  # nothing new was added


def test_merge_no_local_is_noop(tmp_path):
    assert merge_into_drive(tmp_path / "missing", tmp_path / "drive") is False


# --- bind() end to end (needs symlink privilege) ------------------------------
@pytest.fixture
def workspace(tmp_path, monkeypatch):
    if not _symlinks_ok(tmp_path):
        pytest.skip("OS symlink creation not permitted here (works on Colab)")
    repo = tmp_path / "repo"
    (repo / "results" / "activations").mkdir(parents=True)
    (repo / "results" / "activations" / "M0_final.npy").write_bytes(b"x")
    (repo / "results" / "README.md").write_text("seed", encoding="utf-8")
    drive = tmp_path / "drive_root"
    monkeypatch.chdir(repo)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("DPO_DRIVE_ROOT", raising=False)
    return repo, drive


def test_bind_moves_local_results_into_drive_and_symlinks(workspace):
    repo, drive = workspace
    info = bind(str(drive), persist_hf_cache=False, require_mount=False)
    assert info["status"] in ("bound", "merged_then_bound")
    lr = repo / "results"
    assert lr.is_symlink()
    assert lr.resolve() == (drive / "results").resolve()
    assert (drive / "results" / "activations" / "M0_final.npy").read_bytes() == b"x"


def test_bind_is_idempotent(workspace):
    repo, drive = workspace
    bind(str(drive), persist_hf_cache=False, require_mount=False)
    info2 = bind(str(drive), persist_hf_cache=False, require_mount=False)
    assert info2["status"] == "already_bound"
    assert (repo / "results").is_symlink()


def test_bind_sets_hf_home_and_seeds_cache_once(workspace, monkeypatch):
    repo, drive = workspace
    fake_hf = repo.parent / "hf_src"
    (fake_hf / "hub").mkdir(parents=True)
    (fake_hf / "hub" / "model.bin").write_bytes(b"weights")
    monkeypatch.setenv("HF_HOME", str(fake_hf))
    info = bind(str(drive), persist_hf_cache=True, require_mount=False)
    assert info["hf_home"] == str(drive / "hf_cache")
    assert os.environ["HF_HOME"] == str(drive / "hf_cache")
    assert (drive / "hf_cache" / "hub" / "model.bin").read_bytes() == b"weights"


def test_bind_respects_env_override(workspace, monkeypatch):
    repo, drive = workspace
    alt = repo.parent / "alt_drive"
    monkeypatch.setenv("DPO_DRIVE_ROOT", str(alt))
    info = bind(persist_hf_cache=False, require_mount=False)
    assert info["drive_root"] == str(alt)
    assert (repo / "results").resolve() == (alt / "results").resolve()


def test_status_line_readable(workspace):
    repo, drive = workspace
    info = bind(str(drive), persist_hf_cache=False, require_mount=False)
    assert "results/ ->" in status_line(info) and "ephemeral" in status_line(info)


# --- mount guard (no symlink needed) ---------------------------------------
def test_bind_raises_when_drive_not_mounted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(DriveNotMountedError):
        bind("/content/drive/MyDrive/dpo_v2", require_mount=True)


# --- auto HF-cache decision by free space -------------------------------------
def test_persist_hf_cache_auto_skips_when_drive_is_low(tmp_path, monkeypatch):
    import src.colab_persist as cp
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    drive = tmp_path / "d"
    monkeypatch.setattr(cp.shutil, "disk_usage",
                        lambda p: type("U", (), {"free": 1 * 1024 ** 3})())  # 1 GiB free
    if not _symlinks_ok(tmp_path):
        pytest.skip("no symlink privilege")
    info = cp.bind(str(drive), require_mount=False)  # persist_hf_cache=None -> auto
    assert info["persist_hf_cache"] is False
    assert "EPHEMERAL" in info["hf_cache_note"]
    assert not (drive / "hf_cache").exists()


def test_persist_hf_cache_auto_enables_when_drive_has_room(tmp_path, monkeypatch):
    import src.colab_persist as cp
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    drive = tmp_path / "d"
    monkeypatch.setattr(cp.shutil, "disk_usage",
                        lambda p: type("U", (), {"free": 50 * 1024 ** 3})())  # 50 GiB free
    monkeypatch.delenv("HF_HOME", raising=False)
    if not _symlinks_ok(tmp_path):
        pytest.skip("no symlink privilege")
    info = cp.bind(str(drive), require_mount=False)
    assert info["persist_hf_cache"] is True
    assert (drive / "hf_cache").exists()
