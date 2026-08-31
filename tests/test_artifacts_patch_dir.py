"""
Regression test for logs/release_gap_audit.md section 5B: `artifacts` was
briefly a tracked 0-byte regular file (introduced in commit b94d9b7),
which made `mkdir -p artifacts/patches` fail and broke the documented
`git am`/patch-output workflow in logs/RESUME_PROMPT.md. This guards
against that regressing silently.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_artifacts_is_a_directory_not_a_file():
    artifacts = REPO_ROOT / "artifacts"
    assert not artifacts.is_file(), (
        "'artifacts' must not be a tracked regular file -- it blocks "
        "mkdir -p artifacts/patches (see logs/release_gap_audit.md 5B)"
    )


def test_artifacts_patches_directory_exists_and_is_writable(tmp_path_factory):
    patches_dir = REPO_ROOT / "artifacts" / "patches"
    assert patches_dir.is_dir()
    probe = patches_dir / ".write_probe_tmp"
    try:
        probe.write_text("ok")
        assert probe.read_text() == "ok"
    finally:
        probe.unlink(missing_ok=True)


def test_gitignore_excludes_patch_files_from_the_repo():
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "artifacts/patches/*.patch" in gitignore, (
        "patch files must be ignored -- the reproducibility rule requires "
        "the patch itself never be committed, only its SHA-256"
    )
