"""Every repo path named in a source docstring or comment must exist.

Paths written in prose are invisible to the import graph, to compileall and to
the documented-command scan, so a rename leaves them pointing at nothing and
nothing fails. Several had already rotted this way -- a module that moved
packages, a script retired to archive/, a test that was archived with its
module -- and five of them were being written verbatim into published result
JSON, so the rot propagated into artifacts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

PATH_RE = re.compile(
    r"\b(?:src|tests|notebooks|configs|docs|archive|results|data|logs)"
    r"/[A-Za-z0-9_./-]+\.(?:py|json|jsonl|md|ipynb|yaml|yml|csv|sh|npy)\b"
)

# Paths that are produced by a run rather than committed, or that name a
# pattern rather than a literal file.
IGNORE = re.compile(r"\{|\*|<|\.\.\.")


def _iter_sources():
    for p in (REPO_ROOT / "src").rglob("*.py"):
        if "__pycache__" not in p.parts:
            yield p


def _collect() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for p in _iter_sources():
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in PATH_RE.finditer(text):
            ref = m.group(0)
            if IGNORE.search(ref):
                continue
            found.setdefault(ref, set()).add(
                str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            )
    return found


REFERENCES = _collect()

# Only code and documentation are always committed. data/, results/ and logs/
# hold run products and fetched corpora that a docstring legitimately names
# before they exist, so enforcing those would fail on a clean checkout for the
# wrong reason.
ENFORCED_ROOTS = ("src/", "tests/", "notebooks/", "configs/", "docs/", "archive/")


def test_scanner_finds_references():
    """A broken scanner would otherwise make this whole file vacuously pass."""
    assert len(REFERENCES) > 20, f"only {len(REFERENCES)} paths found; scanner is broken"


@pytest.mark.parametrize(
    "ref", sorted(r for r in REFERENCES if r.startswith(ENFORCED_ROOTS))
)
def test_referenced_path_exists(ref):
    assert (REPO_ROOT / ref).exists(), (
        f"{ref} is referenced but does not exist.\n"
        f"Referenced from: {', '.join(sorted(REFERENCES[ref]))}"
    )
