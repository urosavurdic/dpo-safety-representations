"""Every `python -m src.X` command written in docs, notebooks or scripts must resolve.

Module paths embedded in prose, shell strings and notebook cells are invisible to
the import graph and to compileall, so a rename can leave them dangling and
nothing fails until a human runs the command. This test makes them checkable.

It is deliberately strict about what it scans and lenient about what counts as a
failure: only modules under `src.` are checked, and a module that exists but
cannot be imported (missing GPU dependency, say) still passes, because the point
is that the *path* is real.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# `python -m src.some.module`, with or without a leading `!` or `%` (notebooks).
COMMAND_RE = re.compile(r"python\s+-m\s+(src\.[A-Za-z0-9_.]+)")

SCANNED_DIRS = ("docs", "notebooks")
SCANNED_FILES = ("README.md", "CONTRIBUTING.md", "rerun_mechanistic_v2.sh")
SCANNED_SUFFIXES = {".md", ".sh", ".py", ".ipynb"}

# Paths that intentionally record history rather than describe live commands.
EXCLUDED_PARTS = {"archive", "private", "results", "logs", ".git", "__pycache__"}

# Placeholders used when prose talks *about* the command form rather than naming
# a module, e.g. "every `python -m src.X` in the docs is checked".
PLACEHOLDER_RE = re.compile(r"^src\.[A-Z]$|^src\.(module|name|x|X)$")


def _iter_scanned_files():
    for name in SCANNED_FILES:
        p = REPO_ROOT / name
        if p.exists():
            yield p
    for d in SCANNED_DIRS:
        root = REPO_ROOT / d
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix not in SCANNED_SUFFIXES:
                continue
            if EXCLUDED_PARTS & set(p.parts):
                continue
            yield p


def _text_of(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix != ".ipynb":
        return raw
    # Read source cells only; outputs are a record of what ran, not instructions.
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    chunks = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        chunks.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(chunks)


def _collect_references() -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    for path in _iter_scanned_files():
        for module in COMMAND_RE.findall(_text_of(path)):
            if PLACEHOLDER_RE.match(module):
                continue
            refs.setdefault(module, []).append(
                str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            )
    return refs


REFERENCES = _collect_references()


def test_some_commands_were_found():
    """Guards the test itself: a broken scanner would otherwise pass silently."""
    assert REFERENCES, "found no `python -m src.*` commands at all -- scanner is broken"


@pytest.mark.parametrize("module", sorted(REFERENCES))
def test_documented_module_exists(module):
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        spec = None
    except Exception:
        # Importable path, but importing a parent package raised. The path is
        # real, which is what this test is about.
        return
    assert spec is not None, (
        f"`python -m {module}` is documented but does not resolve.\n"
        f"Referenced from: {', '.join(sorted(set(REFERENCES[module])))}"
    )
