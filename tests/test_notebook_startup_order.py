"""Regression test for notebooks/colab_unified_analysis.ipynb's section
order.

Task 3 (Colab/T4 reproducible startup release) requires the notebook to
walk through: fresh clone -> pinned commit -> persistent storage ->
environment setup -> benchmark verification -> split verification ->
focused test gate -> status -> calibration -> resumable run command.

Before this fix, the focused pytest test gate was bundled into the
"Install dependencies, check GPU" section and ran BEFORE the benchmark/
split-manifest verification cell, instead of after it. This is a thin
structural check (markdown header order only, not execution) so future
edits to the notebook cannot silently drift back to that ordering without
this test catching it - it does not run the notebook itself, since that
needs a GPU/Colab environment this repository's CPU-only test suite does
not have.
"""
import json
import re
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/colab_unified_analysis.ipynb")

# The exact required order, matched against each markdown cell's
# "## N. <title>" header, title text only (numbers are re-derived from
# position, not hardcoded here, so a renumbering that keeps the same
# relative order still passes).
REQUIRED_SECTION_TITLES = [
    "Mount Drive",
    "Clone and pin the exact commit",
    "Persistent storage",
    "Install dependencies, check GPU",
    "Benchmark, split-manifest, and gate verification",
    "Focused test gate",
    "Current progress",
    "Throughput calibration",
    "Dry run",
    "Live run",
    "Post-run: bridge outputs, re-validate, session summary",
]

_HEADER_RE = re.compile(r"^##\s+\d+\.\s+(.*)$")


def _load_section_titles():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    titles = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        first_line = "".join(cell["source"]).splitlines()[0]
        m = _HEADER_RE.match(first_line)
        if m:
            titles.append(m.group(1))
    return titles


def test_section_titles_match_required_order_exactly():
    assert _load_section_titles() == REQUIRED_SECTION_TITLES


def test_section_numbers_are_sequential_starting_at_one():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    numbers = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        first_line = "".join(cell["source"]).splitlines()[0]
        m = re.match(r"^##\s+(\d+)\.\s+", first_line)
        if m:
            numbers.append(int(m.group(1)))
    assert numbers == list(range(1, len(numbers) + 1))


def test_focused_test_gate_cell_follows_benchmark_verification_cell():
    """The pytest V2_TEST_SCOPE code cell must be the next code cell after
    the benchmark/split-manifest verification code cell, not before it."""
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    def _find_code_cell(needle):
        for i, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            if needle in "".join(cell["source"]):
                return i
        raise AssertionError(f"no code cell containing {needle!r} found")

    benchmark_idx = _find_code_cell("src.validate_benchmark_v2")
    test_gate_idx = _find_code_cell("V2_TEST_SCOPE = [")

    assert test_gate_idx > benchmark_idx, (
        "focused test gate cell must come after the benchmark verification "
        f"cell (benchmark at index {benchmark_idx}, test gate at "
        f"index {test_gate_idx})"
    )
