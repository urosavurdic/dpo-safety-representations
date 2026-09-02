"""Structural checks for the thin v2 T4 session notebooks (WP-NB).

Not executed (needs GPU/Colab). Verifies: all 7 notebooks exist, valid
nbformat, sequential ``## N.`` markdown headers, THIN (no analysis logic - code
cells only shell out or do trivial glue), each names the 240-270 min target,
and S6/judge notebook consumes only the consolidated manifest.
"""
import json
import re
from pathlib import Path

import pytest

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
SESSION_NOTEBOOKS = [
    "00_setup_and_verify.ipynb",
    "01_calibrate_and_extract.ipynb",
    "02_behavioral_generation.ipynb",
    "03_directions_probes_projections.ipynb",
    "04_causal.ipynb",
    "04b_judge_preflight.ipynb",
    "05_steering_manifest_judge.ipynb",
]
_HEADER = re.compile(r"^##\s+(\d+)\.\s+")


@pytest.fixture(params=SESSION_NOTEBOOKS)
def nb(request):
    path = NB_DIR / request.param
    assert path.exists(), f"missing session notebook {request.param}"
    data = json.loads(path.read_text(encoding="utf-8"))
    return request.param, data


def test_valid_nbformat(nb):
    _name, data = nb
    assert data["nbformat"] == 4
    assert isinstance(data["cells"], list) and data["cells"]


def test_section_numbers_sequential(nb):
    _name, data = nb
    nums = []
    for cell in data["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        first = "".join(cell["source"]).splitlines()[0] if cell["source"] else ""
        m = _HEADER.match(first)
        if m:
            nums.append(int(m.group(1)))
    assert nums == list(range(1, len(nums) + 1)), nums


def test_notebook_is_thin(nb):
    name, data = nb
    for cell in data["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        thin = (
            src.lstrip().startswith("!")
            or src.lstrip().startswith("#")
            or "import " in src
            or "print(" in src
            or "drive.mount" in src
            or "os.makedirs" in src
            or "V2_TEST_SCOPE" in src
        )
        assert thin, f"{name}: code cell is not a thin shell:\n{src}"


QUICK_NOTEBOOKS = {"00_setup_and_verify.ipynb", "04b_judge_preflight.ipynb"}


def test_targets_the_240_270_window(nb):
    name, data = nb
    text = "\n".join("".join(c["source"]) for c in data["cells"] if c["cell_type"] == "markdown")
    if name in QUICK_NOTEBOOKS:
        # setup / preflight are short by design and say so
        assert "240-270" in text, f"{name} should still reference the S1-S5 target for context"
        assert ("Quick session" in text or "No full run here" in text), \
            f"{name} must flag that it is NOT a 240-270 min session"
    else:
        assert "240-270" in text and "300" in text, f"{name} must state the 240-270 / hard-300 target"


def test_judge_notebook_uses_only_the_consolidated_manifest():
    data = json.loads((NB_DIR / "05_steering_manifest_judge.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(c["source"]) for c in data["cells"])
    assert "--response-manifest results/manifests/consolidated_" in text
    assert "--build-consolidated" in text
    assert "results/behavioral_eval/raw.json" not in text  # no 370-era glob
