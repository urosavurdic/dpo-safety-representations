"""Generate notebooks/seedrep.ipynb (never edit the .ipynb by hand).

    python notebooks/_generate_seedrep.py --commit <40-char sha>

One notebook for Kaggle (T4 x2) and Colab (one GPU). Every cell is safe to
rerun: finished work is in the bank and is skipped.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = """# Seed replication and dose curves (resumable on Kaggle or Colab)

Retrains the safety-SFT and DPO arms that start from M1, under seeds 1 and 2
(the original runs are seed 42). It saves checkpoints along the way, measures
each one, then generates and judges the endpoints. All results go to a
private Hugging Face dataset repo (the **bank**).

**If a session dies, open this notebook anywhere, run the cells from the top
again, and it continues where it stopped.** Finished work is never redone.

**Before the first run:**
1. Kaggle: *Settings -> Accelerator -> GPU T4 x2*, *Internet on*.
   *Add-ons -> Secrets*: `HF_TOKEN`, a write token for the account that has
   accepted the licences of `allenai/wildguard` and `google/gemma-2b`.
   Colab: *Runtime -> T4 GPU*; key icon -> `HF_TOKEN`.
2. Inputs: upload `seedrep_inputs.zip`. On Kaggle, as a private Dataset added
   to this notebook. On Colab, to `MyDrive/seedrep/`.
3. Set `BANK` below once, and keep it the same on every platform.

**Order:** run all cells. The preflight cell asserts everything, including
parity with the stored activations, before any long job starts. On Kaggle,
use *Save Version -> Save & Run All (Commit)* so the run survives a closed
browser.
"""

CONFIG = '''# ---- the only cell you edit ---------------------------------------------
BANK = "hf://urosavurdic/dpo-seedrep-bank"   # private HF dataset repo, created if missing
# P1 seeds (branch 1) + behaviour; P2 injection; P3 branch-2 seeds; P4 response-only SFT.
# Queue all four: workers take them in this order and stop cleanly at the deadline;
# the next session continues where this one stopped.
PRIORITIES = ["P1", "P2", "P3", "P4"]
RUN_PREFLIGHT = True        # the first time on a new platform or image
REPO_URL = "https://github.com/urosavurdic/dpo-safety-representations.git"
PINNED_COMMIT = "__PINNED_COMMIT__"
import os, re
assert re.fullmatch(r"[0-9a-f]{40}", PINNED_COMMIT), "PINNED_COMMIT must be a full 40-char SHA"
ON_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or os.path.exists("/kaggle/working")
ON_COLAB = not ON_KAGGLE and os.path.exists("/content")
# stop cleanly before the platform kills the session (Kaggle 12 h; Colab varies)
DEADLINE_MINUTES = 630 if ON_KAGGLE else 300
print("platform:", "kaggle" if ON_KAGGLE else "colab" if ON_COLAB else "local")'''

CLONE = '''# ---- code at a pinned commit ---------------------------------------------
import os, subprocess
ROOT = "/kaggle/working/dpo-safety-representations" if ON_KAGGLE else "/content/dpo-safety-representations"
if not os.path.exists(ROOT):
    subprocess.run(["git", "clone", "--quiet", REPO_URL, ROOT], check=True)
subprocess.run(["git", "-C", ROOT, "fetch", "--quiet", "origin"], check=True)
subprocess.run(["git", "-C", ROOT, "checkout", "--quiet", PINNED_COMMIT], check=True)
head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
assert head == PINNED_COMMIT, (head, PINNED_COMMIT)
os.chdir(ROOT)
print("at", head)'''

INSTALL = '''# ---- libraries: pinned, without touching the preinstalled torch -------------
# Reinstalling requirements upgraded torch under a running kernel once
# (notebook 11). Pinning torch to the version already installed prevents it.
import subprocess, sys
torch_version = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__.split('+')[0])"],
                               capture_output=True, text=True).stdout.strip()
pins = ["transformers==5.14.1", "trl==1.9.2", "peft==0.20.0", "accelerate==1.14.0",
        "bitsandbytes>=0.46.1", "datasets>=3.0", f"torch=={torch_version}"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pins], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
print("installed with torch", torch_version)'''

GATE = '''# ---- import gate: fail here, not three hours in ------------------------------
import torch, transformers, trl, peft, bitsandbytes
print({m.__name__: m.__version__ for m in (torch, transformers, trl, peft, bitsandbytes)})
assert torch.cuda.is_available(), "no GPU: set the accelerator"
N_GPUS = torch.cuda.device_count()
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPUS)])'''

SECRETS = '''# ---- token into the environment (subprocesses read it there) ----------------
import os
from src.seedrep import platform as plat
token = plat.secret("HF_TOKEN")
assert token, "add the HF_TOKEN secret (Kaggle Add-ons -> Secrets, or the Colab key icon)"
os.environ["HF_TOKEN"] = token
os.environ["SEEDREP_BANK"] = BANK
print("token ok")'''

INPUTS = '''# ---- inputs: training files, held-out pairs, fixtures (SHA-checked) ------------
# They are in the repository; the zip bundle is only a fallback for a clone
# that lacks them (upload seedrep_inputs.zip as a Kaggle dataset or to Drive).
import glob, hashlib, json, os, zipfile
MANIFEST = "data/seedrep/inputs_manifest.json"
def verify():
    manifest = json.load(open(MANIFEST))
    bad = [n for n, d in manifest.items()
           if not os.path.exists(n) or hashlib.sha256(open(n, "rb").read()).hexdigest() != d]
    return manifest, bad
manifest, bad = verify()
if bad:
    if ON_COLAB and not os.path.exists("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    zips = (glob.glob("/kaggle/input/**/seedrep_inputs.zip", recursive=True)
            + glob.glob("/content/drive/MyDrive/seedrep/seedrep_inputs.zip"))
    assert zips, f"inputs missing or changed: {bad}; upload seedrep_inputs.zip"
    zipfile.ZipFile(zips[0]).extractall(".")
    manifest, bad = verify()
assert not bad, f"sha256 mismatch: {bad}"
print(f"inputs ok: {len(manifest)} files verified")'''

STATUS = '''# ---- where the run stands (safe to run any time, also from a laptop) ------------
!python -m src.seedrep.status --bank "$BANK" --priorities {" ".join(PRIORITIES)}'''

PREFLIGHT = '''# ---- Run 0: preflight (asserts; ~25 min; its last step is real work) ------------
if RUN_PREFLIGHT:
    !python -m src.seedrep.preflight --bank "$BANK"'''

LAUNCH = '''# ---- the run: one worker per GPU, pinned; progress printed every 5 min -----------
!python -m src.seedrep.launch --bank "$BANK" --priorities {" ".join(PRIORITIES)} --deadline-minutes {DEADLINE_MINUTES}'''

ANALYZE = '''# ---- when P1 is done: the CPU analysis (also runnable locally) -----------------
!python -m src.seedrep.analyze --bank "$BANK"
# writes results/stagecmp/seed_replication.json; copy it into the local repo'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": text.splitlines(keepends=True)}


def build(commit: str) -> dict:
    cells = [md(INTRO), code(CONFIG.replace("__PINNED_COMMIT__", commit)), code(CLONE),
             code(INSTALL), code(GATE), code(SECRETS), code(INPUTS), code(STATUS),
             code(PREFLIGHT), code(LAUNCH), code(STATUS), code(ANALYZE)]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", type=Path, default=HERE / "seedrep.ipynb")
    args = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise SystemExit("--commit must be a full 40-character SHA")
    nb = build(args.commit)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            # IPython shell lines are not Python; check the rest compiles
            src = "".join(cell["source"])
            plain = "\n".join(
                ln[:len(ln) - len(ln.lstrip())] + "pass" if ln.lstrip().startswith("!") else ln
                for ln in src.splitlines())
            compile(plain, "<cell>", "exec")
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    json.loads(text)                      # round-trips: no hand-edited JSON
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(nb['cells'])} cells, pinned {args.commit[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
