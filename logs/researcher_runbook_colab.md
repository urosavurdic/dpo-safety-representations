# Colab Researcher Runbook

Run this **after** the local benchmark-finalization steps are complete and the
frozen benchmark is committed to the branch.

## Cell 1 — Clone and checkout

```python
import subprocess, os

REPO = "https://github.com/urosavurdic/dpo-safety-representations.git"
BRANCH = "agent/c-quadrant-end-to-end-e0e2317a"

subprocess.run(["git", "clone", REPO], check=True)
os.chdir("dpo-safety-representations")
subprocess.run(["git", "fetch", "origin"], check=True)
subprocess.run(["git", "checkout", BRANCH], check=True)
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("Setup complete")
```

## Cell 2 — Verify benchmark and split hashes

```python
import json, hashlib, subprocess
from pathlib import Path

latest = json.loads(Path("data/frozen_v2/LATEST_BENCHMARK.json").read_text())
bench_path = latest["benchmark_path"]
expected_sha = latest["benchmark_sha256"]
actual_sha = hashlib.sha256(Path(bench_path).read_bytes()).hexdigest()

assert actual_sha == expected_sha, f"Benchmark hash mismatch!\nExpected: {expected_sha}\nActual:   {actual_sha}"
print(f"Benchmark hash ✓: {actual_sha}")

split_manifest = json.loads(Path("logs/direction_split_manifest.json").read_text())
split_sha_recorded = split_manifest.get("split_manifest_sha256")
split_sha_actual = hashlib.sha256(Path("logs/direction_split_manifest.json").read_bytes()).hexdigest()
assert split_sha_actual == split_sha_recorded, "Split manifest hash mismatch!"
print(f"Split manifest hash ✓: {split_sha_actual}")

val_status = json.loads(Path("logs/benchmark_validation_status.json").read_text())
print(f"technical_benchmark_status: {val_status['technical_benchmark_status']}")
print(f"reduced_cue_evidence_status: {val_status['reduced_cue_evidence_status']}")
```

## Cell 3 — Check model/checkpoint availability

```python
# Adjust checkpoint paths to your Drive mount
import os

DRIVE_MOUNT = "/content/drive/MyDrive/dpo_safety_checkpoints"
REQUIRED_STAGES = ["M3", "M3_alt"]

for stage in REQUIRED_STAGES:
    ckpt = os.path.join(DRIVE_MOUNT, stage)
    exists = os.path.exists(ckpt)
    print(f"  {stage}: {'✓ found' if exists else '✗ MISSING — abort'}")
```

## Cell 4 — GPU gate dry run

```python
import subprocess
result = subprocess.run(
    ["bash", "rerun_mechanistic_v2.sh", "--dry-run"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
```

## Cell 5 — Run M3 and M3_alt experiments

```python
# This cell runs GPU experiments. Ensure all gate checks passed first.
import subprocess
result = subprocess.run(
    ["bash", "rerun_mechanistic_v2.sh", "--stage", "M3", "--stage", "M3_alt"],
    capture_output=True, text=True
)
print(result.stdout[-3000:])   # tail
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
    raise RuntimeError("rerun_mechanistic_v2.sh failed")
```

## Cell 6 — Copy large artifacts to Drive

```python
import shutil, os
from pathlib import Path

DRIVE_OUT = "/content/drive/MyDrive/dpo_safety_v2_results"
os.makedirs(DRIVE_OUT, exist_ok=True)

# Copy .npy arrays (large — DO NOT commit to Git)
for npy in Path("results/activations").glob("*.npy"):
    shutil.copy(npy, os.path.join(DRIVE_OUT, npy.name))
    print(f"Copied {npy.name}")

# Record hashes of everything copied
import hashlib, json
manifest_path = os.path.join(DRIVE_OUT, "artifact_hashes.json")
hashes = {}
for f in Path(DRIVE_OUT).glob("*.npy"):
    hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
Path(manifest_path).write_text(json.dumps(hashes, indent=2))
print("Hashes recorded:", manifest_path)
```

## Cell 7 — Commit ONLY small artifacts to GitHub

```python
# Commit only: result JSONs, manifests, summaries, hashes
# NEVER commit: .npy arrays, model weights, large CSVs

import subprocess

small_artifacts = [
    "results/behavioral_eval/raw_v2.json",
    "results/refusal_direction/",
    "results/probes/",
    "logs/benchmark_validation_status.json",
    "logs/benchmark_validation_report.md",
]

subprocess.run(["git", "add"] + small_artifacts, check=False)
subprocess.run(["git", "commit", "-m", "results: v2 mechanistic rerun M3/M3_alt"], check=True)
```

## Cell 8 — Create GPU-result branch or patch

```python
import subprocess

BASE = "e0e2317a"
CODE = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:8]
patch_path = f"artifacts/patches/c_quadrant_gpu_{BASE}_{CODE}.patch"

subprocess.run(["mkdir", "-p", "artifacts/patches"], check=True)
with open(patch_path, "w") as f:
    subprocess.run(
        ["git", "diff", "--binary", BASE, "HEAD",
         "--", ".", ":(exclude)artifacts/patches/*.patch"],
        stdout=f, check=True
    )

import hashlib
patch_sha = hashlib.sha256(open(patch_path, "rb").read()).hexdigest()
print(f"Patch: {patch_path}")
print(f"SHA-256: {patch_sha}")
```

## Important: what NOT to commit

- `*.npy` activation arrays (store in Drive only)
- Model checkpoint files
- Any file > 50 MB
- The benchmark JSONL if it was already committed in the local step

For every large artifact stored in Drive, record:
- Drive path
- File size
- SHA-256
- Benchmark SHA-256 it was generated from
- Split manifest SHA-256
- Stage and model identifier
- Git commit of the code that generated it
