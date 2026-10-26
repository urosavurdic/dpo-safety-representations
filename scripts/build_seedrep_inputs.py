"""Build everything the GPU run needs that git does not carry.

    python scripts/build_seedrep_inputs.py

Writes (tracked, small):

* ``data/seedrep/parity_fixture.npz`` -- stored final-token activations of the
  first 32 benchmark rows at the analysis layers, for M1, M2, M1_alt, M2_alt.
  The ref jobs re-extract these rows and refuse to continue unless they agree,
  so a library or hardware difference shows up before any GPU hour is spent.
* ``data/seedrep/inject_vectors.npz`` (+ ``.json``) -- the injection test's
  vectors at layer 24, from the stored activations. With ``u`` the start
  contrast direction, ``e`` the unit direction of the part of DPO's contrast
  change orthogonal to ``u``, and ``m`` the mean activation change DPO made on
  the harmful estimation prompts:

      orth       = (m . e) e         DPO's push on harmful prompts along e
      old_dir    = |orth| u          the same size of push along the start
                                     direction (DPO's own push along u is
                                     close to zero, so it is no control)
      mean_shift = m                 DPO's whole average push
      random     = a unit vector orthogonal to u and e, scaled to |orth|

Also writes ``data/seedrep/inputs_manifest.json``: the SHA-256 of these files
and of the training and held-out files, which the notebook checks before any
GPU work. A fallback bundle, ``.seedrep_cache/seedrep_inputs.zip``, holds the
same files for a clone that lacks them.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "results" / "activations"
OUT = ROOT / "data" / "seedrep"
ZIP = ROOT / ".seedrep_cache" / "seedrep_inputs.zip"
LAYERS = (16, 20, 24, 28)
INJECT_LAYER = 24
N_PARITY = 32
BRANCHES = {"alpaca": ("M1", "M3_direct"), "dolly": ("M1_alt", "M3_direct_alt")}
PROCESSED = ("sft_safety.jsonl", "dpo_pairs.jsonl", "pku_heldout_pairs.jsonl")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ACT / "M1_metadata.json").read_text(encoding="utf-8"))
    quad = np.array([r["quadrant"] for r in meta])
    split = np.array([r.get("split") or "" for r in meta])
    a_est = np.where((quad == "A") & (split == "direction_estimation"))[0]
    d_est = np.where((quad == "D") & (split == "direction_estimation"))[0]

    fixture = {}
    for stage in ("M1", "M2", "M1_alt", "M2_alt"):
        arr = np.load(ACT / f"{stage}_final.npy", mmap_mode="r")
        fixture[f"{stage}_final"] = np.asarray(arr[:N_PARITY][:, list(LAYERS)], dtype=np.float32)
    np.savez(OUT / "parity_fixture.npz", **fixture)

    vectors, info = {}, {}
    rng = np.random.default_rng(20261025)
    for branch, (start, dpo) in BRANCHES.items():
        xs = np.asarray(np.load(ACT / f"{start}_final.npy", mmap_mode="r")[:, INJECT_LAYER], np.float64)
        xd = np.asarray(np.load(ACT / f"{dpo}_final.npy", mmap_mode="r")[:, INJECT_LAYER], np.float64)
        c_start = xs[a_est].mean(0) - xs[d_est].mean(0)
        c_dpo = xd[a_est].mean(0) - xd[d_est].mean(0)
        u = c_start / np.linalg.norm(c_start)
        change = c_dpo - c_start
        orth_part = change - (change @ u) * u
        e = orth_part / np.linalg.norm(orth_part)
        m = (xd - xs)[a_est].mean(0)
        orth = (m @ e) * e
        old_dir = np.linalg.norm(orth) * u
        r = rng.normal(size=u.shape)
        r -= (r @ u) * u + (r @ e) * e
        r *= np.linalg.norm(orth) / np.linalg.norm(r)
        for name, vec in (("old_dir", old_dir), ("orth", orth), ("mean_shift", m), ("random", r)):
            vectors[f"{branch}_{name}"] = vec.astype(np.float32)
        info[branch] = {"start_contrast_norm": float(np.linalg.norm(c_start)),
                        "contrast_change_norm": float(np.linalg.norm(change)),
                        "norms": {k: float(np.linalg.norm(v)) for k, v in
                                  (("old_dir", old_dir), ("orth", orth), ("mean_shift", m), ("random", r))},
                        "m_dot_u": float(m @ u), "m_dot_e": float(m @ e)}
    np.savez(OUT / "inject_vectors.npz", **vectors)
    (OUT / "inject_vectors.json").write_text(json.dumps({"layer": INJECT_LAYER, **info}, indent=1),
                                             encoding="utf-8", newline="\n")

    files = {f"data/processed/{n}": ROOT / "data" / "processed" / n for n in PROCESSED}
    files.update({f"data/seedrep/{p.name}": p for p in sorted(OUT.iterdir()) if p.is_file()})
    manifest = {name: sha(path) for name, path in files.items()
                if not name.endswith("inputs_manifest.json")}
    (OUT / "inputs_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                              encoding="utf-8", newline="\n")
    ZIP.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in files.items():
            z.write(path, name)
        z.writestr("seedrep_inputs_manifest.json", json.dumps(manifest, indent=1))
    print(json.dumps(info, indent=1))
    print(f"wrote {OUT}/ and {ZIP} ({ZIP.stat().st_size / 1e6:.1f} MB, {len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
