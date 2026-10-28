"""Controls for what the harmful-benign contrast encodes.

    python -m src.stagecmp.contrast_controls --xstest-csv PATH [--skip-xstest]

``PATH`` is ``xstest_prompts.csv`` from github.com/paul-rottger/xstest (CC BY
4.0), pinned below by SHA-256; it is not redistributed here.

The main contrast is overt harmful (A, HarmBench) minus plain benign (D,
Alpaca / Dolly / OpenAssistant) prompts. Two things could drive it besides
harm: the benign prompts share source pools with the instruction corpora the
start models were trained on, and the two quadrants differ in source and style.
This module recomputes the along/orthogonal split of each arm's contrast under
four controls:

* **overlap** -- exact (normalised text) and near-duplicate (character
  3-5-gram TF-IDF cosine >= 0.9) matches of every benchmark prompt against
  every tracked training file;
* **source-disjoint benign** -- A minus the OpenAssistant part of D, a source
  neither start model was trained on;
* **style-crossed** -- reworded harmful (C) minus alarming benign (B), whose
  surface cues point the opposite way to their labels;
* **source-matched** -- XSTest's 200 unsafe prompts minus its 250 safe ones
  (quadrant B): minimal pairs from one source in one format. These need new
  forward passes, and both halves are extracted here in one environment
  rather than mixing new activations with the stored ones.

Per control: R, cosine, along and orthogonal parts per arm and layer (final
prompt token), and a prompt bootstrap of the recipe difference (DPO minus SFT,
mean over branches) with nominal and Bonferroni-adjusted intervals. Writes
``results/stagecmp/contrast_controls.json``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from src.seedrep import STATUS_EXPLORATORY, rows
from src.stagecmp.run_geometry import _decompose

OUT = rows.ROOT / "results" / "stagecmp" / "contrast_controls.json"
ACT = rows.ROOT / "results" / "activations"
CACHE = rows.ROOT / ".seedrep_cache" / "contrast_controls"
TRAIN_FILES = ("sft_helpful", "sft_helpful_alt", "sft_safety")
BRANCHES = {"alpaca": ("M1", "M2", "M3_direct"), "dolly": ("M1_alt", "M2_alt", "M3_direct_alt")}
LAYERS = (16, 20, 24, 28)
NEAR_DUP = 0.9
XSTEST_SHA256 = "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


# ------------------------------------------------------------- overlap --

def overlap_scan(bench: list[dict]) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer

    prompts = [r["prompt"] for r in bench]
    out = {"method": f"exact after normalisation; near = char 3-5-gram TF-IDF cosine >= {NEAR_DUP}",
           "files": {}}
    for name in TRAIN_FILES:
        train = [r["prompt"] for r in rows.read_jsonl(rows.ROOT / "data" / "processed" / f"{name}.jsonl")]
        seen = {_norm(t) for t in train}
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(train + prompts)
        best = (vec.transform(prompts) @ vec.transform(train).T).max(axis=1).toarray().ravel()
        hits = lambda idx: [{"record_id": bench[i]["record_id"], "quadrant": bench[i]["quadrant"],
                             "similarity": round(float(best[i]), 3)} for i in idx]
        out["files"][name] = {
            "n_train_prompts": len(train),
            "exact": hits([i for i, p in enumerate(prompts) if _norm(p) in seen]),
            "near": hits(np.where(best >= NEAR_DUP)[0]),
            "max_similarity": float(best.max()),
        }
    return out


# ------------------------------------------------------------ geometry --

def control_geometry(acts: dict, harmful: np.ndarray, benign: np.ndarray, *, b: int,
                     seed: int) -> dict:
    """Point estimates per arm and layer, plus a paired bootstrap of the recipe
    difference. ``acts[stage]`` is ``(n_rows, len(LAYERS), hidden)``."""
    rng = np.random.default_rng(seed)
    nh, nb = len(harmful), len(benign)
    wh = rng.multinomial(nh, np.full(nh, 1 / nh), size=b) / nh
    wb = rng.multinomial(nb, np.full(nb, 1 / nb), size=b) / nb
    out = {"n_harmful": nh, "n_benign": nb, "layers": {}}
    for slot, layer in enumerate(LAYERS):
        cell = {"arms": {}}
        boot = {}
        for branch, (start, sft, dpo) in BRANCHES.items():
            x = {s: np.asarray(acts[s][:, slot], np.float64) for s in (start, sft, dpo)}
            point = {s: x[s][harmful].mean(0) - x[s][benign].mean(0) for s in x}
            rep = {s: wh @ x[s][harmful] - wb @ x[s][benign] for s in x}
            cell["start_norm_" + branch] = float(np.linalg.norm(point[start]))
            for recipe, stage in (("sft", sft), ("dpo", dpo)):
                cell["arms"][f"{recipe}_{branch}"] = {k: float(v) for k, v in
                                                      _decompose(point[start], point[stage]).items()}
                boot[recipe, branch] = _decompose(rep[start], rep[stage])
        for key in ("orth", "cosine", "R"):
            diff = np.mean([boot["dpo", br][key] - boot["sft", br][key] for br in BRANCHES], axis=0)
            point = np.mean([cell["arms"][f"dpo_{br}"][key] - cell["arms"][f"sft_{br}"][key]
                             for br in BRANCHES])
            alpha = 0.05 / len(LAYERS)
            cell[f"recipe_{key}"] = {
                "point": float(point),
                "ci_low": float(np.quantile(diff, 0.025)), "ci_high": float(np.quantile(diff, 0.975)),
                "adj_low": float(np.quantile(diff, alpha / 2)),
                "adj_high": float(np.quantile(diff, 1 - alpha / 2))}
        out["layers"][str(layer)] = cell
    return out


def stored_acts() -> dict:
    stages = [s for st in BRANCHES.values() for s in st]
    return {s: np.load(ACT / f"{s}_final.npy", mmap_mode="r")[:, list(LAYERS)] for s in stages}


# -------------------------------------------------------------- XSTest --

def xstest_rows(path: Path, bench: list[dict]) -> tuple[list[dict], list[dict], str]:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != XSTEST_SHA256:
        raise SystemExit(f"unexpected XSTest file (sha256 {sha})")
    table = list(csv.DictReader(path.open(encoding="utf-8")))
    safe = [r for r in table if r["label"] == "safe"]
    unsafe = [r for r in table if r["label"] == "unsafe"]
    b_prompts = sorted(r["prompt"].strip() for r in bench if r["quadrant"] == "B")
    if sorted(r["prompt"].strip() for r in safe) != b_prompts:
        raise SystemExit("XSTest's safe half is not quadrant B; refusing to pair them")
    as_rows = lambda part: [{"record_id": f"xstest_{r['id']}", "prompt": r["prompt"]} for r in part]
    return as_rows(unsafe), as_rows(safe), sha


def xstest_acts(unsafe: list[dict], safe: list[dict], bench: list[dict]) -> tuple[dict, dict]:
    """Final-token activations of both XSTest halves for every checkpoint,
    cached per stage, with a parity check of the safe half against the stored
    quadrant-B activations."""
    from src.seedrep import extract, models

    CACHE.mkdir(parents=True, exist_ok=True)
    device = models.pick_device()
    tokenizer = models.load_tokenizer()
    b_index = {r["prompt"].strip(): i for i, r in enumerate(bench) if r["quadrant"] == "B"}
    acts, parity = {}, {}
    for stage in [s for st in BRANCHES.values() for s in st]:
        path = CACHE / f"xstest_{stage}.npz"
        if not path.exists():
            print(f"[{stage}] extracting {len(unsafe) + len(safe)} prompts", flush=True)
            model = models.load_merged(models.chain_for(stage=stage), device=device)
            res = extract.extract(model, tokenizer, unsafe + safe, device, layers=list(LAYERS))
            models.free(model)
            np.savez(path, final=res["final"])
        final = np.load(path)["final"]
        acts[stage] = final
        stored = np.load(ACT / f"{stage}_final.npy", mmap_mode="r")
        ref = np.asarray(stored[[b_index[r["prompt"].strip()] for r in safe]][:, list(LAYERS)], np.float64)
        new = final[len(unsafe):].astype(np.float64)
        cos = (ref * new).sum(-1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(new, axis=-1))
        parity[stage] = {"min_cosine": float(cos.min()), "mean_cosine": float(cos.mean())}
        print(f"[{stage}] parity vs stored B: {parity[stage]}", flush=True)
    return acts, parity


# ---------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xstest-csv", type=Path, default=None)
    ap.add_argument("--skip-xstest", action="store_true")
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    bench = rows.benchmark_rows()
    quad = np.array([r["quadrant"] for r in bench])
    source = np.array([r["source"] for r in bench])
    split = np.array([r.get("split") or "" for r in bench])
    a_est = np.where((quad == "A") & (split == "direction_estimation"))[0]

    out = {"status": STATUS_EXPLORATORY, "produced_utc": _now(), "layers": list(LAYERS),
           "bootstrap": {"b": args.b, "seed": args.seed}}
    out["overlap"] = overlap_scan(bench)
    print("overlap:", {k: (len(v["exact"]), len(v["near"])) for k, v in out["overlap"]["files"].items()})

    acts = stored_acts()
    controls = {
        "main": (a_est, np.where((quad == "D") & (split == "direction_estimation"))[0]),
        "source_disjoint_benign": (a_est, np.where((quad == "D") & (source == "OASST1"))[0]),
        "style_crossed": (np.where(quad == "C")[0], np.where(quad == "B")[0]),
    }
    out["controls"] = {}
    for name, (harmful, benign) in controls.items():
        out["controls"][name] = control_geometry(acts, harmful, benign, b=args.b, seed=args.seed)
        print(f"{name}: done", flush=True)

    if not args.skip_xstest:
        if args.xstest_csv is None:
            raise SystemExit("--xstest-csv is required unless --skip-xstest")
        import torch

        if args.threads:
            torch.set_num_threads(args.threads)
        unsafe, safe, sha = xstest_rows(args.xstest_csv, bench)
        x_acts, parity = xstest_acts(unsafe, safe, bench)
        n = len(unsafe)
        cell = control_geometry(x_acts, np.arange(n), np.arange(n, n + len(safe)),
                                b=args.b, seed=args.seed)
        cell.update({"xstest_sha256": sha, "parity_vs_stored_B": parity,
                     "note": "both halves extracted fresh in this environment"})
        out["controls"]["source_matched_xstest"] = cell

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
