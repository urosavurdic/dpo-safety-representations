"""The same measurements, on the six original matched-start checkpoints.

    python -m src.seedrep.originals [--pairs 200] [--parity 16]

For each branch: the start (M1 / M1_alt), the safety-SFT arm and the direct
DPO arm. Per checkpoint it records:

* first-response-token KL from the branch's start, per quadrant;
* held-out PKU diagnostics (``diagnostics.summarise``) with the start as
  reference;
* a parity check: ``forward_batch`` on the first ``--parity`` benchmark rows
  against the stored ``results/activations/{stage}_final.npy``. This tests the
  new extraction code against the arrays every stored result rests on.

Results are written after every checkpoint, and a rerun skips what is done,
so the job survives an interruption on a laptop as well as on a GPU.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from src.seedrep import STATUS_EXPLORATORY, diagnostics, extract, models, rows

OUT = rows.ROOT / "results" / "stagecmp" / "originals_diag.json"
CACHE = rows.ROOT / ".seedrep_cache"
BRANCHES = {
    "alpaca": ("M1", "M2", "M3_direct"),
    "dolly": ("M1_alt", "M2_alt", "M3_direct_alt"),
}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"status": STATUS_EXPLORATORY, "stages": {}}


def _save(path: Path, data: dict) -> None:
    data["updated_utc"] = _now()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8", newline="\n")
    tmp.replace(path)


def parity(model, tokenizer, bench, device, stage: str, n: int) -> dict:
    stored_path = rows.ROOT / "results" / "activations" / f"{stage}_final.npy"
    if n <= 0 or not stored_path.exists():
        return {"checked": False}
    stored = np.load(stored_path, mmap_mode="r")
    final, _, _ = extract.forward_batch(model, tokenizer, [r["prompt"] for r in bench[:n]],
                                        device, layers=None, want_logprobs=False)
    ref = np.asarray(stored[:n], dtype=np.float32)
    cos = [float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))
           for a, b in zip(final[:, 24], ref[:, 24])]
    rel = float(np.abs(final - ref).max() / np.abs(ref).max())
    # stored arrays came from fp16 on a T4 in batches of 8; this run is fp32 in
    # one batch, so agreement is to fp16 tolerance, not bit-exact
    return {"checked": True, "n": n, "min_cosine_L24": min(cos), "max_rel_abs_diff": rel}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--parity", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--bench-limit", type=int, default=0,
                    help="dry runs only: use the first N benchmark rows")
    ap.add_argument("--stages", nargs="*", default=None, help="dry runs only")
    ap.add_argument("--no-heldout", action="store_true",
                    help="first-token KL and parity only (held-out pairs run on the GPU)")
    args = ap.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    CACHE.mkdir(exist_ok=True)
    bench = rows.benchmark_rows()
    if args.bench_limit:
        bench = bench[:args.bench_limit]
    quadrant = np.array([r["quadrant"] for r in bench])
    pairs = rows.heldout_pairs(n=args.pairs)
    device = models.pick_device()
    tokenizer = models.load_tokenizer()
    data = _load(args.out)
    data["config"] = {"pairs": args.pairs, "parity_rows": args.parity, "device": device,
                      "heldout_file": str(rows.HELDOUT_PAIRS.relative_to(rows.ROOT))}

    for branch, stages in BRANCHES.items():
        start = stages[0]
        ref_lp_path = CACHE / f"{start}_first_token_logprobs_n{len(bench)}.npy"
        ref_pairs = data["stages"].get(start, {}).get("pair_scores")
        for stage in stages:
            if args.stages is not None and stage not in args.stages:
                continue
            if data["stages"].get(stage, {}).get("done"):
                print(f"[{stage}] done, skipping")
                continue
            if stage == start and args.no_heldout and ref_lp_path.exists():
                data["stages"][stage] = {"branch": branch, "start": start, "done": True,
                                         "note": "first-token log-probs reused from cache"}
                _save(args.out, data)
                print(f"[{stage}] reference log-probs cached, skipping", flush=True)
                continue
            print(f"[{stage}] loading", flush=True)
            model = models.load_merged(models.chain_for(stage=stage), device=device)
            entry = {"branch": branch, "start": start}
            entry["parity"] = parity(model, tokenizer, bench, device, stage, args.parity)
            print(f"[{stage}] parity {entry['parity']}", flush=True)

            ref_lp = np.load(ref_lp_path) if (stage != start and ref_lp_path.exists()) else None
            res = extract.extract(model, tokenizer, bench, device, layers=[],
                                  ref_logprobs=ref_lp, keep_logprobs=(stage == start))
            if stage == start:
                np.save(ref_lp_path, res["logprobs"])
            else:
                kl = res["kl"]
                entry["first_token_kl"] = {
                    "all": float(kl.mean()),
                    **{q: float(kl[quadrant == q].mean()) for q in "ABCD"},
                }
            entry["first_token_entropy"] = float(res["entropy"].mean())
            print(f"[{stage}] first-token KL {entry.get('first_token_kl')}", flush=True)

            if not args.no_heldout:
                scores = diagnostics.pair_scores(model, tokenizer, pairs, device)
                entry["pair_scores"] = scores
                if stage == start:
                    ref_pairs = scores
                entry["heldout"] = diagnostics.summarise(scores, None if stage == start else ref_pairs)
                print(f"[{stage}] held-out {entry['heldout']}", flush=True)
            entry["done"] = True
            data["stages"][stage] = entry
            _save(args.out, data)
            models.free(model)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
