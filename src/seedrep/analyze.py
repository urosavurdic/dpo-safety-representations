"""Turn the banked GPU results into ``results/stagecmp/seed_replication.json``.

    python -m src.seedrep.analyze --bank hf://USER/seedrep-bank

CPU only. Everything is compared against the start checkpoint *as measured in
the same run* (``ref/{branch}_acts.npz``), and the seed-42 arms are
re-measured there too (``orig`` jobs), so no comparison crosses environments.

Contents:

* ``geometry``: per run (seed 42 = ``orig``, seeds 1, 2) and analysed layer,
  R, cosine, along and orthogonal parts, final-token and mean-pooled;
* ``seed_summary``: per arm, the range over runs, and the recipe gap against
  the run-to-run spread (descriptive: three runs give no interval);
* ``dose``: per run and checkpoint, R / along / orth at layers 20 and 24,
  relative drift, first-token KL and weight-change norm, for dose curves;
* ``behaviour``: WildGuard rates per quadrant per run, and the refusal share
  of the 500 sampled SFT targets;
* ``injection``: C-prompt outcomes per condition against no injection;
* ``summary_text``: one-sentence summaries worded from the data.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from src.seedrep import STATUS_EXPLORATORY, rows
from src.seedrep.jobs import ORIGINAL_ARMS, SEEDS, START, run_id
from src.stagecmp.run_geometry import _decompose

OUT = rows.ROOT / "results" / "stagecmp" / "seed_replication.json"
LAYERS = (16, 20, 24, 28)


def _now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _groups(bench):
    quad = np.array([r["quadrant"] for r in bench])
    split = np.array([r.get("split") or "" for r in bench])
    return (np.where((quad == "A") & (split == "direction_estimation"))[0],
            np.where((quad == "D") & (split == "direction_estimation"))[0], quad)


def _contrast(x, a, d):
    return x[a].astype(np.float64).mean(0) - x[d].astype(np.float64).mean(0)


def _cell(start, post, a, d) -> dict:
    out = _decompose(_contrast(start, a, d), _contrast(post, a, d))
    return {k: float(v) for k, v in out.items()}


def _wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"rate": p, "ci_low": c - h, "ci_high": c + h, "n": n}


def _rates(records, quadrants="ABCD"):
    out = {}
    for q in quadrants:
        wg = [r["wildguard"] for r in records if r.get("quadrant") == q]
        ref = [w["response_refusal"] for w in wg if w.get("response_refusal") is not None]
        harm = [w["response_harm"] for w in wg if w.get("response_harm") is not None]
        sr = [r["strong_reject"]["score"] for r in records if r.get("quadrant") == q
              and (r.get("strong_reject") or {}).get("score") is not None]
        out[q] = {"refusal": _wilson(sum(ref), len(ref)), "harmful": _wilson(sum(harm), len(harm)),
                  "strongreject_mean": float(np.mean(sr)) if sr else None}
    return out


def analyze(bank, work: Path, layers=LAYERS, dose_layers=(20, 24), summary_layer=24,
            bench=None) -> dict:
    bench = bench or rows.benchmark_rows()
    a, d, _ = _groups(bench)
    work.mkdir(parents=True, exist_ok=True)

    def fetch(remote):
        local = work / remote
        if local.exists() or bank.get(remote, local):
            return local
        return None

    def load_npz(remote):
        path = fetch(remote)
        return np.load(path) if path else None

    out = {"status": STATUS_EXPLORATORY, "produced_utc": _now(), "geometry": {}, "dose": {},
           "behaviour": {}, "injection": {}}
    for branch in START:
        ref = load_npz(f"ref/{branch}_acts.npz")
        if ref is None:
            continue
        runs = {f"{recipe}_orig": f"measure/orig_{stage}/final.npz"
                for recipe, stage in zip(("sft", "dpo"), ORIGINAL_ARMS[branch])}
        for recipe in ("sft", "dpo", "sft_aol"):
            for seed in SEEDS:
                rid = run_id(recipe, branch, seed)
                runs[rid] = f"measure/{rid}/final.npz"
        for name, remote in runs.items():
            m = load_npz(remote)
            if m is None:
                continue
            cells = {}
            for pooling in ("final", "pooled"):
                cells[pooling] = {str(l): _cell(ref[pooling][:, l], m[pooling][:, l], a, d)
                                  for l in layers}
            out["geometry"][f"{branch}/{name}"] = {**cells, "kl_mean": float(np.nanmean(m["kl"]))}

        for recipe in ("sft", "dpo", "sft_aol"):
            for seed in SEEDS:
                rid = run_id(recipe, branch, seed)
                steps = sorted({f.split("/")[-1][:-5] for f in bank.list(f"measure/{rid}/")
                                if f.endswith(".json")})
                curve = []
                for step in steps:
                    m = load_npz(f"measure/{rid}/{step}.npz")
                    meta = json.loads(fetch(f"measure/{rid}/{step}.json").read_text())
                    avail = [int(x) for x in m["layers"]]
                    point = {"step": meta["step"], "weight_change": meta["weight_change"]["total"],
                             "kl_mean": meta["first_token_kl_mean"]}
                    for l in dose_layers:
                        if l not in avail:
                            continue
                        xs = ref["final"][:, l].astype(np.float64)
                        xq = m["final"][:, avail.index(l)].astype(np.float64)
                        cell = _cell(xs, xq, a, d)
                        drift = np.linalg.norm(xq - xs, axis=1) / np.linalg.norm(xs, axis=1)
                        cell["relative_drift"] = float(drift.mean())
                        point[f"L{l}"] = cell
                    curve.append(point)
                if curve:
                    out["dose"][f"{branch}/{rid}"] = curve

    for name in bank.list("judge/"):
        if not name.endswith("/judged.json"):
            continue
        unit = name.split("/")[1]
        records = json.loads(fetch(name).read_text(encoding="utf-8"))["records"]
        if unit == "pku_chosen":
            ref = [r["wildguard"]["response_refusal"] for r in records
                   if r["wildguard"].get("response_refusal") is not None]
            out["behaviour"]["pku_chosen_refusal"] = _wilson(sum(ref), len(ref))
            harm = [r["wildguard"]["response_harm"] for r in records
                    if r["wildguard"].get("response_harm") is not None]
            # the dataset labels these targets safe; this is WildGuard's own reading
            out["behaviour"]["pku_chosen_harmful"] = _wilson(sum(harm), len(harm))
        elif unit.startswith("inject_"):
            out["injection"][unit] = _rates(records, "C")["C"]
        else:
            out["behaviour"][unit] = _rates(records)

    out["seed_summary"] = summarise(out["geometry"], str(summary_layer))
    out["summary_text"] = summary_text(out, summary_layer)
    return out


def summarise(geometry: dict, layer: str = "24") -> dict:
    out = {}
    for branch in START:
        arms = {}
        for recipe in ("sft", "dpo"):
            names = [f"{branch}/{recipe}_orig"] + [f"{branch}/{run_id(recipe, branch, s)}" for s in SEEDS]
            vals = {k: [geometry[n]["final"][layer][k] for n in names if n in geometry]
                    for k in ("R", "cosine", "along", "orth")}
            if vals["orth"]:
                arms[recipe] = {k: {"min": min(v), "max": max(v), "mean": float(np.mean(v)),
                                    "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                    "n_runs": len(v)} for k, v in vals.items()}
        if len(arms) == 2:
            gap = arms["dpo"]["orth"]["mean"] - arms["sft"]["orth"]["mean"]
            sds = [s for s in (arms["dpo"]["orth"]["sd"], arms["sft"]["orth"]["sd"]) if s]
            arms["orth_gap"] = gap
            arms["orth_gap_over_max_sd"] = gap / max(sds) if sds else None
            arms["separated"] = arms["dpo"]["orth"]["min"] > arms["sft"]["orth"]["max"]
        if arms:
            out[branch] = arms
    return out


WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


def summary_text(result: dict, layer: int = 24) -> dict:
    s = result["seed_summary"]
    parts, covered = [], []
    for branch, label in (("alpaca", "branch 1"), ("dolly", "branch 2")):
        arm = s.get(branch)
        if not arm or "dpo" not in arm or "sft" not in arm:
            continue
        n = min(arm["dpo"]["orth"]["n_runs"], arm["sft"]["orth"]["n_runs"])
        if n < 2:
            continue
        covered.append((label, n))
        verb = "reproduce" if arm["separated"] else "do not cleanly reproduce"
        parts.append(
            f"In {label}, {WORDS.get(n, n)} training runs per arm {verb} this: at layer {layer} "
            f"the orthogonal part spans {arm['dpo']['orth']['min']:.2f}--"
            f"{arm['dpo']['orth']['max']:.2f} across DPO runs and "
            f"{arm['sft']['orth']['min']:.2f}--{arm['sft']['orth']['max']:.2f} across SFT runs.")
    if not covered:
        return {"sentence": "", "limit": "one training run per cell"}
    limit = "; ".join(f"{WORDS.get(n, n)} training runs per arm in {label}" for label, n in covered)
    if len(covered) == 1:
        limit += ", one in the other branch"
    limit += "; run-to-run spread is reported as a range, not an interval"
    return {"sentence": " ".join(parts), "limit": limit}


def main() -> int:
    from src.seedrep import platform as plat
    from src.seedrep.bank import open_bank

    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--work", type=Path, default=rows.ROOT / ".seedrep_cache" / "analysis")
    args = ap.parse_args()
    result = analyze(open_bank(args.bank, token=plat.secret("HF_TOKEN")), args.work)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({"seed_summary": result["seed_summary"],
                      "summary_text": result["summary_text"]}, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
