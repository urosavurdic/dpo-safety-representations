"""Build the held-out PKU-SafeRLHF pairs used by ``src.seedrep.diagnostics``.

    python scripts/build_heldout_pairs.py

Same filter as ``src.data_pipeline.data_prep.build_matched_pairs`` (the two
responses disagree on safety; chosen = the safer one; no empty fields), drawn
from the PKU-SafeRLHF *test* split, with every prompt that appears in the
4,000 training pairs removed. Deterministic: a fixed seed and sorted input.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from src.data_pipeline.data_prep import build_matched_pairs
from src.seedrep.rows import DPO_DATA, HELDOUT_PAIRS, read_jsonl

N_PAIRS = 400
SEED = 20261025


def main() -> int:
    from datasets import load_dataset

    train_prompts = {r["prompt"].strip() for r in read_jsonl(DPO_DATA)}
    test = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")
    pairs, _ = build_matched_pairs(test, n_target=len(test), seed=SEED)
    kept = [p for p in pairs if p["prompt"].strip() not in train_prompts]
    # one pair per prompt, so no prompt is counted twice
    seen, unique = set(), []
    for p in sorted(kept, key=lambda p: (p["prompt"], p["chosen"])):
        if p["prompt"] not in seen:
            seen.add(p["prompt"])
            unique.append(p)
    random.Random(SEED).shuffle(unique)
    selected = unique[:N_PAIRS]
    HELDOUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with open(HELDOUT_PAIRS, "w", encoding="utf-8", newline="\n") as handle:
        for p in selected:
            handle.write(json.dumps(p, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(HELDOUT_PAIRS.read_bytes()).hexdigest()
    print(f"{len(test)} test rows -> {len(pairs)} safety-disagreeing -> "
          f"{len(kept)} not in training -> {len(unique)} unique prompts; "
          f"wrote {len(selected)} to {HELDOUT_PAIRS} (sha256 {digest[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
