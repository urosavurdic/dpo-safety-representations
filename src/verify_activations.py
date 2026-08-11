"""
Sanity check for extracted activations - correct shapes, no NaN/all-zero
arrays, metadata aligned with the controlled eval set, and a check that
different prompts actually produced different vectors (catches a bug
where the same vector gets silently copied for every row).
"""
import json
from pathlib import Path

import numpy as np

STAGES = ["M0", "M1", "M2", "M3"]
EXPECTED_N_PROMPTS = 370
EXPECTED_QUADRANT_COUNTS = {"A": 50, "B": 250, "C": 20, "D": 50}


def main():
    act_dir = Path("results/activations")
    for stage in STAGES:
        final = np.load(act_dir / f"{stage}_final.npy")
        pooled = np.load(act_dir / f"{stage}_pooled.npy")
        with open(act_dir / f"{stage}_metadata.json", encoding="utf-8") as f:
            meta = json.load(f)

        print(f"\n=== {stage} ===")
        print(f"final shape: {final.shape}, pooled shape: {pooled.shape}, metadata entries: {len(meta)}")

        assert final.shape[0] == EXPECTED_N_PROMPTS, f"Expected {EXPECTED_N_PROMPTS} prompts, got {final.shape[0]}"
        assert pooled.shape == final.shape, "final/pooled shape mismatch"
        assert len(meta) == EXPECTED_N_PROMPTS, "metadata count mismatch"
        assert not np.isnan(final).any(), "NaN in final-token activations!"
        assert not np.isnan(pooled).any(), "NaN in pooled activations!"
        assert not np.all(final == 0), "final-token activations are all zero!"

        unique_rows = len(np.unique(final[:, -1, :].round(4), axis=0))
        print(f"Unique final-layer vectors (of {final.shape[0]}): {unique_rows}")
        assert unique_rows > EXPECTED_N_PROMPTS * 0.9, "Suspiciously few unique vectors - possible extraction bug"

        quadrant_counts = {}
        for row in meta:
            quadrant_counts[row["quadrant"]] = quadrant_counts.get(row["quadrant"], 0) + 1
        print(f"Quadrant counts: {quadrant_counts}")
        assert quadrant_counts == EXPECTED_QUADRANT_COUNTS, f"Quadrant mismatch: {quadrant_counts}"

        print(f"{stage}: OK")

    print("\nAll stages verified.")


if __name__ == "__main__":
    main()