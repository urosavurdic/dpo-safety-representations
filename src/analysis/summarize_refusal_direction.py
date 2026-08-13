"""
Component 4b: compact summary of eval_refusal_direction.py's output.

Reads the already-saved JSON files, prints a small table at a handful of
representative layers instead of dumping all 29 -- meant to be pasted
back directly rather than inspected file-by-file.
"""
import json
from pathlib import Path

OUT_DIR = Path("results/refusal_direction")
SAMPLE_LAYERS = [0, 7, 14, 21, 28]
STAGES = ["M0", "M1", "M2", "M3"]


def main():
    with open(OUT_DIR / "cosine_similarity.json", encoding="utf-8") as f:
        cosine = json.load(f)
    with open(OUT_DIR / "quadrant_projections.json", encoding="utf-8") as f:
        proj = json.load(f)

    print("=== Cosine similarity vs M0 (direction stability) ===")
    print(f"{'layer':>6} " + " ".join(f"{s:>8}" for s in STAGES))
    for layer in SAMPLE_LAYERS:
        row = [cosine["vs_M0"][s][layer] for s in STAGES]
        print(f"{layer:>6} " + " ".join(f"{v:8.4f}" for v in row))

    print("\n=== Cosine similarity, adjacent stages ===")
    adj_keys = list(cosine["adjacent"].keys())
    print(f"{'layer':>6} " + " ".join(f"{k:>12}" for k in adj_keys))
    for layer in SAMPLE_LAYERS:
        row = [cosine["adjacent"][k][layer] for k in adj_keys]
        print(f"{layer:>6} " + " ".join(f"{v:12.4f}" for v in row))

    print("\n=== Mean projection by quadrant, per stage ===")
    for stage in STAGES:
        print(f"\n-- {stage} --")
        quadrants = sorted(proj[stage].keys())
        print(f"{'layer':>6} " + " ".join(f"{q:>8}" for q in quadrants))
        for layer in SAMPLE_LAYERS:
            row = [proj[stage][q][layer] for q in quadrants]
            print(f"{layer:>6} " + " ".join(f"{v:8.3f}" for v in row))


if __name__ == "__main__":
    main()