"""Confirms M0-M3 activations are genuinely different, not identical."""
import numpy as np

LAYERS_TO_CHECK = [0, 14, 28]
STAGES = ["M0", "M1", "M2", "M3"]


def main():
    means = {stage: np.load(f"results/activations/{stage}_final.npy").mean(axis=0) for stage in STAGES}
    print(f"{'Pair':<12} " + " ".join(f"Layer {l:<6}" for l in LAYERS_TO_CHECK))
    for a, b in [("M0", "M1"), ("M1", "M2"), ("M2", "M3"), ("M0", "M3")]:
        sims = []
        for layer in LAYERS_TO_CHECK:
            va, vb = means[a][layer], means[b][layer]
            sims.append(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
        print(f"{a} vs {b:<8} " + " ".join(f"{s:.4f}    " for s in sims))


if __name__ == "__main__":
    main()