"""
LoRA-subspace check: how much of M3's refusal direction lies within what
the M2->M3 LoRA update (r=64) could structurally express.

This doesn't test whether LoRA CAUSED the observed direction -- it tests
whether the direction is even COMPATIBLE with what a rank-64 update could
write into the residual stream. If a large fraction of the direction's
norm lies outside that subspace, the "this is just a LoRA artifact"
explanation is directly weakened, not just theoretically flagged as a
limitation.

Only o_proj (attention output) and down_proj (MLP output) are checked --
the two module types that write directly into the residual stream.
q/k/v_proj and gate/up_proj act on internal representations, not the
residual stream, so their column spaces aren't directly comparable to a
residual-stream direction.

Runs fully locally, CPU-only, no Colab -- only downloads the adapter's own
(small, a few MB) weight file, never the full base model.

ASSUMPTION -- not independently verified: that the adapter repo below
holds ONLY the M2->M3 step's own LoRA delta, not a merged multi-stage
chain. Fails with a clear KeyError-based message if the naming doesn't
match what this assumes, rather than silently computing on wrong data.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

M2_M3_ADAPTER_REPO = "urosavurdic/qwen2.5-1.5b-m3-dpo"
CHECK_LAYERS = [7, 14, 21, 28]  # hidden_states indices; decoder block = index - 1
TARGET_MODULES = ["o_proj", "down_proj"]


def download_adapter_weights(repo_id=M2_M3_ADAPTER_REPO):
    path = hf_hub_download(repo_id=repo_id, filename="adapter_model.safetensors")
    return load_file(path)


def find_lora_B_key(state_dict, decoder_idx, module_name):
    candidates = [
        k for k in state_dict
        if f"layers.{decoder_idx}." in k and module_name in k and "lora_B" in k
    ]
    if len(candidates) != 1:
        raise KeyError(
            f"Expected exactly one lora_B key for layer {decoder_idx}, module {module_name}; "
            f"found {len(candidates)}: {candidates}. Adapter's key naming doesn't match what "
            f"this script assumed -- run list(state_dict.keys())[:20] and paste it back."
        )
    return candidates[0]


def subspace_capture_fraction(direction, lora_B_matrix):
    """Fraction of `direction`'s squared norm lying in the column space of
    lora_B_matrix (shape [out_dim, rank]). direction: unit vector, (out_dim,)."""
    Q, _ = torch.linalg.qr(lora_B_matrix)  # orthonormal basis for the column space
    proj = Q.T @ direction
    return torch.sum(proj ** 2).item()  # direction is unit norm -> this IS the fraction

def random_direction_capture_fraction(lora_B_matrix, hidden_dim, n_samples=200, seed=0):
    rng = np.random.default_rng(seed)
    fractions = []
    for _ in range(n_samples):
        v = rng.normal(size=hidden_dim)
        v = torch.from_numpy(v / np.linalg.norm(v)).float()
        fractions.append(subspace_capture_fraction(v, lora_B_matrix))
    return float(np.mean(fractions)), float(np.std(fractions))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-repo", default=M2_M3_ADAPTER_REPO)
    args = parser.parse_args()

    print(f"Downloading adapter weights from {args.adapter_repo}...")
    state_dict = download_adapter_weights(args.adapter_repo)
    print(f"Loaded {len(state_dict)} tensors.")

    directions = np.load("results/refusal_direction/M3_direction.npy")  # (29, hidden_dim)

    results = {}
    for hs_index in CHECK_LAYERS:
        decoder_idx = hs_index - 1
        direction = torch.from_numpy(directions[hs_index]).float()
        direction = direction / direction.norm()

        per_module = {}
        for module_name in TARGET_MODULES:
            key = find_lora_B_key(state_dict, decoder_idx, module_name)
            lora_B = state_dict[key].float()
            frac = subspace_capture_fraction(direction, lora_B)
            rand_mean, rand_std = random_direction_capture_fraction(lora_B, lora_B.shape[0])
            print(f"  (random-direction baseline: {rand_mean:.3f} ± {rand_std:.3f})")
            per_module[module_name] = {"real_direction": frac, "random_baseline_mean": rand_mean, "random_baseline_std": rand_std}
            print(f"Layer {hs_index}, {module_name}: {frac:.3f} of direction's norm captured "
                  f"by rank-{lora_B.shape[1]} subspace")
        results[hs_index] = per_module

    out_path = Path("results/interpretability/lora_subspace_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()