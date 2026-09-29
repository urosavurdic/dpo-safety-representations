"""
Sublayer attribution: split each block's residual-stream write into its
attention and MLP halves, and ask how each half relates to the refusal
direction.

results/activations/{stage}_{final,pooled}.npy holds block OUTPUTS only -
h^(l), in which the attention and MLP writes are already summed and no longer
separable. This script re-runs the forward pass with hooks on each block's
self_attn and mlp modules to capture the two writes individually:

    h^(l) - h^(l-1)  ==  attn_write^(l) + mlp_write^(l)

That identity is asserted per batch as a correctness check on the hooks.

Two analyses, answering different questions:

  projection   attn_write . d^(l) and mlp_write . d^(l), where d^(l) is the
               difference-in-means refusal direction. Asks whether each
               component pushes ALONG the direction already identified.
  probe        logistic regression fit on attn_write alone, and on mlp_write
               alone. Asks whether harmful-vs-benign is decodable from a
               component's write at all, in any direction.

d^(l) is estimated on quadrant A vs D within the direction_estimation split
only, matching the rest of the pipeline. Projections are therefore in-sample
for those prompts, so every summary is broken out by split.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from src.training.model import try_load_stage_model
from src.training.eval_generation import build_generation_prompt

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
BATCH_SIZE = 8
POOL_WINDOW = 5
POS_QUADRANT = "A"   # overtly harmful
NEG_QUADRANT = "D"   # plainly benign
DIRECTION_SPLIT = "direction_estimation"


def compute_pool_window(attn_len, pool_window=POOL_WINDOW):
    return min(pool_window, attn_len)


def load_controlled_eval(path="data/processed/controlled_eval.jsonl"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def find_decoder_layers(model):
    """The ModuleList of decoder blocks, found structurally so that a
    PEFT/LoRA wrapper around the base model does not break the path."""
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) >= 8:
            first = module[0]
            if hasattr(first, "self_attn") and hasattr(first, "mlp"):
                return module
    raise RuntimeError("could not locate decoder layers (no ModuleList with self_attn/mlp)")


def refusal_directions(block_states, rows, pooling_name):
    """Per-layer difference-in-means direction from the ALREADY SAVED block
    outputs, estimated on POS vs NEG within DIRECTION_SPLIT. Returns
    (L, d) unit vectors, one per block-output index."""
    pos = [i for i, r in enumerate(rows)
           if r["quadrant"] == POS_QUADRANT and r.get("split") == DIRECTION_SPLIT]
    neg = [i for i, r in enumerate(rows)
           if r["quadrant"] == NEG_QUADRANT and r.get("split") == DIRECTION_SPLIT]
    if not pos or not neg:
        raise RuntimeError(f"empty direction-estimation group: {len(pos)} pos, {len(neg)} neg")
    contrast = block_states[pos].mean(axis=0) - block_states[neg].mean(axis=0)
    norms = np.linalg.norm(contrast, axis=-1, keepdims=True)
    print(f"  {pooling_name}: direction from {len(pos)} {POS_QUADRANT} vs "
          f"{len(neg)} {NEG_QUADRANT} prompts; contrast norm range "
          f"{norms.min():.2f}-{norms.max():.2f}")
    return contrast / np.clip(norms, 1e-12, None)


class SublayerCapture:
    """Collects each block's attention write and MLP write for one forward
    pass. The self_attn module's output is already post-o_proj and the mlp
    module's output is already post-down_proj, so both are exactly the
    tensors added to the residual stream."""

    def __init__(self, layers):
        self.layers = layers
        self.attn = {}
        self.mlp = {}
        self.handles = []

    def __enter__(self):
        for idx, layer in enumerate(self.layers):
            self.handles.append(
                layer.self_attn.register_forward_hook(self._make(idx, self.attn)))
            self.handles.append(
                layer.mlp.register_forward_hook(self._make(idx, self.mlp)))
        return self

    def _make(self, idx, store):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            store[idx] = tensor.detach()
        return hook

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def clear(self):
        self.attn.clear()
        self.mlp.clear()


def extract_batch(model, tokenizer, capture, prompts, device, n_layers, hidden_dim, atol):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        texts = [build_generation_prompt(tokenizer, p) for p in prompts]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        capture.clear()
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        n = len(prompts)
        keys = ("attn_final", "attn_pooled", "mlp_final", "mlp_pooled")
        out = {k: np.zeros((n, n_layers, hidden_dim), dtype=np.float32) for k in keys}
        worst = 0.0

        for layer_idx in range(n_layers):
            attn_w = capture.attn[layer_idx].float()
            mlp_w = capture.mlp[layer_idx].float()
            # the hooks must reproduce the block's total write, exactly
            delta = (outputs.hidden_states[layer_idx + 1]
                     - outputs.hidden_states[layer_idx]).float()
            worst = max(worst, float((delta - (attn_w + mlp_w)).abs().max()))
            for i in range(n):
                attn_len = int(inputs["attention_mask"][i].sum().item())
                window = compute_pool_window(attn_len)
                out["attn_final"][i, layer_idx] = attn_w[i, -1].cpu().numpy()
                out["mlp_final"][i, layer_idx] = mlp_w[i, -1].cpu().numpy()
                out["attn_pooled"][i, layer_idx] = attn_w[i, -window:].mean(dim=0).cpu().numpy()
                out["mlp_pooled"][i, layer_idx] = mlp_w[i, -window:].mean(dim=0).cpu().numpy()

        if worst > atol:
            raise RuntimeError(
                f"hook identity failed: max |(h_l - h_l-1) - (attn + mlp)| = {worst:.3e} "
                f"> {atol:.3e}. The captured tensors are not the block's two writes; "
                f"attribution downstream would be wrong.")
        return out, worst
    finally:
        tokenizer.padding_side = original_padding_side


def summarise(writes, directions, rows, pooling):
    """Per-layer projections onto d, write norms, and the angle between the
    two writes; plus per-quadrant/split means and quadrant-agreement rates."""
    attn = writes[f"attn_{pooling}"]                     # (N, L, D)
    mlp = writes[f"mlp_{pooling}"]
    unit_d = directions[1:]                              # drop the embedding index

    proj_attn = np.einsum("nld,ld->nl", attn, unit_d)
    proj_mlp = np.einsum("nld,ld->nl", mlp, unit_d)
    norm_attn = np.linalg.norm(attn, axis=-1)
    norm_mlp = np.linalg.norm(mlp, axis=-1)
    denom = np.clip(norm_attn * norm_mlp, 1e-12, None)
    cos_attn_mlp = np.einsum("nld,nld->nl", attn, mlp) / denom

    groups = {}
    for quad in sorted({r["quadrant"] for r in rows}):
        for split in sorted({str(r.get("split")) for r in rows}):
            idx = [i for i, r in enumerate(rows)
                   if r["quadrant"] == quad and str(r.get("split")) == split]
            if not idx:
                continue
            a, m = proj_attn[idx], proj_mlp[idx]
            in_sample = (split == DIRECTION_SPLIT
                         and quad in (POS_QUADRANT, NEG_QUADRANT))
            groups[f"{quad}/{split}"] = {
                "n": len(idx),
                "in_sample_for_direction": in_sample,
                "proj_attn_mean": a.mean(axis=0).round(4).tolist(),
                "proj_mlp_mean": m.mean(axis=0).round(4).tolist(),
                "norm_attn_mean": norm_attn[idx].mean(axis=0).round(4).tolist(),
                "norm_mlp_mean": norm_mlp[idx].mean(axis=0).round(4).tolist(),
                "cos_attn_mlp_mean": cos_attn_mlp[idx].mean(axis=0).round(4).tolist(),
                # which quadrant of the (attn.d, mlp.d) plane, per layer
                "frac_both_positive": ((a > 0) & (m > 0)).mean(axis=0).round(4).tolist(),
                "frac_both_negative": ((a < 0) & (m < 0)).mean(axis=0).round(4).tolist(),
                "frac_disagree": ((a > 0) != (m > 0)).mean(axis=0).round(4).tolist(),
            }
    per_prompt = {"proj_attn": proj_attn, "proj_mlp": proj_mlp,
                  "norm_attn": norm_attn, "norm_mlp": norm_mlp,
                  "cos_attn_mlp": cos_attn_mlp}
    return groups, per_prompt


def probe_components(writes, rows, pooling, layer_indices, seed=0):
    """CV AUC of a linear probe fit on each component's write ALONE.
    POS vs NEG across all splits, 5-fold stratified - a decodability
    question, independent of the difference-in-means direction."""
    keep = [i for i, r in enumerate(rows)
            if r["quadrant"] in (POS_QUADRANT, NEG_QUADRANT)]
    y = np.array([1 if rows[i]["quadrant"] == POS_QUADRANT else 0 for i in keep])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = {}
    for comp in ("attn", "mlp"):
        features = writes[f"{comp}_{pooling}"][keep]
        aucs = {}
        for layer_idx in layer_indices:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=1.0))
            scores = cross_val_score(clf, features[:, layer_idx], y,
                                     cv=cv, scoring="roc_auc")
            aucs[int(layer_idx) + 1] = {"auc_mean": round(float(scores.mean()), 4),
                                        "auc_std": round(float(scores.std()), 4)}
        out[comp] = aucs
    out["_n"] = {"pos": int(y.sum()), "neg": int((1 - y).sum())}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=["M0", "M3"],
                        help="stages to run (default: the two endpoints)")
    parser.add_argument("--pooling", nargs="+", default=["final", "pooled"],
                        choices=["final", "pooled"])
    parser.add_argument("--probe-layers", nargs="+", type=int, default=None,
                        help="block indices (1-28) to probe; default 12,16,20,24,28")
    parser.add_argument("--limit", type=int, default=None, help="limit prompts (dry run)")
    parser.add_argument("--atol", type=float, default=2e-2,
                        help="tolerance for the hook identity check (bf16 weights are coarse)")
    parser.add_argument("--save-vectors", action="store_true",
                        help="also save the raw write vectors (float16, ~100MB per array)")
    parser.add_argument("--out-dir", default="results/sublayer_attribution")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_rows = load_controlled_eval()
    if args.limit:
        eval_rows = eval_rows[:args.limit]
    print(f"Loaded {len(eval_rows)} controlled-eval prompts.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir = Path("results/activations")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for stage_name in args.stages:
        print(f"\n=== {stage_name} ===")
        model = try_load_stage_model(stage_name)
        if model is None:
            print(f"  {stage_name}: no checkpoint, skipping")
            continue

        layers = find_decoder_layers(model)
        n_layers = len(layers)
        hidden_dim = model.config.hidden_size
        probe_layers = args.probe_layers or [l for l in (12, 16, 20, 24, 28) if l <= n_layers]
        print(f"  {n_layers} blocks, hidden {hidden_dim}; probing blocks {probe_layers}")

        chunks, worst = [], 0.0
        with SublayerCapture(layers) as capture:
            for i in range(0, len(eval_rows), BATCH_SIZE):
                batch_rows = eval_rows[i:i + BATCH_SIZE]
                got, residual = extract_batch(
                    model, tokenizer, capture, [r["prompt"] for r in batch_rows],
                    device, n_layers, hidden_dim, args.atol)
                chunks.append(got)
                worst = max(worst, residual)
                print(f"    {min(i + BATCH_SIZE, len(eval_rows))}/{len(eval_rows)} done")

        writes = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
        del chunks
        gc.collect()
        print(f"  hook identity check passed, max residual {worst:.3e} "
              f"(tol {args.atol:.0e})")

        summary = {"stage": stage_name, "n_prompts": len(eval_rows),
                   "n_blocks": n_layers, "hidden_dim": hidden_dim,
                   "hook_identity_max_residual": worst,
                   "pos_quadrant": POS_QUADRANT, "neg_quadrant": NEG_QUADRANT,
                   "direction_split": DIRECTION_SPLIT,
                   "note": "layer axis of every list below is block index 1..N "
                           "(the embedding index is dropped)",
                   "pooling": {}}

        for pooling in args.pooling:
            block_path = act_dir / f"{stage_name}_{pooling}.npy"
            if not block_path.exists():
                print(f"  {pooling}: {block_path} missing, cannot estimate "
                      f"direction - skipping")
                continue
            block_states = np.load(block_path)
            if block_states.shape[0] != len(eval_rows):
                print(f"  {pooling}: {block_path} has {block_states.shape[0]} rows, "
                      f"eval set has {len(eval_rows)} - skipping (re-extract first)")
                continue
            print(f"  {pooling}:")
            directions = refusal_directions(block_states, eval_rows, pooling)
            groups, per_prompt = summarise(writes, directions, eval_rows, pooling)
            probes = probe_components(writes, eval_rows, pooling,
                                      [l - 1 for l in probe_layers])
            summary["pooling"][pooling] = {"groups": groups, "probes": probes}

            payload = dict(per_prompt)
            payload["directions"] = directions
            if args.save_vectors:
                for key, value in writes.items():
                    if key.endswith(pooling):
                        payload[key] = value.astype(np.float16)
            npz_path = out_dir / f"{stage_name}_{pooling}_attribution.npz"
            np.savez_compressed(npz_path, **payload)
            print(f"    wrote {npz_path}")
            for layer in probe_layers:
                attn_auc = probes["attn"][layer]["auc_mean"]
                mlp_auc = probes["mlp"][layer]["auc_mean"]
                print(f"    block {layer:>2}: probe AUC  attn {attn_auc:.3f}   "
                      f"mlp {mlp_auc:.3f}")

        summary_path = out_dir / f"{stage_name}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote {summary_path}")

        del model, writes
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
