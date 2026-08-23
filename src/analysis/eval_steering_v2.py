"""
Component 5b, v2: a clean, redoable, fully-configurable version of steering.

Preserves eval_steering.py exactly as-is (its results, e.g.
results/raw/steering_raw_D.json / steering_raw_D_L21.json, are NOT deleted
or touched) -- this is a separate, independently-runnable implementation
that fixes two design issues the original diagnosis (see PROJECT_CONTEXT.md
/ HANDOFF.md steering notes) identified from the EXISTING results, not new
runs:

1. The old default single-layer test (L21) sits outside the layer range
   (24-28) causal ablation found FULLY sufficient to explain quadrant A's
   refusal suppression -- testing sufficiency at a layer ablation didn't
   validate as load-bearing was never a fair test. This version's default
   single layer is 24 (the start of that validated range), configurable
   via --layers.
2. The old multi-layer run (14-28) calibrated each layer's alpha to that
   layer's OWN natural activation scale, then applied all 15 simultaneously
   -- on a residual stream, each addition persists forward AND gets added
   to again at every subsequent steered layer, so total injected magnitude
   compounds with layer count rather than staying at any single layer's
   natural scale. This is the most likely mechanism for the old 98%
   degenerate-output result. --alpha-coefficient (default 1.0, matches old
   behavior for a single layer) lets you scale down the per-layer magnitude
   specifically to counteract this when steering multiple layers at once.

Necessary vs. sufficient: causal ablation already showed the direction is
NECESSARY for refusal (removing it collapses refusal). This experiment
tests SUFFICIENCY (does adding it alone induce refusal) -- the two together
are what "sufficient and necessary" means in the mechanistic-interp sense.

Also (unlike the original, which only ever ran quadrant D) supports
quadrant A as a side-effect check in the same run: does steering perturb
already-correct harmful-prompt behavior, or is any effect selective to the
ambiguous/benign case as a genuine over-refusal story would predict?

Statistical analysis is NOT reimplemented here. This produces raw rows in
the SAME schema eval_causal_ablation.py/eval_behavioral.py already use
({prompt, quadrant, source, stage, response}) with condition names
{tag}_baseline / {tag}_steered -- point the already-generalized
summarize_causal_ablation.py --stage, mcnemar_causal_ablation.py
--conditions --quadrant --category, and bootstrap_causal_effect.py at the
output file directly, exactly as printed at the end of a run.
"""
import argparse
import gc
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.training.model import load_stage_model
from src.analysis.eval_causal_ablation import (
    get_decoder_layers, generate_batch, load_controlled_eval, filter_to_held_out_behavioral_split, BATCH_SIZE,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MAX_NEW_TOKENS = 200  # must match eval_causal_ablation.py / eval_behavioral.py
DEFAULT_LAYERS = [24]  # inside the ablation-validated 24-28 range, unlike the old default (21)
CAUSALLY_VALIDATED_RANGE = range(24, 29)  # for the "outside validated range" warning only


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def steer_direction(hidden_states, direction, alpha):
    direction = direction.to(dtype=hidden_states.dtype, device=hidden_states.device)
    return hidden_states + alpha * direction


def make_steering_hook(direction, alpha):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (steer_direction(output[0], direction, alpha),) + output[1:]
        return steer_direction(output, direction, alpha)
    return hook


def register_steering_hooks(model, directions_by_layer, alphas_by_layer):
    decoder_layers = get_decoder_layers(model)
    handles = []
    for hs_index, direction in directions_by_layer.items():
        decoder_idx = hs_index - 1
        handle = decoder_layers[decoder_idx].register_forward_hook(
            make_steering_hook(direction, alphas_by_layer[hs_index])
        )
        handles.append(handle)
    return handles


def resolve_alphas(layers, stage, alpha_source, alpha_value, alpha_coefficient,
                    quadrant_projections_path="results/refusal_direction/quadrant_projections.json"):
    """Two alpha sources:
    - "quadrant_a_projection" (default, matches original design): each
      layer's own natural mean-quadrant-A-projection scale.
    - "fixed": a single --alpha-value applied identically to every steered
      layer, regardless of that layer's natural activation scale.
    Either way, --alpha-coefficient is a final multiplier on top -- the
    "normalized alpha" knob for keeping total injected magnitude comparable
    when steering multiple layers at once (see module docstring point 2).
    """
    if alpha_source == "fixed":
        if alpha_value is None:
            raise ValueError("--alpha-value is required when --alpha-source=fixed")
        base_alphas = {L: alpha_value for L in layers}
    elif alpha_source == "quadrant_a_projection":
        with open(quadrant_projections_path, encoding="utf-8") as f:
            quadrant_projections = json.load(f)
        quadrant_a_proj = quadrant_projections[stage]["A"]
        base_alphas = {L: float(quadrant_a_proj[L]) for L in layers}
    else:
        raise ValueError(f"Unknown alpha_source: {alpha_source}")

    return {L: base_alphas[L] * alpha_coefficient for L in layers}


def build_output_path(tag, overwrite):
    out_path = Path(f"results/raw/steering_v2_{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Pass --overwrite to replace it, or --tag to pick a "
            "different name -- steering_v2 never overwrites a previous run's results by accident."
        )
    return out_path


def default_tag(stage, layers, alpha_source, alpha_coefficient, quadrants):
    layers_str = "-".join(str(l) for l in layers)
    quad_str = "".join(quadrants)
    coef_str = f"{alpha_coefficient:g}".replace(".", "p")
    return f"{stage}_L{layers_str}_{alpha_source}_coef{coef_str}_Q{quad_str}"


def load_existing_baseline(quadrants, path="results/raw/causal_ablation_raw_wide.json", baseline_stage="M3_baseline"):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [r for r in rows if r["stage"] == baseline_stage and r["quadrant"] in quadrants]


def run_condition(model, tokenizer, eval_rows, device, condition_name, out_rows):
    for i in range(0, len(eval_rows), BATCH_SIZE):
        batch_rows = eval_rows[i:i + BATCH_SIZE]
        responses = generate_batch(model, tokenizer, [r["prompt"] for r in batch_rows], device, max_new_tokens=MAX_NEW_TOKENS)
        for row, response in zip(batch_rows, responses):
            out_rows.append({
                "prompt": row["prompt"], "quadrant": row["quadrant"], "source": row["source"],
                "stage": condition_name, "response": response,
            })
        print(f"    [{condition_name}] {min(i + BATCH_SIZE, len(eval_rows))}/{len(eval_rows)} done")


def build_run_config(args, layers, alphas_by_layer, out_path):
    return {
        "component": "eval_steering_v2",
        "stage": args.stage,
        "layers": layers,
        "layers_outside_causally_validated_range": sorted(set(layers) - set(CAUSALLY_VALIDATED_RANGE)),
        "alpha_source": args.alpha_source,
        "alpha_value": args.alpha_value,
        "alpha_coefficient": args.alpha_coefficient,
        "resolved_alphas_by_layer": alphas_by_layer,
        "quadrants": args.quadrants,
        "skip_baseline": args.skip_baseline,
        "limit": args.limit,
        "direction_source": args.direction_source,
        "generation": {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "deterministic": True},
        "output_path": str(out_path),
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_run_config(cfg, out_path):
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return meta_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="M3", help="Model/direction stage to steer (default: M3)")
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                         help=f"hidden_states layer indices to steer (default: {DEFAULT_LAYERS}, "
                              "inside the ablation-validated 24-28 range -- pass multiple for "
                              "multi-layer steering, e.g. --layers 14 15 ... 28)")
    parser.add_argument("--alpha-source", default="quadrant_a_projection",
                         choices=["quadrant_a_projection", "fixed"])
    parser.add_argument("--alpha-value", type=float, default=None,
                         help="Required if --alpha-source=fixed; magnitude applied to every steered layer")
    parser.add_argument("--alpha-coefficient", type=float, default=1.0,
                         help="Final multiplier on the resolved alpha -- use <1.0 for multi-layer runs "
                              "to counteract residual-stream compounding (see module docstring)")
    parser.add_argument("--quadrants", nargs="+", default=["D"], choices=["A", "B", "C", "D"],
                         help="Which quadrants to steer+evaluate (default: D only, over-refusal test; "
                              "pass 'A D' to also run the harmful-prompt side-effect check)")
    parser.add_argument("--direction-source", default=None,
                         help="Override path to the direction .npy (default: "
                              "results/refusal_direction/{stage}_direction.npy)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-baseline", action="store_true",
                         help="Reuse existing baseline rows from causal_ablation_raw_wide.json "
                              "instead of regenerating (only valid if it already covers --quadrants)")
    parser.add_argument("--tag", default=None, help="Output filename tag (default: auto-built from config)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output file")
    args = parser.parse_args()

    layers = sorted(set(args.layers))
    tag = args.tag or default_tag(args.stage, layers, args.alpha_source, args.alpha_coefficient, args.quadrants)
    out_path = build_output_path(tag, args.overwrite)

    outside_range = sorted(set(layers) - set(CAUSALLY_VALIDATED_RANGE))
    if outside_range:
        print(f"NOTE: layer(s) {outside_range} are outside the ablation-validated 24-28 range -- "
              "a null result there is not evidence against the direction's causal role, since "
              "ablation itself only validated 24-28 as sufficient. Proceeding anyway.")

    direction_path = Path(args.direction_source or f"results/refusal_direction/{args.stage}_direction.npy")
    alphas_by_layer = resolve_alphas(layers, args.stage, args.alpha_source, args.alpha_value, args.alpha_coefficient)
    print(f"Steering layers {layers}, resolved alphas: {[round(alphas_by_layer[L], 3) for L in layers]}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_rows = [r for r in load_controlled_eval() if r["quadrant"] in args.quadrants]
    eval_rows = filter_to_held_out_behavioral_split(eval_rows)
    if args.limit:
        eval_rows = eval_rows[:args.limit]
    print(f"Loaded {len(eval_rows)} prompts across quadrants {args.quadrants} "
          f"(A/D restricted to held_out_behavioral split).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading {args.stage}...")
    model = load_stage_model(args.stage)

    all_directions = np.load(direction_path)
    directions_by_layer = {L: torch.from_numpy(all_directions[L]) for L in layers}

    out_rows = []
    baseline_name, steered_name = f"{tag}_baseline", f"{tag}_steered"

    if args.skip_baseline:
        out_rows.extend(load_existing_baseline(args.quadrants))
        # Relabel to this run's own condition name so downstream stats tools
        # (which match on the {tag}_baseline / {tag}_steered pair) find them.
        for row in out_rows:
            row["stage"] = baseline_name
        print(f"\n=== Condition 1/2: reused {len(out_rows)} existing baseline rows ===")
        if len(out_rows) < len(eval_rows):
            print(f"  WARNING: only found {len(out_rows)}/{len(eval_rows)} baseline rows for "
                  f"quadrants {args.quadrants} -- causal_ablation_raw_wide.json may not cover all of them.")
    else:
        print("\n=== Condition 1/2: baseline (no steering) ===")
        run_condition(model, tokenizer, eval_rows, device, baseline_name, out_rows)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_rows, f, ensure_ascii=False, indent=2)

    print("\n=== Condition 2/2: steered (hooks active) ===")
    handles = register_steering_hooks(model, directions_by_layer, alphas_by_layer)
    try:
        run_condition(model, tokenizer, eval_rows, device, steered_name, out_rows)
    finally:
        for h in handles:
            h.remove()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    cfg = build_run_config(args, layers, alphas_by_layer, out_path)
    meta_path = save_run_config(cfg, out_path)

    print(f"\nDone. {len(out_rows)} rows saved to {out_path}")
    print(f"Run config/metadata saved to {meta_path}")
    print("\nStatistical analysis (reuses existing, already-tested infra -- not reimplemented here):")
    print(f"  python -m src.analysis.summarize_causal_ablation --file {out_path} --stage {tag}")
    for q in args.quadrants:
        cat = "refusal" if q in ("A", "C") else "refusal"  # induced-refusal is the relevant test for either
        print(f"  python -m src.analysis.mcnemar_causal_ablation --file {out_path} "
              f"--conditions {baseline_name} {steered_name} --quadrant {q} --category {cat}")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
