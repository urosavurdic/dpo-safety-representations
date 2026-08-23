"""
Next Steps item 4: actually diagnose (not just avoid) the degenerate-
collapse mechanism under multi-layer steering. Runs generation under four
conditions on the same small set of quadrant-D prompts, tracking the
residual-stream norm at every decoder layer, every generation step, via
ResidualNormTracker (src/interpretability/residual_norm_tracking.py):

  1. baseline           -- no steering at all (establishes the "trained,
                            typical" per-layer norm range via
                            compute_baseline_range).
  2. collapsing          -- layers 14-28 simultaneously, alpha_coefficient
                            1.0 (uncorrected) -- replicates the historical
                            98%-degenerate config from
                            steering_raw_D_MULTILAYER_14to28_DEPRECATED.json
                            as closely as this script's defaults allow.
  3. noncollapsing        -- single layer 24 (this repo's current steering
                            default, inside the ablation-validated 24-28
                            range) -- the comparison case that mostly
                            doesn't collapse.
  4. collapsing_norm_preserving (only with --also-test-fix) -- SAME layers
                            as (2), but every steering hook is replaced
                            with make_norm_preserving_steering_hook instead
                            of the normal additive hook -- directly tests
                            whether removing the magnitude-growth component
                            specifically (while still injecting the
                            direction) avoids the collapse. If this
                            condition's degenerate rate drops back toward
                            (3)'s while (2)'s stays high, that's real
                            evidence for the magnitude/norm-range
                            hypothesis, not just a plausible story.

REQUIRES a GPU and HF Hub network access for the actual model -- neither
was available in the sandboxed environment this script was written in (see
CLAUDE.md's documented sandbox limitations). This has been checked for
import-time and structural correctness (see the test file's coverage of
the pure-logic helpers below) but the actual generation/tracking has NOT
been executed against the real model by the agent that wrote it -- run it
for real, then look at the output file and the follow-up plotting script
(plot_residual_norms.py) before drawing any conclusion.

Usage:
    python -m src.analysis.eval_residual_norm_diagnostic
    python -m src.analysis.eval_residual_norm_diagnostic --also-test-fix --n-prompts 12
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.analysis.eval_causal_ablation import filter_to_held_out_behavioral_split, load_controlled_eval
from src.analysis.eval_refusal_classifier import is_degenerate
from src.analysis.eval_steering_v2 import make_steering_hook, resolve_alphas
from src.interpretability.residual_norm_tracking import (
    ResidualNormTracker,
    compare_to_baseline,
    compute_baseline_range,
    first_step_exceeding_p99,
    make_norm_preserving_steering_hook,
)
from src.training.eval_generation import build_generation_prompt
from src.training.model import load_stage_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MAX_NEW_TOKENS = 200  # must match eval_causal_ablation.py / eval_steering_v2.py
COLLAPSING_LAYERS = list(range(14, 29))    # historical multi-layer collapse config
NONCOLLAPSING_LAYERS = [24]                # current single-layer default


def get_decoder_layers(model):
    return model.model.layers


def generate_one_with_tracking(model, tokenizer, prompt, device, decoder_layers,
                                steering_hooks_by_layer=None, max_new_tokens=MAX_NEW_TOKENS):
    """steering_hooks_by_layer: {decoder_idx: hook_fn} or None for no
    steering. Registers the given steering hooks AND a full-model
    ResidualNormTracker together (order matters: PyTorch runs forward
    hooks in registration order, so the tracker is registered AFTER the
    steering hook at any layer that has both, meaning it records the
    ALREADY-STEERED value at that layer -- the point of this diagnostic
    is exactly to see the post-steering norm, so this ordering is
    intentional, not incidental)."""
    text = build_generation_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    steering_handles = []
    if steering_hooks_by_layer:
        for idx, hook_fn in steering_hooks_by_layer.items():
            steering_handles.append(decoder_layers[idx].register_forward_hook(hook_fn))

    tracker = ResidualNormTracker()
    tracker.register(model=None, decoder_layers=decoder_layers)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        records = tracker.collect()
    finally:
        tracker.remove()
        for h in steering_handles:
            h.remove()

    return response, records


def build_steering_hooks(direction_tensor_by_layer, alphas_by_layer, norm_preserving=False):
    hooks = {}
    for hs_index, direction in direction_tensor_by_layer.items():
        decoder_idx = hs_index - 1
        alpha = alphas_by_layer[hs_index]
        hooks[decoder_idx] = (
            make_norm_preserving_steering_hook(direction, alpha) if norm_preserving
            else make_steering_hook(direction, alpha)
        )
    return hooks


def summarize_config(records_list):
    """records_list: list of per-prompt {"response": str, "is_degenerate": bool}
    dicts for one condition. Pure aggregation, no torch -- testable without
    a real model."""
    n = len(records_list)
    n_degenerate = sum(1 for r in records_list if r["is_degenerate"])
    return {
        "n_prompts": n,
        "n_degenerate": n_degenerate,
        "degenerate_rate": n_degenerate / n if n else None,
    }


def build_norm_summary(condition_records, baseline_range):
    """condition_records: {decoder_idx: list[step] of norms} for ONE
    prompt/config. Returns {decoder_idx: {"first_step_exceeding_p99":
    int|None, "max_z_score": float}} -- the headline "how far outside, and
    when" figures for this one run, reusing compare_to_baseline/
    first_step_exceeding_p99 rather than recomputing anything."""
    comparison = compare_to_baseline(condition_records, baseline_range)
    out = {}
    for idx, entries in comparison.items():
        z_scores = [e["z_score"] for e in entries if np.isfinite(e["z_score"])]
        out[idx] = {
            "first_step_exceeding_p99": first_step_exceeding_p99(entries),
            "max_z_score": max(z_scores) if z_scores else None,
        }
    return out


def run_diagnostic(model, tokenizer, device, direction_path, stage, prompts,
                    also_test_fix=False):
    decoder_layers = get_decoder_layers(model)
    all_directions = np.load(direction_path)

    conditions = {
        "baseline": None,
        "collapsing": (COLLAPSING_LAYERS, False),
        "noncollapsing": (NONCOLLAPSING_LAYERS, False),
    }
    if also_test_fix:
        conditions["collapsing_norm_preserving"] = (COLLAPSING_LAYERS, True)

    raw_results = {name: [] for name in conditions}

    for prompt in prompts:
        for name, spec in conditions.items():
            if spec is None:
                hooks = None
            else:
                layers, norm_preserving = spec
                alphas_by_layer = resolve_alphas(layers, stage, "quadrant_a_projection", None, 1.0)
                direction_tensor_by_layer = {L: torch.from_numpy(all_directions[L]) for L in layers}
                hooks = build_steering_hooks(direction_tensor_by_layer, alphas_by_layer,
                                              norm_preserving=norm_preserving)

            response, norm_records = generate_one_with_tracking(
                model, tokenizer, prompt, device, decoder_layers, steering_hooks_by_layer=hooks,
            )
            raw_results[name].append({
                "prompt": prompt, "response": response,
                "is_degenerate": is_degenerate(response),
                "norm_records": norm_records,
            })
            print(f"  [{name}] {prompt[:50]!r} -> degenerate={is_degenerate(response)}")

    return raw_results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="M3")
    parser.add_argument("--quadrant", default="D", choices=["A", "B", "C", "D"])
    parser.add_argument("--n-prompts", type=int, default=8,
                         help="Small on purpose -- this is a per-token, per-layer, per-step "
                              "diagnostic, not a statistical power run; see run_full_steering.py "
                              "for the actual sufficiency numbers.")
    parser.add_argument("--also-test-fix", action="store_true",
                         help="Also run the collapsing layer set with the norm-preserving hook "
                              "substituted in, to directly test whether it prevents the collapse.")
    parser.add_argument("--direction-source", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    direction_path = Path(args.direction_source or f"results/refusal_direction/{args.stage}_direction.npy")
    out_path = Path(args.out or f"results/raw/residual_norm_diagnostic_{args.stage}.json")
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists -- pass --out to write elsewhere, "
                               "this script doesn't overwrite by default.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_rows = [r for r in load_controlled_eval() if r["quadrant"] == args.quadrant]
    eval_rows = filter_to_held_out_behavioral_split(eval_rows)
    prompts = [r["prompt"] for r in eval_rows[:args.n_prompts]]
    print(f"Using {len(prompts)} quadrant-{args.quadrant} held-out-behavioral prompts.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading {args.stage}...")
    model = load_stage_model(args.stage)

    raw_results = run_diagnostic(model, tokenizer, device, direction_path, args.stage, prompts,
                                  also_test_fix=args.also_test_fix)

    baseline_pooled = {}
    for entry in raw_results["baseline"]:
        for idx, steps in entry["norm_records"].items():
            baseline_pooled.setdefault(idx, []).extend(steps)
    baseline_range = compute_baseline_range(baseline_pooled)

    output = {"stage": args.stage, "quadrant": args.quadrant, "baseline_range": baseline_range,
              "conditions": {}}
    for name, entries in raw_results.items():
        output["conditions"][name] = {
            "summary": summarize_config(entries),
            "prompts": [
                {
                    "prompt": e["prompt"], "response": e["response"], "is_degenerate": e["is_degenerate"],
                    "norm_records": e["norm_records"],
                    "norm_summary": build_norm_summary(e["norm_records"], baseline_range) if name != "baseline" else None,
                }
                for e in entries
            ],
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== Summary ===")
    for name, cond in output["conditions"].items():
        s = cond["summary"]
        print(f"  {name:28s} degenerate: {s['n_degenerate']}/{s['n_prompts']} ({s['degenerate_rate']:.0%})"
              if s["degenerate_rate"] is not None else f"  {name}: n=0")
    print(f"\nWritten to {out_path}")
    print(f"Next: python -m src.analysis.plot_residual_norms --file {out_path}")


if __name__ == "__main__":
    main()
