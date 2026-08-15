"""
Component 5b: steering (activation addition) -- causal complement to
ablation. Adds the refusal direction INTO the residual stream instead of
removing it, on quadrant D (genuinely benign) prompts: does nudging
toward "looks harmful on this axis" induce refusal where none is
warranted?

Magnitude is anchored to real data, not an arbitrary constant: alpha per
layer = that layer's mean quadrant-A projection at M3 (Component 4,
already computed) -- i.e., push D's activations to look, on this axis,
like a typical obviously-harmful prompt does.

Same layer range as the wide ablation (14-28), since that's the range
that fully suppressed both target behaviors -- the natural range to test
the opposite intervention on.

Runs on Colab (GPU) -- real generation, same as eval_causal_ablation.py.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.convert_slow_tokenizers_checkpoints_to_fast import args

from src.training.model import load_stage_model
from src.analysis.eval_causal_ablation import get_decoder_layers, generate_batch, load_controlled_eval, BATCH_SIZE

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MAX_NEW_TOKENS = 200  # must match eval_causal_ablation.py / eval_behavioral.py
STEER_LAYERS = list(range(14, 29))


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


def load_existing_baseline_D(path="results/raw/causal_ablation_raw_wide.json"):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [r for r in rows if r["stage"] == "M3_baseline" and r["quadrant"] == "D"]


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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=None,
                     help="Steer only this single hidden_states layer instead of the full 14-28 range")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-baseline", action="store_true",
                         help="Reuse quadrant-D M3_baseline rows from causal_ablation_raw_wide.json")
    parser.add_argument("--stage", default="M3", choices=["M3", "M3-direct"],
                        help="Model stage to steer (default: M3)")
    
    args = parser.parse_args()

    direction_path = Path(f"results/refusal_direction/{args.stage}_direction.npy")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_rows = [r for r in load_controlled_eval() if r["quadrant"] == "D"]
    if args.limit:
        eval_rows = eval_rows[:args.limit]
    print(f"Loaded {len(eval_rows)} quadrant-D (benign) prompts.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading M3...")
    stage_name = args.stage
    model = load_stage_model(stage_name)

    all_directions = np.load(direction_path)
    with open("results/refusal_direction/quadrant_projections.json", encoding="utf-8") as f:
        quadrant_a_proj = json.load(f)["M3"]["A"]

    with open("results/refusal_direction/quadrant_projections.json", encoding="utf-8",) as f:
        quadrant_projections = json.load(f)

    quadrant_a_proj = quadrant_projections[stage_name]["A"]

    steer_layers = [args.layer] if args.layer is not None else STEER_LAYERS
    directions_by_layer = {L: torch.from_numpy(all_directions[L]) for L in steer_layers}
    alphas_by_layer = {L: float(quadrant_a_proj[L]) for L in steer_layers}
    print(f"Steering layers {steer_layers}, alpha (quadrant-A mean projection): "
      f"{[round(alphas_by_layer[L], 2) for L in steer_layers]}")

    out_rows = []
    suffix = f"_L{args.layer}" if args.layer is not None else ""
    if args.stage == "M3" and args.layer is None:
        out_path = Path("results/raw/steering_raw_D.json")
    else:
        suffix = f"_L{args.layer}" if args.layer is not None else ""
        out_path = Path(f"results/raw/steering_raw_{args.stage}_D{suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.skip_baseline:
        out_rows.extend(load_existing_baseline_D())
        print(f"\n=== Condition 1/2: reused {len(out_rows)} existing baseline rows ===")
    else:
        print("\n=== Condition 1/2: baseline (no steering) ===")
        run_condition(model, tokenizer, eval_rows, device, "M3_baseline", out_rows)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_rows, f, ensure_ascii=False, indent=2)

    print("\n=== Condition 2/2: steered (hooks active) ===")
    handles = register_steering_hooks(model, directions_by_layer, alphas_by_layer)
    try:
        run_condition(model, tokenizer, eval_rows, device, "M3_steered", out_rows)
    finally:
        for h in handles:
            h.remove()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
    print(f"\nDone. {len(out_rows)} rows saved to {out_path}")
    

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()