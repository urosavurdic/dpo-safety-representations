"""
Component 5: causal ablation check (H4). Multi-layer scope, hidden_states
layers 14-28 (decoder blocks 13-27, 0-indexed).

Runs ON COLAB (GPU) -- this does real generation, unlike Components 2-4.

Scope deliberately kept to generation only: writes a raw JSON file with
the SAME SCHEMA as eval_behavioral.py's output (stage set to
"M3_baseline" / "M3_ablated"). Classification and Wilson-CI stats are NOT
reimplemented here -- point the existing eval_refusal_classifier.py /
eval_stats.py at this file, treating the two conditions as two more
stages. (Reusing already-validated code beats re-guessing its signature.)

Ablation: at each target layer's output (every token position), project
out the component along that layer's M3 diff-in-means direction (from
Component 4, results/refusal_direction/M3_direction.npy):
    h' = h - (h . d) d,   d unit-normalized.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.convert_slow_tokenizers_checkpoints_to_fast import argparse

from src.training.model import load_stage_model
from src.training.eval_generation import build_generation_prompt

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 200  # ASSUMPTION -- must match eval_behavioral.py's setting exactly for a valid comparison. Check and edit if different.
ABLATE_LAYERS = list(range(24, 29))   # was range(14, 29) — narrower, targets the deepest layers  # hidden_states indices 14..28 inclusive


def get_decoder_layers(model):
    """Qwen2/Llama-style HF models expose the transformer blocks at
    model.model.layers. If this raises, print(model) and paste the
    top-level structure back rather than guessing further."""
    try:
        return model.model.layers
    except AttributeError as e:
        raise AttributeError(
            "Could not find model.model.layers -- the attribute path assumed "
            "here (standard for Qwen2ForCausalLM) doesn't match this checkpoint's "
            "actual class. Run print(model) and paste the top-level structure back."
        ) from e

def _output_suffix(layers):
    if layers[0] == 14: return "wide"
    if layers[0] == 24: return "narrow"
    return f"L{layers[0]}-{layers[-1]}"

def ablate_direction(hidden_states, direction):
    """Project OUT the component along `direction` from `hidden_states`.
    hidden_states: (..., hidden_dim). direction: (hidden_dim,), unit-normalized."""
    direction = direction.to(dtype=hidden_states.dtype, device=hidden_states.device)
    proj = torch.einsum("...h,h->...", hidden_states, direction)
    return hidden_states - proj.unsqueeze(-1) * direction


def make_ablation_hook(direction):
    """Handles both raw-tensor and tuple-with-hidden-states-first outputs,
    since this varies across HF model classes/versions."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            new_hidden = ablate_direction(output[0], direction)
            return (new_hidden,) + output[1:]
        return ablate_direction(output, direction)
    return hook


def register_ablation_hooks(model, directions_by_layer):
    """directions_by_layer: {hidden_states_index: (hidden_dim,) tensor}.
    Returns handles -- caller must .remove() them after use."""
    decoder_layers = get_decoder_layers(model)
    handles = []
    for hs_index, direction in directions_by_layer.items():
        decoder_idx = hs_index - 1  # hidden_states[i] = output of decoder_layers[i-1]
        handle = decoder_layers[decoder_idx].register_forward_hook(make_ablation_hook(direction))
        handles.append(handle)
    return handles


def load_controlled_eval(path="data/processed/controlled_eval.jsonl"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def generate_batch(model, tokenizer, prompts, device, max_new_tokens=MAX_NEW_TOKENS):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        texts = [build_generation_prompt(tokenizer, p) for p in prompts]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    finally:
        tokenizer.padding_side = original_padding_side


def run_condition(model, tokenizer, eval_rows, device, stage_name, out_rows):
    for i in range(0, len(eval_rows), BATCH_SIZE):
        batch_rows = eval_rows[i:i + BATCH_SIZE]
        responses = generate_batch(model, tokenizer, [r["prompt"] for r in batch_rows], device)
        for row, response in zip(batch_rows, responses):
            out_rows.append({
                "prompt": row["prompt"],
                "quadrant": row["quadrant"],
                "source": row["source"],
                "model_stage": stage_name,
                "response": response,
            })
        print(f"    [{stage_name}] {min(i + BATCH_SIZE, len(eval_rows))}/{len(eval_rows)} done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit eval prompts (dry run)")
    parser.add_argument("--skip-baseline", action="store_true",
                         help="Skip regenerating baseline -- use if you're reusing M3 rows from behavioral_eval_raw.json instead")
    parser.add_argument(
    "--stage",
    default="M3",
    choices=["M3", "M3_direct"],
    help="Model stage to run causal ablation on (default: M3)",
)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_rows = load_controlled_eval()
    if args.limit:
        eval_rows = eval_rows[:args.limit]
    print(f"Loaded {len(eval_rows)} controlled-eval prompts.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading {args.stage}...")
    model = load_stage_model(args.stage)

    direction_path = Path(f"results/refusal_direction/{args.stage}_direction.npy")
    all_directions = np.load(direction_path)  # (29, hidden_dim)
    directions_by_layer = {L: torch.from_numpy(all_directions[L]) for L in ABLATE_LAYERS}
    print(f"Loaded {args.stage} directions, ablating hidden_states layers {ABLATE_LAYERS[0]}-{ABLATE_LAYERS[-1]} "
          f"(decoder blocks {ABLATE_LAYERS[0]-1}-{ABLATE_LAYERS[-1]-1})")

    out_rows = []
    out_path = Path(f"results/raw/causal_ablation_raw_{_output_suffix(ABLATE_LAYERS)}.json") #out_path = Path("results/causal_ablation_raw_narrow.json") #out_path = Path("results/causal_ablation_raw.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_baseline:
        print("\n=== Condition 1/2: baseline (no ablation) ===")
        run_condition(model, tokenizer, eval_rows, device, f"{args.stage}_baseline", out_rows)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_rows, f, ensure_ascii=False, indent=2)
        print(f"Baseline saved ({len(out_rows)} rows) -- checkpoint before touching hooks.")
    else:
        print("\n=== Condition 1/2: SKIPPED (reusing existing baseline rows) ===")

    print("\n=== Condition 2/2: ablated (hooks active) ===")
    handles = register_ablation_hooks(model, directions_by_layer)
    try:
        run_condition(model, tokenizer, eval_rows, device, f"{args.stage}_ablated", out_rows)
    finally:
        for h in handles:
            h.remove()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
    print(f"\nDone. {len(out_rows)} rows saved to {out_path}")
    print("Next: point eval_refusal_classifier.py / eval_stats.py at this file, "
          f"treating '{args.stage}_baseline' and '{args.stage}_ablated' as two more stages.")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()