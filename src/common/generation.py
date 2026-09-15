"""Batched generation and activation-hook plumbing for intervention runs.

Shared by the steering and residual-norm scripts. Imports torch, so it must not
be pulled in by any CPU-only path -- src/common/__init__.py is intentionally
empty so importing a sibling module here does not drag torch along.

Extracted verbatim from eval_causal_ablation.py, which is now archived. The
behaviour is unchanged; only the surrounding documentation was corrected.
"""

import json

import torch

from src.training.eval_generation import build_generation_prompt

__all__ = [
    "MODEL_NAME",
    "BATCH_SIZE",
    "MAX_NEW_TOKENS",
    "get_decoder_layers",
    "ablate_direction",
    "make_ablation_hook",
    "register_ablation_hooks",
    "load_controlled_eval",
    "filter_to_held_out_behavioral_split",
    "generate_batch",
]

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
BATCH_SIZE = 8

# Matches the generation length used for the behavioural runs. eval_behavioral.py
# has no constant of its own: its main path calls
# src.training.eval_generation.generate, whose default is 200, and its only
# explicit override is max_new_tokens=60 for the short capability probes. Change
# both together or the comparison stops being like-for-like.
MAX_NEW_TOKENS = 200

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

def filter_to_held_out_behavioral_split(eval_rows):
    """Keep all quadrant B/C rows untouched (no split assigned - not used to
    build the direction, so no circularity risk), plus only the
    "held_out_behavioral" half of quadrant A/D. This is the causal-testing
    counterpart of eval_refusal_direction.filter_to_direction_estimation_split
    - apply this wherever A/D prompts get used for a CAUSAL test (ablation,
    steering) so the direction's causal effect is never judged on the same
    A/D prompts it was estimated from. Do NOT apply this in eval_behavioral.py
    or anywhere characterizing general behavior (not a circularity risk there,
    and using only 20% of A/D would needlessly throw away statistical power).
    Rows with quadrant A/D but no "split" key (activations extracted before
    the split existed) are dropped here rather than silently included -
    re-extract activations/rebuild the eval set first if you hit this."""
    return [
        r for r in eval_rows
        if r["quadrant"] not in ("A", "D") or r.get("split") == "held_out_behavioral"
    ]

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
