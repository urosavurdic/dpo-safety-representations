"""
Component 2: activation extraction across M0-M3 on the controlled eval set.
Extracts residual-stream hidden states at every layer, at two positions
(final prompt token + pooled last-5-tokens). Feeds Components 3 (probes)
and 4 (refusal direction). Saved per-stage immediately, not at the end.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.training.model import try_load_stage_model
from src.training.eval_generation import build_generation_prompt

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
STAGES = [
    "M0", "M1", "M2", "M3", "M3_direct",
    "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt",
]  # alt branch trains/pushes independently across sessions, may be partially ready at any time
BATCH_SIZE = 8
POOL_WINDOW = 5


def compute_pool_window(attn_len, pool_window=POOL_WINDOW):
    return min(pool_window, attn_len)


def load_controlled_eval(path="data/processed/controlled_eval.jsonl"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def extract_batch(model, tokenizer, prompts, device):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        texts = [build_generation_prompt(tokenizer, p) for p in prompts]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        num_layers = len(outputs.hidden_states)
        hidden_dim = outputs.hidden_states[0].shape[-1]
        batch_final = np.zeros((len(prompts), num_layers, hidden_dim), dtype=np.float32)
        batch_pooled = np.zeros_like(batch_final)

        for i in range(len(prompts)):
            attn_len = int(inputs["attention_mask"][i].sum().item())
            pool_window = compute_pool_window(attn_len)
            for layer_idx in range(num_layers):
                layer_hidden = outputs.hidden_states[layer_idx][i]
                batch_final[i, layer_idx] = layer_hidden[-1].float().cpu().numpy()
                batch_pooled[i, layer_idx] = layer_hidden[-pool_window:].float().mean(dim=0).cpu().numpy()

        return batch_final, batch_pooled
    finally:
        tokenizer.padding_side = original_padding_side


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit eval prompts (dry run)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_rows = load_controlled_eval()
    if args.limit:
        eval_rows = eval_rows[:args.limit]
    print(f"Loaded {len(eval_rows)} controlled-eval prompts.")

    out_dir = Path("results/activations")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for stage_name in STAGES:
        final_path = out_dir / f"{stage_name}_final.npy"
        pooled_path = out_dir / f"{stage_name}_pooled.npy"
        meta_path = out_dir / f"{stage_name}_metadata.json"

        if final_path.exists() and pooled_path.exists() and meta_path.exists():
            print(f"\n=== {stage_name}: already extracted, skipping ===")
            continue

        print(f"\n=== {stage_name}: extracting (batch size {BATCH_SIZE}) ===")
        model = try_load_stage_model(stage_name)
        if model is None:
            continue

        all_final, all_pooled = [], []
        for i in range(0, len(eval_rows), BATCH_SIZE):
            batch_rows = eval_rows[i:i + BATCH_SIZE]
            batch_final, batch_pooled = extract_batch(model, tokenizer, [r["prompt"] for r in batch_rows], device)
            all_final.append(batch_final)
            all_pooled.append(batch_pooled)
            print(f"    {min(i + BATCH_SIZE, len(eval_rows))}/{len(eval_rows)} done")

        final_array = np.concatenate(all_final, axis=0)
        pooled_array = np.concatenate(all_pooled, axis=0)
        np.save(final_path, final_array)
        np.save(pooled_path, pooled_array)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([{"prompt": r["prompt"], "quadrant": r["quadrant"], "source": r["source"]} for r in eval_rows], f, ensure_ascii=False, indent=2)

        print(f"Saved {stage_name}: final {final_array.shape}, pooled {pooled_array.shape}")
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\nDone. Activations in results/activations/")


if __name__ == "__main__":
    main()