"""Per-checkpoint measurements: activations, first-response-token KL, ‖ΔW‖.

``forward_batch`` is a faithful replica of
``frozen_run_pipeline.activation_batch`` (left padding, the same chat template,
``final`` = last prompt token, ``pooled`` = mean of the last five), with two
additions: it can keep only some layers, and it returns the log-softmax of the
final-position logits. That distribution is the model's prediction for the
first response token. Its KL from the starting checkpoint is the
output-space update size reported next to the weight and activation sizes.

Batching mirrors ``stage_extract``: length-sorted shards of 8 restored to
benchmark order, with the same OOM backoff.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pipeline.frozen_run_pipeline import POOL_WINDOW, clear_cuda_cache, token_measure
from src.pipeline.shards import plan_shards, run_with_oom_backoff

ANALYSIS_LAYERS = (16, 20, 24, 28)


def forward_batch(model, tokenizer, prompts, device, layers=None, want_logprobs=True):
    """``(final, pooled, logprobs)`` for one batch.

    ``final``/``pooled`` are ``(n, len(layers), hidden)`` float32, with layer
    index ``l`` meaning ``hidden_states[l]`` (0 = embeddings), as in the stored
    ``(654, 29, 1536)`` arrays. ``logprobs`` is ``(n, vocab)`` float32 or None.
    """
    import torch

    from src.training.eval_generation import build_generation_prompt

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        texts = [build_generation_prompt(tokenizer, prompt) for prompt in prompts]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        keep = list(range(len(hidden_states))) if layers is None else list(layers)
        n, hidden_dim = len(prompts), hidden_states[0].shape[-1]
        final = np.zeros((n, len(keep), hidden_dim), dtype=np.float32)
        pooled = np.zeros_like(final)
        for row_index in range(n):
            attention_length = int(inputs["attention_mask"][row_index].sum().item())
            window = min(POOL_WINDOW, attention_length)
            for slot, layer in enumerate(keep):
                hidden = hidden_states[layer][row_index]
                final[row_index, slot] = hidden[-1].float().cpu().numpy()
                pooled[row_index, slot] = hidden[-window:].float().mean(dim=0).cpu().numpy()

        logprobs = None
        if want_logprobs:
            logprobs = torch.log_softmax(outputs.logits[:, -1, :].float(), dim=-1).cpu().numpy()
        return final, pooled, logprobs
    finally:
        tokenizer.padding_side = original_padding_side


def kl_rows(ref_logprobs: np.ndarray, logprobs: np.ndarray) -> np.ndarray:
    """``KL(p_ref || p)`` per row, in nats. Both arguments are log-probabilities."""
    ref = ref_logprobs.astype(np.float64)
    return (np.exp(ref) * (ref - logprobs.astype(np.float64))).sum(axis=1)


def extract(model, tokenizer, rows, device, *, batch_size=8, layers=None,
            ref_logprobs=None, keep_logprobs=False) -> dict:
    """Run every row once, in length-sorted shards, and restore benchmark order.

    With ``ref_logprobs`` (``(len(rows), vocab)``, benchmark order), the KL of
    each row's first-token distribution from the reference is computed shard by
    shard, so the full ``(654, vocab)`` array is never held twice.
    """
    position = {row["record_id"]: i for i, row in enumerate(rows)}
    shards = plan_shards(rows, batch_size, measure=token_measure(tokenizer))
    n_layers = None
    final = pooled = None
    kl = np.full(len(rows), np.nan)
    entropy = np.full(len(rows), np.nan)
    kept = [None] * len(rows) if keep_logprobs else None

    for shard in shards:
        def call(sub):
            return forward_batch(model, tokenizer, [r["prompt"] for r in sub], device,
                                 layers=layers, want_logprobs=True)

        f, p, lp = run_with_oom_backoff(
            shard, call,
            combine=lambda a, b: tuple(np.concatenate([x, y], axis=0) for x, y in zip(a, b)),
            on_retry=clear_cuda_cache,
        )
        if final is None:
            n_layers = f.shape[1]
            final = np.zeros((len(rows), n_layers, f.shape[2]), dtype=np.float32)
            pooled = np.zeros_like(final)
        idx = np.array([position[r["record_id"]] for r in shard])
        final[idx], pooled[idx] = f, p
        entropy[idx] = -(np.exp(lp.astype(np.float64)) * lp).sum(axis=1)
        if ref_logprobs is not None:
            kl[idx] = kl_rows(ref_logprobs[idx], lp)
        if keep_logprobs:
            for offset, i in enumerate(idx):
                kept[i] = lp[offset].astype(np.float16)

    out = {"final": final, "pooled": pooled, "entropy": entropy,
           "layers": list(layers) if layers is not None else list(range(n_layers))}
    if ref_logprobs is not None:
        out["kl"] = kl
    if keep_logprobs:
        out["logprobs"] = np.stack(kept)
    return out


# ----------------------------------------------------------- weight size -- #

def adapter_update(adapter_dir: Path) -> dict:
    """``{(layer, module): Update}`` for one adapter directory, relative to the
    model it was trained on (for a seed run: the merged start checkpoint)."""
    from safetensors.numpy import load_file

    from src.stagecmp.weights import LowRankTerm, Update, parse_adapter

    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    scaling = float(config["lora_alpha"]) / float(config["r"])
    factors = parse_adapter(load_file(str(adapter_dir / "adapter_model.safetensors")))
    return {slot: Update([LowRankTerm(sign=1.0, b=f["B"], a=f["A"], scaling=scaling)])
            for slot, f in factors.items()}


def update_norms(updates: dict, n_layers: int = 28) -> dict:
    """Frobenius norm of the merged change: total, residual writers, two halves."""
    from src.stagecmp.weights import RESIDUAL_WRITERS, frobenius_norm

    sq = {slot: frobenius_norm(u) ** 2 for slot, u in updates.items()}
    half = n_layers // 2
    return {
        "total": float(np.sqrt(sum(sq.values()))),
        "residual_writers": float(np.sqrt(sum(v for (l, m), v in sq.items()
                                              if m in RESIDUAL_WRITERS))),
        "early_layers": float(np.sqrt(sum(v for (l, m), v in sq.items() if l < half))),
        "late_layers": float(np.sqrt(sum(v for (l, m), v in sq.items() if l >= half))),
        "n_slots": len(sq),
    }
