"""Sharded, banked generation with the frozen settings.

Greedy decoding, at most 200 new tokens, repetition penalty 1.1, left padding,
the ChatML stop tokens: ``crossbranch.worker.generation_batch_with_injector``,
which is a line-for-line replica of ``frozen_run_pipeline.generation_batch``
plus an optional activation injector. Rows use ``result_row``'s schema, so
the judge loaders read them unchanged.

Prompts are split into length-sorted shards of ``batch`` rows. Each shard is
written atomically to the work directory; every ``bank_every`` shards the new
ones go to the bank in one commit. A rerun pulls the banked shards and
continues from the first missing one.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.pipeline.frozen_run_pipeline import MAX_NEW_TOKENS, clear_cuda_cache, token_measure
from src.pipeline.shards import DeadlineReached, plan_shards, run_with_oom_backoff
from src.seedrep.train import fault_point


def response_row(row, label: str, condition: str, response: str, shas) -> dict:
    return {
        "record_id": row.get("record_id"),
        "prompt": row["prompt"],
        "quadrant": row["quadrant"],
        "source": row.get("source", row.get("source_dataset")),
        "source_dataset": row.get("source_dataset"),
        "c_construction": row.get("c_construction"),
        "pair_id": row.get("pair_id"),
        "split": row.get("split"),
        "stage": condition,
        "condition": condition,
        "model_stage": label,
        "response": response,
        "benchmark_sha256": shas[0],
        "split_manifest_sha256": shas[1],
        "generation": {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False,
                       "repetition_penalty": 1.1},
    }


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def generate(model, tokenizer, rows, device, *, unit: str, label: str, condition: str,
             shas, bank, work: Path, scratch: Path, batch: int = 8, bank_every: int = 10,
             injector=None, deadline=None, shard_seconds: float = 30.0, log=print) -> list[dict]:
    """All rows' responses in input order; resumes from banked shards."""
    from src.crossbranch.worker import generation_batch_with_injector

    local = Path(work) / "gen" / unit
    remote = f"gen/{unit}"
    for name in bank.list(remote + "/"):
        if name.endswith(".json") and not (local / Path(name).name).exists():
            bank.get(name, local / Path(name).name)
    shards = plan_shards(rows, batch, measure=token_measure(tokenizer))
    pending_bank = []

    def flush(message):
        if pending_bank:
            bank.put({f"{remote}/{p.name}": p for p in pending_bank}, message)
            pending_bank.clear()

    try:
        for index, shard in enumerate(shards):
            path = local / f"shard_{index:04d}.json"
            if path.exists():
                continue
            if deadline is not None and deadline.would_exceed(shard_seconds * 2):
                raise DeadlineReached(f"{unit}: stopped before shard {index}")
            fault_point("gen", index, scratch)
            responses = run_with_oom_backoff(
                shard,
                lambda sub: generation_batch_with_injector(model, tokenizer, sub, device,
                                                            MAX_NEW_TOKENS, injector),
                combine=lambda a, b: a + b,
                on_retry=clear_cuda_cache,
            )
            _atomic_json(path, [response_row(r, label, condition, resp, shas)
                                for r, resp in zip(shard, responses)])
            pending_bank.append(path)
            if len(pending_bank) >= bank_every:
                flush(f"{unit}: shards up to {index}")
                log(f"  [{unit}] {index + 1}/{len(shards)} shards banked")
    finally:
        # bank whatever finished, also when interrupted or out of time
        try:
            flush(f"{unit}: shards so far")
        except Exception as exc:  # never mask the original error
            log(f"  [{unit}] could not bank pending shards: {exc}")

    by_id = {}
    for index in range(len(shards)):
        for out in json.loads((local / f"shard_{index:04d}.json").read_text(encoding="utf-8")):
            by_id[out["record_id"]] = out
    ordered = [by_id[r["record_id"]] for r in rows]
    responses_path = local / "responses.json"
    _atomic_json(responses_path, ordered)
    bank.put({f"{remote}/responses.json": responses_path}, f"{unit}: consolidated responses")
    return ordered
