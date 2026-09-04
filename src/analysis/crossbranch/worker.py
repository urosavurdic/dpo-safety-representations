"""GPU worker: run one (condition, coefficient) unit.

Deliberately standalone. It imports free functions from v2_pipeline but adds
no stage, subcommand or edit there -- that module's stage graph, parser and
contract tests are frozen.

``generation_batch`` is replicated here (~20 lines) rather than imported,
because the injector needs a ``set_batch`` call between tokenisation and
``generate`` and the frozen version offers no hook for it. The repo already
duplicates ``load_controlled_eval`` across modules for the same
torch-import-coupling reason.

``main()`` is untested by design, matching eval_steering_v2.main /
eval_causal_ablation.main / the v2_pipeline intervention mains: it needs a
real model on a GPU. Everything it composes is unit-tested elsewhere.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.crossbranch.branches import (
    BRANCHES,
    COEFFICIENTS,
    INJECT_LAYER,
    INJECT_MODE,
    MODEL,
    P0,
    POSITION,
    VECTOR,
    checkpoint_for,
    direction_tag,
    get,
    resolve,
)
from src.analysis.crossbranch.delta import DELTAS_DIR, artifact_path, load_delta_map
from src.analysis.v2_pipeline import (
    DEFAULT_ACT_BATCH,
    DEFAULT_GEN_BATCH,
    build_context,
    clear_cuda_cache,
    free_model,
    plan_shards,
    quadrant_rows,
    result_row,
    run_sharded,
    run_with_oom_backoff,
    token_measure,
)
from src.v2_io import write_json_lf

RAW_DIR = Path("results/crossbranch/raw")
SHARD_DIR = RAW_DIR / "shards"


def output_path(tag: str, condition: str, coef: float | None, out_dir=RAW_DIR) -> Path:
    suffix = "na" if coef is None else f"{coef:g}"
    return Path(out_dir) / f"crossbranch_{tag}_{condition}_coef{suffix}.json"


def generation_batch_with_injector(
    model, tokenizer, rows, device, max_new_tokens, injector=None
):
    """Faithful replica of v2_pipeline.generation_batch, plus set_batch.

    The only behavioural difference is the injector call, which must happen
    after tokenisation (it needs the attention mask) and before generate.
    """
    import torch

    from src.analysis.crossbranch.inject import assert_greedy_single_beam
    from src.training.eval_generation import (
        build_generation_prompt,
        get_generation_eos_ids,
    )

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        texts = [build_generation_prompt(tokenizer, r["prompt"]) for r in rows]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

        if injector is not None:
            injector.set_batch(
                [r["record_id"] for r in rows], inputs.get("attention_mask")
            )

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=get_generation_eos_ids(tokenizer),
        )
        assert_greedy_single_beam(**gen_kwargs)

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    finally:
        tokenizer.padding_side = original_padding_side


def make_generator(ctx, model, tokenizer, device, condition, checkpoint, meta, injector):
    def process(shard):
        def call(sub_shard):
            responses = generation_batch_with_injector(
                model, tokenizer, sub_shard, device, ctx.max_new_tokens, injector
            )
            out = []
            for row, response in zip(sub_shard, responses):
                r = result_row(row, ctx, condition, condition, checkpoint, response)
                r.update(meta)
                out.append(r)
            return out

        return run_with_oom_backoff(
            shard, call, combine=lambda a, b: a + b, on_retry=clear_cuda_cache
        )

    return process


def rows_for_unit(ctx_rows, quadrants, limit=None):
    """Quadrant/held-out-filtered rows for one unit, then optionally capped.

    Pulled out of run_unit so it is testable without a model: the ordering
    matters. Applying a raw-benchmark limit before quadrant_rows (which
    build_context's own --limit does) would slice from the front of the
    654-row file, where quadrant A/D rows interleave
    direction_estimation/held_out_behavioral -- so limiting first can hand
    back fewer, and a different sample, than the caller asked for.
    """
    rows = quadrant_rows(ctx_rows, list(quadrants))
    if limit is not None:
        rows = rows[:limit]
    return rows


def run_unit(args) -> bool:
    from src.analysis.crossbranch.inject import PerRowDeltaInjector
    from src.training.model import load_stage_model

    cond = get(args.condition)
    if cond.stage_gate != P0 and not args.allow_stage2:
        raise SystemExit(
            f"{cond.name} is gated '{cond.stage_gate}', not '{P0}'. The first "
            "implementation pass runs Stage 1 only. Pass --allow-stage2 "
            "deliberately if that is really what you intend."
        )
    if cond.kind == MODEL and args.coef is not None:
        raise SystemExit(f"{cond.name} is a model condition and takes no --coef.")
    if cond.kind == VECTOR and args.coef is None:
        raise SystemExit(f"{cond.name} is a vector condition and requires --coef.")

    # --limit is a MICRO-VALIDATION knob on the (quadrant-filtered, held-out)
    # rows this unit would otherwise run on -- not on the raw 654-row
    # benchmark. build_context's own --limit slices the raw benchmark BEFORE
    # quadrant/split filtering, and quadrant A/D rows interleave
    # direction_estimation/held_out_behavioral in the file, so applying it
    # there would silently hand back fewer (and a different sample of) rows
    # than requested. Detach it, build the full context, filter, then apply.
    requested_limit = args.limit
    args.limit = None
    ctx = build_context(args)
    roles = resolve(args.source_branch, args.target_branch)
    tag = direction_tag(args.source_branch, args.target_branch)
    checkpoint = checkpoint_for(args.condition, args.source_branch, args.target_branch)

    out_path = output_path(tag, args.condition, args.coef, args.out_dir)
    if out_path.exists() and not args.force:
        print(f"{out_path} exists; --force to rerun.")
        return True

    rows = rows_for_unit(ctx.rows, args.quadrants, requested_limit)
    print(
        f"{args.condition} coef={args.coef} -> checkpoint {checkpoint} "
        f"on {len(rows)} rows ({''.join(args.quadrants)})"
    )

    meta = {
        "coef": args.coef,
        "inject_layer": args.layer if cond.kind == VECTOR else None,
        "inject_mode": INJECT_MODE if cond.kind == VECTOR else None,
        "position": POSITION if cond.kind == VECTOR else None,
        "condition_kind": cond.kind,
        "delta_source": cond.artifact,
        "transfer_map": "identity" if cond.kind == VECTOR else None,
        "source_branch": roles["source_branch"],
        "target_branch": roles["target_branch"],
        "resolved_stage": checkpoint,
    }

    delta_map = None
    if cond.kind == VECTOR:
        path = artifact_path(cond.artifact, args.layer, args.deltas_dir)
        if not path.exists():
            raise SystemExit(
                f"{cond.name} needs {path}, which does not exist. Run\n"
                "  python -m src.analysis.crossbranch.delta\n"
                "first (it will itself refuse unless the four 654-row "
                "activation arrays are bound to the frozen benchmark)."
            )
        delta_map = load_delta_map(path)
        missing = [r["record_id"] for r in rows if r["record_id"] not in delta_map]
        if missing:
            raise SystemExit(
                f"{len(missing)} rows have no delta in {path}, first: {missing[:3]}"
            )
        meta["delta_artifact"] = str(path)

    model = tokenizer = injector = None
    try:
        import torch
        from transformers import AutoTokenizer

        model = load_stage_model(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if delta_map is not None:
            injector = PerRowDeltaInjector(
                delta_map, coef=args.coef, layer=args.layer, mode=INJECT_MODE
            ).register(model)
            meta["injected_norm_mean"] = float(
                sum(
                    float((abs(delta_map[r["record_id"]]) ** 2).sum() ** 0.5)
                    for r in rows
                )
                / max(len(rows), 1)
                * abs(args.coef)
            )

        store = ctx.store(Path(args.shard_dir))
        unit_key = store.unit_key(args.condition, f"coef{args.coef or 'na'}")
        shards = plan_shards(rows, ctx.gen_batch, measure=token_measure(tokenizer))
        finished = run_sharded(
            store,
            unit_key,
            shards,
            make_generator(
                ctx, model, tokenizer, device, args.condition, checkpoint, meta, injector
            ),
            ctx.deadline,
            label=f"{tag}/{args.condition}@{args.coef}",
        )
        if not finished:
            print("Deadline reached; unit is resumable from its committed shards.")
            return False

        write_json_lf(out_path, store.merge_unit(unit_key, order=ctx.order))
        write_json_lf(
            out_path.with_name(out_path.stem + "_binding.json"),
            {**ctx.bind(), "condition": args.condition, **meta,
             "n_rows": len(rows), "quadrants": list(args.quadrants)},
        )
        print(f"Wrote {out_path}")
        return True
    finally:
        if injector is not None:
            injector.remove()
        if model is not None:
            free_model(model)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cross-branch transfer worker (one unit).")
    p.add_argument("--condition", required=True)
    p.add_argument("--coef", type=float, default=None)
    p.add_argument("--source-branch", default="A", choices=sorted(BRANCHES))
    p.add_argument("--target-branch", default="B", choices=sorted(BRANCHES))
    p.add_argument("--layer", type=int, default=INJECT_LAYER)
    p.add_argument("--quadrants", nargs="+", default=["A", "B", "C", "D"])
    p.add_argument("--deltas-dir", default=str(DELTAS_DIR))
    p.add_argument("--out-dir", default=str(RAW_DIR))
    p.add_argument("--shard-dir", default=str(SHARD_DIR))
    p.add_argument("--eval-set", default=None)
    p.add_argument("--benchmark-sha256", default=None)
    p.add_argument("--split-manifest", default="logs/direction_split_manifest.json")
    # v2_pipeline.resolve_batch_size accepts an int or the literal "auto"
    # (which reads logs/t4_calibration.json). It calls int(value), so None is
    # not a valid default -- mirror the pipeline's own default instead.
    p.add_argument("--gen-batch", default=DEFAULT_GEN_BATCH)
    p.add_argument("--act-batch", default=DEFAULT_ACT_BATCH)
    p.add_argument("--deadline-minutes", type=float, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--allow-stage2",
        action="store_true",
        help="Permit a condition gated beyond Stage 1. Off by default so a "
             "stray flag cannot start unapproved work.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.coef is not None and args.coef not in COEFFICIENTS:
        print(f"warning: coef {args.coef} is outside the frozen set {COEFFICIENTS}")
    run_unit(args)


if __name__ == "__main__":
    main()
