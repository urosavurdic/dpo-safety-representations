"""
Complete near-duplicate contamination check for C2, C3, C4.

Place this file at:
    src/diagnostics/complete_neardup_check.py

Run from repo root as:
    python -m src.diagnostics.complete_neardup_check

Requires:
    pip install sentence-transformers

v3: fixes a major performance bug in v2 - training-file embeddings are now
computed ONCE and reused across every record/file, instead of being
recomputed from scratch for every single record (which made a 36-record
file take ~144 full passes over a 20k-prompt training corpus).
"""

import json
import shutil
import time
from pathlib import Path

REPO = Path(__file__).parent.parent.parent

FILES = {
    "C2": {
        "path": REPO / "data/quadrant_c_pipeline/secondary_c2_stylistic.jsonl",
        "text_extractor": lambda r: [t for t in [
            r.get("source_prompt"), r.get("original_prompt")
        ] if t],
    },
    "C3": {
        "path": REPO / "data/quadrant_c_pipeline/secondary_c3_contextual.jsonl",
        "text_extractor": lambda r: [r["source_prompt"]],
    },
    "C4": {
        "path": REPO / "data/quadrant_c_pipeline/secondary_c4_dual_use.jsonl",
        "text_extractor": lambda r: list(r["prompt_variants"].values()),
    },
}

TRAINING_FILES = {
    "sft_helpful.jsonl":     REPO / "data/processed/sft_helpful.jsonl",
    "sft_helpful_alt.jsonl": REPO / "data/processed/sft_helpful_alt.jsonl",
    "sft_safety.jsonl":      REPO / "data/processed/sft_safety.jsonl",
    "dpo_pairs.jsonl":       REPO / "data/processed/dpo_pairs.jsonl",
}

THRESHOLD = 0.9
MODEL_NAME = "all-MiniLM-L6-v2"
PLACEHOLDERS = {
    "unknown_run_complete_neardup_check",
    "unknown_near_dup_check_requires_HF_model_download",
}


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def load_and_embed_training_files(model):
    """Load each training file's prompts and embed them ONCE. Returns
    {label: {"prompts": [...], "embeddings": ndarray, "normalized_set": {...}}}
    or {label: None} if the file is missing."""
    result = {}
    for label, path in TRAINING_FILES.items():
        if not path.exists():
            print(f"  [skip] {label} not found")
            result[label] = None
            continue
        rows = load_jsonl(path)
        prompts = [r["prompt"] for r in rows if "prompt" in r]
        print(f"  Embedding {label}: {len(prompts)} prompts...", end=" ", flush=True)
        t0 = time.time()
        embeddings = model.encode(
            prompts, normalize_embeddings=True, show_progress_bar=False, batch_size=256
        )
        print(f"done ({time.time() - t0:.1f}s)")
        result[label] = {
            "prompts": prompts,
            "embeddings": embeddings,
            "normalized_set": {" ".join(p.split()).lower() for p in prompts},
        }
    return result


def check_eval_texts_against_training(eval_texts, model, train_data):
    """eval_texts: list of strings (already deduped by caller if desired).
    Returns {label: {"exact": [...], "near": [...]}} for each training file."""
    results = {}
    if not eval_texts:
        return results

    eval_emb = model.encode(eval_texts, normalize_embeddings=True, show_progress_bar=False)

    for label, data in train_data.items():
        if data is None:
            results[label] = "unknown"
            continue

        exact = [t for t in eval_texts if " ".join(t.split()).lower() in data["normalized_set"]]

        sim_matrix = eval_emb @ data["embeddings"].T  # (n_eval, n_train)
        near = []
        for i, eval_text in enumerate(eval_texts):
            for j, train_text in enumerate(data["prompts"]):
                sim = float(sim_matrix[i][j])
                if sim >= THRESHOLD:
                    near.append({"eval_prompt": eval_text, "closest_train_prompt": train_text, "similarity": round(sim, 4)})

        results[label] = {"exact": exact, "near": near}
    return results


def process_file(label, path, text_extractor, model, train_data):
    if not path.exists():
        print(f"  [{label}] file not found - skipping (run build script first)")
        return
    records = load_jsonl(path)
    needs_update = any(
        isinstance(v, dict) and v.get("near") in PLACEHOLDERS
        for r in records for v in r.get("training_contamination", {}).values()
    )
    if not needs_update:
        print(f"  [{label}] already complete - skipping")
        return

    print(f"  [{label}] {len(records)} records...")
    shutil.copy(path, path.with_suffix(".jsonl.bak"))

    hits = 0
    updated = []
    t0 = time.time()
    for idx, rec in enumerate(records):
        texts = text_extractor(rec)
        neardup = check_eval_texts_against_training(texts, model, train_data)

        new_contam = {}
        for fl, existing in rec.get("training_contamination", {}).items():
            if isinstance(existing, dict) and existing.get("near") in PLACEHOLDERS:
                real = neardup.get(fl)
                if real is None or real == "unknown":
                    new_contam[fl] = existing
                else:
                    new_contam[fl] = {"exact": len(real["exact"]), "near": len(real["near"]), "near_flagged": real["near"]}
                    if real["exact"] or real["near"]:
                        hits += 1
                        rec["agent_pre_screen"] = "exclude"
                        rec["agent_reason"] = f"near-dup hit in {fl} (exact={len(real['exact'])}, near={len(real['near'])})"
            else:
                new_contam[fl] = existing
        rec["training_contamination"] = new_contam
        updated.append(rec)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(records):
            print(f"    {idx + 1}/{len(records)} records checked ({time.time() - t0:.1f}s elapsed)")

    save_jsonl(path, updated)
    print(f"    Done. {hits} new contamination hits. Backup saved as {path.stem}.jsonl.bak")


def main():
    print("Loading sentence-transformer model (downloads ~80MB on first run)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    print("\nEmbedding training files (once, reused for every record/file below)...")
    train_data = load_and_embed_training_files(model)

    print("\nRunning near-dup checks...")
    for label, cfg in FILES.items():
        process_file(label, cfg["path"], cfg["text_extractor"], model, train_data)

    print("\nDone. Run verification commands next.")


if __name__ == "__main__":
    main()