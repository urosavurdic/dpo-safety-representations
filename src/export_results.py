"""
Packages results/ into a clean, categorized, manifested folder for moving
between local machine, GitHub, Colab, and Drive - without restructuring the
actual repository (results/'s real layout, and everything that reads/writes
it, stays exactly as-is; this only reorganizes a COPY on the way out).

Three tiers, so a report-sized export doesn't accidentally balloon:
  - essential:  every small JSON result (behavioral summaries, probe
                results, refusal-direction geometry, causal/steering
                summaries, bootstrap/McNemar outputs, manifests). Always
                included.
  - direction_vectors: results/refusal_direction/*.npy - small (29 layers x
                hidden_dim floats per stage, KB-scale), not the same thing
                as raw per-prompt activations. Included by default.
  - raw_activations: results/activations/*.npy - the full per-prompt pooled
                activation arrays. These are gitignored (never committed -
                see .gitignore) and can be large; only included if present
                locally AND --include-raw-activations is passed.

Usage:
    python -m src.export_results                          # essential + direction vectors
    python -m src.export_results --include-raw-activations # + raw activations if present locally
    python -m src.export_results --output my_export_folder
"""
import argparse
import fnmatch
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path("results")

# (destination subfolder, fnmatch pattern relative to results/). fnmatch's
# `*` matches across path separators too (unlike glob's), so "behavioral_eval/*.json"
# already covers files nested arbitrarily deep under behavioral_eval/, not
# just direct children - no "**" needed (pathlib.Path.match() does NOT
# support "**" recursive matching before Python 3.13's full_match(), which
# is why this uses fnmatch instead - caught by test_collect_functions_
# against_a_fake_results_tree). Order matters - first matching pattern wins,
# so more specific patterns are listed first.
EXPORT_CATEGORIES = [
    ("behavioral", "behavioral_eval/*.json"),
    ("probes", "probes/*.json"),
    ("activations_metadata", "activations/*.json"),  # small per-prompt metadata, NOT the *.npy arrays
    ("robustness", "raw/causal_ablation_raw_m1d*.json"),  # alt/direct-control causal files - must
    ("robustness", "raw/causal_ablation_raw_*alt*.json"),  # come BEFORE the general causal pattern below
    ("causal", "raw/causal_ablation_raw*.json"),
    ("causal", "summaries/causal_ablation*.json"),
    ("causal", "summaries/bootstrap_ci_causal_ablation*.json"),
    ("steering", "raw/steering_raw*.json"),
    ("steering", "raw/steering_v2_*.json"),
    ("steering", "summaries/steering*.json"),
    ("interpretability", "interpretability/*.json"),
    ("refusal_direction", "refusal_direction/*.json"),
    ("summaries", "summaries/*.json"),  # catch-all for anything else under summaries/
    ("manifests", "manifests/*.json"),
]

DIRECTION_VECTOR_PATTERN = "refusal_direction/*.npy"
RAW_ACTIVATION_PATTERN = "activations/*.npy"


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def file_checksum(path, chunk_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def categorize(rel_path: Path):
    """Returns the destination category subfolder for a results/-relative
    path, or None if it matched no category (skipped, not silently
    miscategorized as a catch-all - see main()'s reporting of skipped files).
    Uses fnmatch (not Path.match(), which doesn't support "**" recursive
    matching before Python 3.13) - fnmatch's "*" already matches across "/"
    so plain "dir/*.json" patterns cover arbitrarily nested files too.

    Bare top-level files (results/foo.json, no subdirectory) fall back to
    "diagnostics" - handled as an explicit special case, NOT a "*.json"
    pattern in EXPORT_CATEGORIES, because fnmatch's cross-"/" "*" would make
    that a silent catch-all for EVERYTHING (defeating the point of returning
    None for genuinely uncategorized nested paths)."""
    posix = rel_path.as_posix()
    for category, pattern in EXPORT_CATEGORIES:
        if fnmatch.fnmatch(posix, pattern):
            return category
    if len(rel_path.parts) == 1:  # no subdirectory -> bare top-level file
        return "diagnostics"
    return None


def collect_essential_files():
    """All JSON files under results/, categorized, deduplicated (a file
    could in principle match category iteration order only once since we
    return on first match in categorize())."""
    files = []
    for path in sorted(RESULTS_DIR.glob("**/*.json")):
        rel = path.relative_to(RESULTS_DIR)
        category = categorize(rel)
        if category is not None:
            files.append((path, rel, category))
    return files


def collect_direction_vectors():
    return sorted(RESULTS_DIR.glob(DIRECTION_VECTOR_PATTERN))


def collect_raw_activations():
    return sorted(RESULTS_DIR.glob(RAW_ACTIVATION_PATTERN))


def build_manifest(essential_files, direction_vectors, raw_activations, output_dir):
    entries = []
    for src_path, rel, category in essential_files:
        entries.append({
            "source": str(rel), "category": category, "size_bytes": src_path.stat().st_size,
            "sha256": file_checksum(src_path),
        })
    for src_path in direction_vectors:
        entries.append({
            "source": str(src_path.relative_to(RESULTS_DIR)), "category": "refusal_direction_vectors",
            "size_bytes": src_path.stat().st_size, "sha256": file_checksum(src_path),
        })
    for src_path in raw_activations:
        entries.append({
            "source": str(src_path.relative_to(RESULTS_DIR)), "category": "raw_activations",
            "size_bytes": src_path.stat().st_size, "sha256": file_checksum(src_path),
        })

    return {
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_files": len(entries),
        "total_size_bytes": sum(e["size_bytes"] for e in entries),
        "includes_raw_activations": len(raw_activations) > 0,
        "files": entries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_export", help="Destination folder (default: results_export)")
    parser.add_argument("--include-raw-activations", action="store_true",
                         help="Also copy results/activations/*.npy if present locally (large, "
                              "gitignored, not needed to reproduce reported numbers - only needed "
                              "for re-deriving directions/probes from scratch)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing --output folder")
    args = parser.parse_args()

    output_dir = Path(args.output)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    essential_files = collect_essential_files()
    direction_vectors = collect_direction_vectors()
    raw_activations = collect_raw_activations() if args.include_raw_activations else []

    skipped = [p for p in RESULTS_DIR.rglob("*.json") if categorize(p.relative_to(RESULTS_DIR)) is None]

    print(f"Essential JSON files: {len(essential_files)} (skipped, uncategorized: {len(skipped)})")
    for src_path, rel, category in essential_files:
        dest = output_dir / category / rel.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest)

    print(f"Refusal-direction vectors: {len(direction_vectors)}")
    for src_path in direction_vectors:
        dest = output_dir / "refusal_direction" / src_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest)

    if args.include_raw_activations:
        print(f"Raw activations (--include-raw-activations): {len(raw_activations)}")
        for src_path in raw_activations:
            dest = output_dir / "raw_activations" / src_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest)
    else:
        n_available = len(collect_raw_activations())
        if n_available:
            print(f"Raw activations: {n_available} present locally but NOT included "
                  "(pass --include-raw-activations to include them)")

    manifest = build_manifest(essential_files, direction_vectors, raw_activations, output_dir)
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if skipped:
        print(f"\nWARNING: {len(skipped)} JSON file(s) under results/ matched no export category "
              "(not copied, not in manifest) - EXPORT_CATEGORIES in src/export_results.py may need "
              "a pattern for these:")
        for p in skipped:
            print(f"  {p}")

    print(f"\nExported {manifest['total_files']} files ({manifest['total_size_bytes'] / 1024:.1f} KB) to {output_dir}/")
    print(f"Manifest: {output_dir}/manifest.json")


if __name__ == "__main__":
    main()
