"""
Run from the repo root with the project's venv active:

    python diagnose_c_b_repro.py

Writes a full, untruncated field-by-field diff to c_b_repro_diff.txt and
also prints a short summary to stdout. Does not modify anything.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import src.analysis.c_b_paired_delta_analysis as c_b  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent


def diff(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a.keys()) | set(b.keys())
        for k in sorted(keys, key=str):
            diff(a.get(k, "<MISSING>"), b.get(k, "<MISSING>"), f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"LEN {len(a)}", f"LEN {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    else:
        if a != b:
            out.append((path, a, b))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_md = str(Path(tmp) / "c_b_rerun.md")
        tmp_json = str(Path(tmp) / "c_b_rerun.json")
        argv = [
            "--review-csv", "data/review/c_review_queue.csv",
            "--benchmark-latest", "data/frozen_v2/LATEST_BENCHMARK.json",
            "--gate-config", "logs/benchmark_gate_config.json",
            "--formatting-config-source", "logs/3d_b_lexical_outlierness_pilot.json",
            "--bootstrap-seed", "20260901", "--n-bootstrap", "10000",
            "--permutation-seed", "20260902", "--n-permutations", "100000",
            "--out-md", tmp_md,
            "--out-json", tmp_json,
        ]
        analysis = c_b.main(argv)

    committed = json.loads((REPO_ROOT / c_b.DEFAULT_OUT_JSON).read_text(encoding="utf-8"))
    rerun = json.loads(json.dumps(analysis))

    out = []
    diff(committed, rerun, "", out)

    with open(REPO_ROOT / "c_b_repro_diff.txt", "w", encoding="utf-8") as f:
        f.write(f"software_versions (committed): {committed.get('software_versions')}\n")
        f.write(f"software_versions (rerun):     {rerun.get('software_versions')}\n")
        f.write(f"total differing leaf fields: {len(out)}\n\n")
        for path, cv, rv in out:
            f.write(f"{path}\n  committed={cv!r}\n  rerun    ={rv!r}\n\n")

    # Bucket the differences by whether the path touches a
    # tfidf_logreg-derived feature vs. everything else, and by
    # population, so the LogisticRegression/solver hypothesis is easy
    # to confirm or rule out at a glance.
    tfidf_related = [p for p, *_ in out if "tfidf_logreg" in p or "cue_tfidf" in p]
    non_tfidf = [p for p, *_ in out if p not in tfidf_related and "software_versions" not in p]
    by_population = {}
    for p, *_ in out:
        for pop in (
            "population_1_all_valid_accepted_pairs",
            "population_2_assistance_type_preserved_yes",
            "population_3_assistance_type_preserved_partial",
        ):
            if f"/{pop}/" in p or p.endswith(f"/{pop}"):
                by_population.setdefault(pop, 0)
                by_population[pop] += 1

    print(f"software_versions committed: {committed.get('software_versions')}")
    print(f"software_versions rerun:     {rerun.get('software_versions')}")
    print(f"total differing leaf fields: {len(out)}")
    print(f"differing fields touching tfidf_logreg/cue_tfidf: {len(tfidf_related)}")
    print(f"differing fields NOT touching tfidf_logreg (excl. software_versions): {len(non_tfidf)}")
    print(f"differing fields by population: {by_population}")
    print("Full detail written to c_b_repro_diff.txt")
    if non_tfidf:
        print("\nFirst 5 non-tfidf differing fields (most informative if solver isn't the cause):")
        for p, *_ in out:
            if p in non_tfidf[:5]:
                print(" ", p)


if __name__ == "__main__":
    main()
