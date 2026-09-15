# archive/

Code that is no longer part of the live pipeline, kept because it produced
results that are still cited, documents a methodology decision, or is simply the
record of how the project got here.

Nothing in this directory is imported by `src/`, exercised by `tests/`, or
required to reproduce any current number. It is not maintained. If you are
looking for how something works today, you are in the wrong directory.

| Entry | What it was | Why it is here |
|---|---|---|
| `colab_migration/apply_colab_fixes.py` | One-shot batch migration script that rewrote five source files during the move to the frozen benchmark. | Fail-closes unless `HEAD` equals a specific commit that is now far behind, so it can never run again. It also carries verbatim frozen *copies* of five live source files as string literals, which made every shared helper look duplicated. |
| `colab_migration/diagnose_c_b_repro.py`, `colab_migration/c_b_repro_diff.txt` | A one-off debugging script and its captured output, used to chase a reproduction mismatch in the quadrant-C paired delta analysis. | The mismatch was resolved. The script has no callers. |
| `quadrant_c_arm2/` | A second, parallel arm of quadrant-C construction ("Arm 2"), with its own `src/` and `tests/` tree. | The arm was formally deferred and never fed the benchmark. It sat inside the repo as a second source root, which made the package layout ambiguous. Internal imports were repointed from `legacy.quadrant_c_arm2.*` to `archive.quadrant_c_arm2.*` when it moved here. |
| `diagnostics/` (10 modules) | One-off diagnostic scripts: a full-finetune smoke test, quadrant-C inspection, two source/leakage searches, a cross-stage diff check, a 370-era activation verifier, a probe-layer sweep, a qualitative spot-check, and a two-script refusal-classifier validation workflow. | None is imported by anything, referenced by any test, or the producer of a committed artifact. `verify_activations.py` additionally hardcodes `EXPECTED_N_PROMPTS = 370` and would assert-fail against every current artifact; it also collided by basename with the live `src/analysis/verify_activations.py`. |
| `crossbranch/build_manual_audit.py` | Built a blinded manual-audit packet for the Stage-2 gate quadrant. | Zero references anywhere, and it writes only into `private/`, so no committed artifact depends on it. |
| `370era/eval_causal_ablation.py` (+ its test) | The standalone causal-ablation runner from the 370-row era. | Superseded by the benchmark-bound pipeline, and already refused to run without `--allow-legacy`. Its generation and hook helpers were extracted to `src/common/generation.py` **first**, verified by moving its own helper tests across unchanged; three live modules still import them. It produced `results/_legacy_370era/raw/causal_ablation_raw_{narrow,wide}.json`, which are themselves quarantined under that directory. |

## Modules the brief proposed archiving that were kept instead

Each of these is the sole producer of a **committed** artifact. Archiving them
would have made published data unreproducible from `src/`, so they stay:

| Module | Produces |
|---|---|
| `src/data_pipeline/lexical_outlierness_pilot.py` | `logs/3d_b_lexical_outlierness_pilot.json`, which is **SHA-pinned** in two `PINNED_INPUT_HASHES` tables. It was proposed for archival as "superseded by `lexical_outlierness.py`", but that module has no `main()` and no entry point — it is the library, and the pilot is the runner that imports it. |
| `src/data_pipeline/build_c2_ahb.py` | `data/quadrant_c_pipeline/secondary_c2_stylistic.jsonl` (tracked) |
| `src/interpretability/lora_subspace_check.py` | `results/interpretability/lora_subspace_check.json` (tracked), and it has a live test |
| `src/analysis/r_authored_distribution.py` | `results/c_construction_audit/r_authored_distribution.json` (tracked) |

## What is deliberately NOT here

`results/_legacy_370era/` stays under `results/`. It holds the only tracked copy
of the pre-freeze 370-row artifacts, including the two deprecated steering runs
that document why single-layer steering replaced multi-layer steering. Those are
evidence, not dead weight, and the directory name already says what it is.
