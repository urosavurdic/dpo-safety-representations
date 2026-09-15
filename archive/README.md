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

## What is deliberately NOT here

`results/_legacy_370era/` stays under `results/`. It holds the only tracked copy
of the pre-freeze 370-row artifacts, including the two deprecated steering runs
that document why single-layer steering replaced multi-layer steering. Those are
evidence, not dead weight, and the directory name already says what it is.
