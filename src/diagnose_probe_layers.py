"""
Throwaway diagnostic: prints CV accuracy at EVERY layer (not just argmax
"best layer") for M0 and M3, to check whether many layers tie near 1.0
(explains identical "best layer" output despite genuinely different
underlying activations) vs. something stranger.
"""
from src.eval_probes import load_stage_activations, split_by_quadrant, split_b_train_holdout, probe_layer

for stage in ["M0", "M1", "M2", "M3"]:
    arr, meta = load_stage_activations(stage)
    bq = split_by_quadrant(arr, meta)
    b_train, b_holdout = split_b_train_holdout(bq["B"])
    print(f"\n=== {stage} ===")
    for i in range(arr.shape[1]):
        r = probe_layer(bq["A"], b_train, b_holdout, bq["C"], bq["D"], i)
        print(f"  layer {i:2d}: acc={r['cv_accuracy_mean']:.3f}  D->unsafe={r['quadrant_d_flagged_unsafe_frac']:.3f}")