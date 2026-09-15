"""The training stages, and the stage subsets different analyses run over.

Stage names are frozen: they appear in artifact filenames and in committed
result rows, so they are never renamed.

    M0          base model, no fine-tuning
    M1          SFT on a helpful corpus
    M2          SFT on safety data
    M3          DPO, initialised from M2
    M3_direct   DPO applied straight to M1, skipping M2
    *_alt       the parallel branch, whose M1 trained on a different source
                corpus; everything downstream uses the same safety data

Several subsets exist because they answer different questions, and mixing them
has caused real mislabelling before -- notably treating an M3 vs M3_direct
comparison as if it were one step of the sequential trajectory.
"""

__all__ = [
    "ALL_STAGES",
    "SEQUENTIAL_STAGES",
    "ALT_SEQUENTIAL_STAGES",
    "INTERVENTION_STAGES",
    "TRAINED_STAGES",
]

# Every stage that exists. Nine of them.
ALL_STAGES = [
    "M0",
    "M1",
    "M2",
    "M3",
    "M3_direct",
    "M1_alt",
    "M2_alt",
    "M3_alt",
    "M3_direct_alt",
]

# The true sequential trajectory. M3_direct branches from M1 and is a parallel
# control, not a step that follows M3.
SEQUENTIAL_STAGES = ["M0", "M1", "M2", "M3"]

# The alt branch's own trajectory. M0 is shared: the base model is the same.
ALT_SEQUENTIAL_STAGES = ["M0", "M1_alt", "M2_alt", "M3_alt"]

# Causal ablation runs on the four DPO endpoints.
INTERVENTION_STAGES = ["M3", "M3_direct", "M3_alt", "M3_direct_alt"]

# Every stage that received training, i.e. all nine except the base model.
# Steering runs over these.
TRAINED_STAGES = [s for s in ALL_STAGES if s != "M0"]
