"""
Check whether Alpaca's own training data contains the instruction-template
leakage pattern we're seeing in generation (e.g. "#Instruction:", "#Output:",
"<noinput>" appearing inside the *output* field itself, not just the
instruction field where it belongs).
"""
from datasets import load_dataset

PATTERNS = ["#instruction:", "#output:", "#input:", "<noinput>"]


def contains_pattern(text):
    text_lower = text.lower()
    return any(p in text_lower for p in PATTERNS)


def main():
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    matches = [row for row in dataset if contains_pattern(row["output"])]
    print(f"Alpaca rows where 'output' field itself contains template leakage: {len(matches)} / {len(dataset)}")
    for row in matches[:5]:
        print(f"\n  instruction: {row['instruction'][:100]}")
        print(f"  output: {row['output'][:200]}")


if __name__ == "__main__":
    main()