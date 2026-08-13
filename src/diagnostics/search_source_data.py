"""
Diagnostic: does PKU-SafeRLHF contain vehicle-theft/hotwiring-specific
content at all, anywhere in the full dataset - not just our sample? The
category-coverage check ruled out biased sampling (our sample tracked the
full pool proportionally, ~45-50%, uniformly across every category). This
checks a more specific hypothesis: maybe this particular skill is just rare
in the source dataset overall, since none of its 19 harm categories map
cleanly onto "circumventing a physical security mechanism."
"""
from datasets import load_dataset

KEYWORDS = [
    "hotwir", "hot-wir", "hot wir", "steal a car", "steal your car",
    "car theft", "start a car without", "bypass the ignition",
]


def contains_keyword(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in KEYWORDS)


def main():
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")

    matches = [row["prompt"] for row in dataset if contains_keyword(row["prompt"])]

    print(f"Prompts matching vehicle-theft keywords in full PKU-SafeRLHF (73,907 rows): {len(matches)}")
    for m in matches[:10]:
        print(f"  - {m}")


if __name__ == "__main__":
    main()