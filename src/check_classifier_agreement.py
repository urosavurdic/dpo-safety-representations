import json


def main():
    with open("results/classifier_validation_sample.json", "r", encoding="utf-8") as f:
        sample = json.load(f)

    labeled = [r for r in sample if r["human_label"] is not None]
    if not labeled:
        print("No entries have a human_label filled in yet. Edit the file first.")
        return

    agree = sum(1 for r in labeled if r["refused"] == r["human_label"])
    print(f"Agreement: {agree}/{len(labeled)} = {100*agree/len(labeled):.1f}%")

    disagreements = [r for r in labeled if r["refused"] != r["human_label"]]
    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s):")
        for r in disagreements:
            print(f"\n  stage={r['stage']} classifier={r['refused']} human={r['human_label']}")
            print(f"  completion: {r['completion'][:200]}")


if __name__ == "__main__":
    main()