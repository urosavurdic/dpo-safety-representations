from src.analysis.mcnemar_causal_ablation import contingency_table


def test_contingency_table_counts_correctly():
    pairs = [(True, True), (True, False), (True, False), (False, True), (False, False), (False, False)]
    assert contingency_table(pairs) == [[1, 2], [1, 2]]