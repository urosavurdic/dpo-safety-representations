"""
Focused unit tests for src/data_pipeline/lexical_outlierness.py.

Per 3D-B task scope, this deliberately does NOT run the full historical
test suite and does NOT depend on real repository data files - everything
here uses small synthetic corpora, mirroring the convention already
established in tests/test_cue_scoring.py. Covers: preprocessing edge
cases, the S4 grouping Jaccard edge case, S4 union-find grouping on a
synthetic near-duplicate pair, S2/S3 OOV/zero-vector/degenerate-n_i
handling, and the S6 percentile tie rule.
"""
import math

from src.data_pipeline.lexical_outlierness import (
    build_groups,
    calibrated_percentile,
    fold_selfinfo,
    fold_tfidf,
    jaccard,
    normalize_text,
    score_selfinfo,
    score_tfidf,
    shingles5,
    similarity_text,
    tokenize,
    word_ngrams_1_2,
)


def test_normalize_text_url_and_whitespace():
    out = normalize_text("Hello   WORLD!\n\nVisit HTTPS://Example.com/x?y=1 now.")
    assert out == "hello world! visit <url> now."


def test_normalize_text_nfkc():
    # full-width digits normalize to ASCII under NFKC
    out = normalize_text("\uff11\uff12\uff13")
    assert out == "123"


def test_tokenize_apostrophe_not_word_char():
    toks = tokenize(normalize_text("don't stop"))
    assert toks == ["don", "t", "stop"]


def test_word_ngrams_1_2():
    feats = word_ngrams_1_2(["a", "b", "c"])
    assert feats[("a",)] == 1
    assert feats[("a", "b")] == 1
    assert feats[("b", "c")] == 1
    assert ("c", "d") not in feats


def test_shingles5_short_string_empty():
    assert shingles5("abcd") == set()
    assert shingles5("") == set()
    assert shingles5("abcde") == {"abcde"}


def test_jaccard_empty_edge_case():
    # explicit 0.0, never a ZeroDivisionError, for either/both empty
    assert jaccard(set(), set()) == 0.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard(set(), {"a"}) == 0.0


def test_jaccard_normal():
    assert jaccard({"ab", "bc"}, {"bc", "cd"}) == 1 / 3


def test_similarity_text_strips_punctuation_not_letters():
    out = similarity_text("Hello, World!!")
    assert out == "hello world"


def _mk_row(rid, text):
    return {
        "record_id": rid,
        "prompt_text": text,
        "exact_duplicate_canonical_record_id": None,
        "normalized_duplicate_canonical_record_id": None,
    }


def test_build_groups_merges_near_duplicates_and_freezes_singletons():
    rows = [
        _mk_row("A", "how do i pick a lock quickly"),
        _mk_row("B", "how do i pick a lock fast"),  # near-dup of A, J>=0.6 expected
        _mk_row("C", "explain quantum entanglement in simple terms please"),
    ]
    record_to_group, artifact = build_groups(rows)
    assert record_to_group["A"] == record_to_group["B"]
    assert record_to_group["C"] not in (record_to_group["A"],)
    # group_id = lexicographically smallest member
    assert record_to_group["A"] == "A"
    assert artifact["n_groups"] == 2
    assert artifact["threshold"] == 0.6


def test_build_groups_seeds_from_frozen_canonical_fields():
    rows = [
        _mk_row("X", "totally unrelated text about gardening tips"),
        _mk_row("Y", "another totally different sentence on cooking"),
    ]
    rows[1]["exact_duplicate_canonical_record_id"] = "X"
    record_to_group, _ = build_groups(rows)
    assert record_to_group["X"] == record_to_group["Y"] == "X"


def test_tfidf_method_ranks_lexical_outlier_higher():
    # Three near-identical reference docs about "bake bread", one
    # outlier held-out doc sharing no vocabulary; outlier must score
    # strictly higher (more atypical) than a near-identical held-out doc.
    ref_docs = [
        "how do i bake bread at home",
        "how do i bake bread quickly",
        "how do i bake bread today",
    ]
    ref_feats = [word_ngrams_1_2(tokenize(normalize_text(d))) for d in ref_docs]
    idf, mu, _ = fold_tfidf(ref_feats, weights=None)

    similar_feats = word_ngrams_1_2(tokenize(normalize_text("how do i bake bread now")))
    # shares exactly one reference token ("bread") so it stays in-vocab
    # (non-NaN) while still being lexically atypical relative to the
    # "how do i bake bread X" template.
    outlier_feats = word_ngrams_1_2(tokenize(normalize_text("bread giraffes juggle spreadsheets")))

    s_similar = score_tfidf(similar_feats, idf, mu)
    s_outlier = score_tfidf(outlier_feats, idf, mu)
    assert s_outlier is not None and not math.isnan(s_outlier)
    assert math.isnan(s_similar) is False
    assert s_outlier > s_similar


def test_tfidf_zero_vector_is_nan_not_imputed():
    ref_feats = [word_ngrams_1_2(tokenize(normalize_text("apples and oranges")))]
    idf, mu, _ = fold_tfidf(ref_feats, weights=None)
    # holdout doc shares no n-grams with the single-doc reference vocab
    holdout_feats = word_ngrams_1_2(tokenize(normalize_text("zzz qqq wibble")))
    s = score_tfidf(holdout_feats, idf, mu)
    assert math.isnan(s)


def test_selfinfo_oov_token_gets_max_surprisal_not_a_crash():
    ref_tokens = [["a", "b", "a"], ["a", "c"]]
    counts, total, vsize = fold_selfinfo(ref_tokens, weights=None)
    s_known = score_selfinfo(["a"], counts, total, vsize)
    s_oov = score_selfinfo(["zzz_never_seen"], counts, total, vsize)
    assert s_oov > s_known  # OOV token is maximally surprising
    assert not math.isnan(s_oov)


def test_selfinfo_degenerate_empty_tokens_is_nan():
    ref_tokens = [["a", "b"]]
    counts, total, vsize = fold_selfinfo(ref_tokens, weights=None)
    s = score_selfinfo([], counts, total, vsize)
    assert math.isnan(s)


def test_calibrated_percentile_tie_rule():
    # 1 strictly below + 2 ties -> (1 + 0.5*2) / 4 = 0.5
    p = calibrated_percentile(0.3, [0.1, 0.3, 0.3, 0.5])
    assert p == 0.5


def test_calibrated_percentile_nan_score_is_undefined():
    assert calibrated_percentile(float("nan"), [0.1, 0.2]) is None


def test_calibrated_percentile_empty_reference_is_undefined():
    assert calibrated_percentile(0.5, []) is None


def test_calibrated_percentile_weighted_variant():
    # weight 2.0 on the reference point below 0.5, weight 1.0 above threshold example
    p = calibrated_percentile(0.5, [0.2, 0.3], ref_weights=[2.0, 1.0])
    assert p == 1.0  # both weighted references are below 0.5 -> full weight below


def test_source_balancing_weights_sum_to_one_per_source():
    # sanity: reproduces the S8 weight formula directly
    ref_sources = ["StrongREJECT"] * 4 + ["SimpleSafetyTests"] * 2
    from collections import Counter

    counts = Counter(ref_sources)
    weights = [0.5 / counts[s] for s in ref_sources]
    per_source_total = {}
    for s, w in zip(ref_sources, weights):
        per_source_total[s] = per_source_total.get(s, 0.0) + w
    assert abs(per_source_total["StrongREJECT"] - 0.5) < 1e-9
    assert abs(per_source_total["SimpleSafetyTests"] - 0.5) < 1e-9
    assert abs(sum(weights) - 1.0) < 1e-9
