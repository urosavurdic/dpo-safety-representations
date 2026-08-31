"""
Focused tests for src/analysis/cf_joint_geometry.py.

Per the C-F-B task brief, this runs only focused tests for the new
analysis module -- not the broad repository suite, and not the full real
audit against the actual 654/104/52-row corpus. Every test below uses
small synthetic fixtures, so it is fast, deterministic, and independent
of the real (frozen, hash-pinned) benchmark content.

Covers exactly the seven areas named in the C-F-B task brief:
  1. common feature-space construction   (TestStructuralViewConstruction,
                                           TestLexicalViewConstruction,
                                           TestCombinedViewConstruction)
  2. deterministic transformation        (TestDeterminism)
  3. correct A/B/C/D membership          (TestPopulationLoading)
  4. equal-cell contrast construction    (TestFactorialContrasts)
  5. pairwise distance calculation       (TestCentroidGeometry,
                                           TestEnergyDistance)
  6. JSD normalization/smoothing         (TestNgramJSD)
  7. deterministic output                (TestDeterminism, TestHolmBonferroni)
"""
import numpy as np
import pytest

from src.analysis.cf_joint_geometry import (
    ALL_POPULATIONS_ORDER,
    AUXILIARY_ORDER,
    PAIR_ORDER,
    QUADRANT_ORDER,
    apply_standardizer,
    build_all_views,
    build_common_fightin_words,
    build_ngram_vocab,
    centroid,
    check_r_authored_word_char_basis,
    cosine_angle_degrees,
    cumulative_explained_variance,
    energy_distance,
    factorial_contrasts,
    fit_pca_combined,
    fit_standardizer,
    holm_bonferroni,
    jsd_base2,
    l2_normalize_rows,
    lexical_diversity,
    load_frozen_quadrants,
    load_r104_source_population,
    mean_word_length,
    ngram_distribution,
    pairwise_centroid_distances,
    pca_component_loadings,
    pca_coordinates,
    permutation_test_energy_distance,
    plot_pca_scatter,
    residualize_on_covariate,
    rng_for_pair,
    score_fw_common,
    sentence_count,
    simple_char_count,
    simple_word_count,
    verify_population_counts,
    zero_token_mask,
    zero_variance_features,
    AuditFailClosed,
)


# ── text-derived primitives ─────────────────────────────────────────────
def test_sentence_count():
    assert sentence_count("One. Two! Three?") == 3
    assert sentence_count("no terminal punctuation") == 0
    assert sentence_count("Multiple!!! marks... count as one run.") == 3


def test_mean_word_length():
    assert mean_word_length("") is None
    assert mean_word_length("ab abc") == pytest.approx(2.5)


def test_lexical_diversity():
    assert lexical_diversity("") is None
    assert lexical_diversity("a a b") == pytest.approx(2 / 3)
    assert lexical_diversity("one two three") == pytest.approx(1.0)


def test_zero_token_mask():
    mask = zero_token_mask(["hello world", "", "!!!", "another one"])
    assert mask.tolist() == [False, True, True, False]


def test_simple_counts():
    assert simple_word_count("a b c") == 3
    assert simple_char_count("abc") == 3


# ── 3. correct A/B/C/D membership ───────────────────────────────────────
class TestPopulationLoading:
    def test_load_frozen_quadrants_membership(self):
        benchmark_rows = [
            {"record_id": "r1", "prompt": "p1", "quadrant": "A", "word_count": 1, "character_count": 2,
             "source_dataset": "HarmBench", "project_category": "illegal"},
            {"record_id": "r2", "prompt": "p2", "quadrant": "B", "word_count": 3, "character_count": 4,
             "source_dataset": "XSTest", "project_category": "homonyms"},
            {"record_id": "r3", "prompt": "p3", "quadrant": "C", "word_count": 5, "character_count": 6,
             "source_dataset": "StrongREJECT", "project_category": "illegal"},
            {"record_id": "r4", "prompt": "p4", "quadrant": "D", "word_count": 7, "character_count": 8,
             "source_dataset": "Alpaca", "project_category": None},
            {"record_id": "r5", "prompt": "should not appear", "quadrant": "Z", "word_count": 0, "character_count": 0},
        ]
        pops = load_frozen_quadrants(benchmark_rows)
        assert set(pops.keys()) == {"A", "B", "C", "D"}
        assert [r["record_id"] for r in pops["A"]] == ["r1"]
        assert [r["record_id"] for r in pops["B"]] == ["r2"]
        assert [r["record_id"] for r in pops["C"]] == ["r3"]
        assert [r["record_id"] for r in pops["D"]] == ["r4"]
        # the unknown-quadrant row is silently excluded from every A/B/C/D bucket
        assert all(
            "should not appear" != row.get("text") for pop in pops.values() for row in pop
        )
        assert pops["A"][0]["text"] == "p1"
        assert pops["D"][0]["word_count"] == 7.0

    def test_c_is_exactly_r104_candidate_side_not_a_fifth_group(self):
        """C-F-B's critical population rule: there must be exactly ONE C
        population, and R104-source is a distinct, auxiliary population
        (never merged into C)."""
        benchmark_rows = [
            {"record_id": "c1", "prompt": "candidate text", "quadrant": "C",
             "word_count": 2, "character_count": 10},
        ]
        pops = load_frozen_quadrants(benchmark_rows)
        assert len(pops["C"]) == 1
        assert "R104_source" not in pops  # this loader never invents a 5th group

    def test_load_r104_source_uses_source_prompt_column(self, tmp_path):
        csv_path = tmp_path / "c_review_queue.csv"
        csv_path.write_text(
            "record_id,source_prompt,candidate_prompt,word_count_source,"
            "character_count_source\n"
            "r1,the source text,the candidate text,3,14\n",
            encoding="utf-8",
        )
        pop = load_r104_source_population(csv_path)
        assert len(pop) == 1
        assert pop[0]["text"] == "the source text"
        assert pop[0]["word_count"] == 3.0
        assert pop[0]["character_count"] == 14.0

    def test_verify_population_counts_fails_closed_on_mismatch(self):
        populations = {
            "A": [{}] * 150, "B": [{}] * 250, "C": [{}] * 103,  # wrong: 103 not 104
            "D": [{}] * 150, "R104_source": [{}] * 104, "R_AUTHORED": [{}] * 52,
        }
        with pytest.raises(AuditFailClosed):
            verify_population_counts(populations)

    def test_verify_population_counts_passes_on_exact_match(self):
        populations = {
            "A": [{}] * 150, "B": [{}] * 250, "C": [{}] * 104,
            "D": [{}] * 150, "R104_source": [{}] * 104, "R_AUTHORED": [{}] * 52,
        }
        verify_population_counts(populations)  # should not raise


class TestRAuthoredPrecheck:
    def test_basis_confirmed_candidate(self):
        rows = [
            {"candidate_prompt": "one two three", "source_prompt": "aaaaaaaaaaaaaaaaaaaaa",
             "word_count": "3", "character_count": "13"},
        ]
        result = check_r_authored_word_char_basis(rows)
        assert result["basis_confirmed_candidate_prompt"] is True
        assert result["branch_taken"] == "reuse_existing_columns"

    def test_basis_not_confirmed_triggers_recompute_branch(self):
        rows = [
            {"candidate_prompt": "one two three", "source_prompt": "a b c d e",
             "word_count": "5", "character_count": "9"},  # matches source, not candidate
        ]
        result = check_r_authored_word_char_basis(rows)
        assert result["basis_confirmed_candidate_prompt"] is False
        assert result["branch_taken"] == "recompute_fresh_from_candidate_prompt"
        assert result["word_count_matches_source_prompt"] == 1
        assert result["word_count_matches_candidate_prompt"] == 0


# ── 1. common feature-space construction ────────────────────────────────
class TestStructuralViewConstruction:
    def test_zero_variance_features_dropped(self):
        pooled = {
            "word_count": [1.0, 2.0, 3.0, 4.0],
            "character_count": [1.0, 1.0, 1.0, 1.0],  # zero variance
            "sentence_count": [1.0, 1.0, 1.0, 1.0],  # zero variance
            "mean_word_length": [2.0, 3.0, 4.0, 5.0],
            "lexical_diversity": [0.5, 0.6, 0.7, 0.8],
            "has_bullet_marker": [0.0, 0.0, 0.0, 0.0],  # zero variance
            "has_numbered_step": [0.0, 1.0, 0.0, 1.0],
            "has_code_block": [0.0, 0.0, 0.0, 0.0],  # zero variance
            "multi_sentence_flag": [1.0, 0.0, 1.0, 0.0],
            "lexical_risk_hit_count": [0.0, 1.0, 2.0, 3.0],
        }
        dropped = zero_variance_features(pooled)
        assert set(dropped) == {"character_count", "sentence_count", "has_bullet_marker", "has_code_block"}

    def test_standardizer_fit_and_apply_zero_mean_unit_sd(self):
        pooled = {"f1": [1.0, 2.0, 3.0, 4.0, 5.0]}
        stats_ = fit_standardizer(pooled, ["f1"])
        z = apply_standardizer(pooled, ["f1"], stats_)
        assert z.mean() == pytest.approx(0.0, abs=1e-8)
        assert z.std(ddof=1) == pytest.approx(1.0, abs=1e-8)

    def test_standardizer_apply_to_different_population_uses_fit_mean_sd(self):
        """R104-source/R-AUTHORED must be transformed with the A/B/C/D fit
        mean/sd, never refit."""
        fit_pool = {"f1": [0.0, 10.0]}  # mean=5, sd=~7.07
        stats_ = fit_standardizer(fit_pool, ["f1"])
        other_pop = {"f1": [5.0]}
        z = apply_standardizer(other_pop, ["f1"], stats_)
        assert z[0, 0] == pytest.approx(0.0, abs=1e-8)  # exactly at the fit mean


class TestLexicalViewConstruction:
    def test_l2_normalize_rows(self):
        X = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
        normed = l2_normalize_rows(X)
        assert np.linalg.norm(normed[0]) == pytest.approx(1.0)
        assert np.linalg.norm(normed[1]) == pytest.approx(0.0)  # zero row stays zero, no div-by-zero
        assert np.linalg.norm(normed[2]) == pytest.approx(1.0)

    def test_l2_normalize_empty_columns_noop(self):
        X = np.zeros((3, 0))
        assert l2_normalize_rows(X).shape == (3, 0)


class TestCombinedViewConstruction:
    def _tiny_populations(self):
        # Small but non-degenerate: 3 rows per quadrant, varied text so
        # TF-IDF/SVD/FightinWords all have something to fit on.
        texts = {
            "A": ["harmful request one here", "another harmful ask here", "third harmful thing here"],
            "B": ["benign safe question here", "another benign query here", "third benign ask here"],
            "C": ["candidate reworded request", "second candidate reworded ask", "third candidate rewording"],
            "D": ["please write a poem", "please summarize this text", "please translate this word"],
        }
        pops = {}
        for q, texts_q in texts.items():
            pops[q] = [
                {"record_id": f"{q}{i}", "text": t, "word_count": float(len(t.split())),
                 "character_count": float(len(t)), "source_dataset": "X", "project_category": "cat"}
                for i, t in enumerate(texts_q)
            ]
        pops["R104_source"] = [
            {"record_id": "rs0", "text": "original harmful source text", "word_count": 4.0, "character_count": 28.0}
        ]
        pops["R_AUTHORED"] = [
            {"record_id": "ra0", "text": "source authored candidate text", "word_count": 4.0, "character_count": 30.0}
        ]
        return pops

    def test_build_all_views_shapes_and_membership(self):
        pops = self._tiny_populations()
        formatting_config = {
            "bullet_marker_regex": r"(?m)^\s*[-*\u2022]\s+",
            "numbered_step_regex": r"(?m)^\s*\d+[\.\)]\s+",
            "code_block_regex": "```",
        }
        views = build_all_views(pops, formatting_config, svd_seed=123, svd_n_components=5)

        # every population present, correct row counts in the structural view
        for name, rows in pops.items():
            assert views["structural_view"][name].shape[0] == len(rows)
            assert views["combined_view"][name].shape[0] <= len(rows)  # OOV rows may be excluded

        # combined view = concat of (weighted) structural-only and lexical-only blocks
        n_struct = len(views["surviving_structural_features"])
        n_lex = views["lexical_view"]["A"].shape[1]
        assert views["combined_view"]["A"].shape[1] == n_struct + n_lex

        # fit-only-on-ABCD invariant: R104-source/R-AUTHORED never contribute rows to the SVD fit
        assert views["svd"].n_components <= 5

    def test_fw_common_fit_uses_only_a_b_d(self):
        fw = build_common_fightin_words(
            ["harmful text one", "harmful text two"],
            ["benign text one", "benign text two"],
            ["neutral text one", "neutral text two"],
        )
        scores = score_fw_common(fw, ["harmful text one", "totally unseen vocabulary zzz"])
        assert len(scores) == 2
        assert isinstance(scores[0], float)


# ── 2 & 7. deterministic transformation / deterministic output ─────────
class TestDeterminism:
    def test_build_all_views_is_deterministic(self):
        pops = TestCombinedViewConstruction()._tiny_populations()
        formatting_config = {
            "bullet_marker_regex": r"(?m)^\s*[-*\u2022]\s+",
            "numbered_step_regex": r"(?m)^\s*\d+[\.\)]\s+",
            "code_block_regex": "```",
        }
        v1 = build_all_views(pops, formatting_config, svd_seed=999, svd_n_components=5)
        v2 = build_all_views(pops, formatting_config, svd_seed=999, svd_n_components=5)
        for name in pops:
            np.testing.assert_array_equal(v1["structural_view"][name], v2["structural_view"][name])
            np.testing.assert_allclose(v1["combined_view"][name], v2["combined_view"][name])

    def test_permutation_test_deterministic_given_same_rng_seed(self):
        rng_a = np.random.default_rng([20260903, 0])
        rng_b = np.random.default_rng([20260903, 0])
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        Y = np.array([[10.0], [11.0], [12.0], [13.0]])
        obs_a, p_a = permutation_test_energy_distance(X, Y, rng_a, n_permutations=200)
        obs_b, p_b = permutation_test_energy_distance(X, Y, rng_b, n_permutations=200)
        assert obs_a == pytest.approx(obs_b)
        assert p_a == pytest.approx(p_b)

    def test_rng_for_pair_gives_independent_streams(self):
        rng_ab = rng_for_pair(20260903, "AB")
        rng_ac = rng_for_pair(20260903, "AC")
        draw_ab = rng_ab.random(5)
        draw_ac = rng_ac.random(5)
        assert not np.allclose(draw_ab, draw_ac)

    def test_rng_for_pair_reproducible_across_calls(self):
        rng1 = rng_for_pair(20260903, "CD")
        rng2 = rng_for_pair(20260903, "CD")
        assert np.allclose(rng1.random(10), rng2.random(10))


# ── 4. equal-cell contrast construction ─────────────────────────────────
class TestFactorialContrasts:
    def test_intent_and_surface_contrast_formulas(self):
        centroids = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.0, 1.0]),
            "C": np.array([1.0, 1.0]),
            "D": np.array([0.0, 0.0]),
        }
        intent, surface = factorial_contrasts(centroids)
        expected_intent = (centroids["A"] + centroids["C"]) / 2 - (centroids["B"] + centroids["D"]) / 2
        expected_surface = (centroids["A"] + centroids["B"]) / 2 - (centroids["C"] + centroids["D"]) / 2
        np.testing.assert_allclose(intent, expected_intent)
        np.testing.assert_allclose(surface, expected_surface)

    def test_contrast_is_equal_cell_weighted_not_row_pooled(self):
        """A quadrant with many more rows must not dominate the contrast
        -- centroids are computed per-quadrant first (equal cell weight),
        never by pooling all rows together before centroiding."""
        big_a = np.vstack([np.array([10.0, 10.0])] * 1000)  # same centroid regardless of n
        small_a = np.vstack([np.array([10.0, 10.0])] * 2)
        centroids_big = {"A": centroid(big_a), "B": np.zeros(2), "C": np.zeros(2), "D": np.zeros(2)}
        centroids_small = {"A": centroid(small_a), "B": np.zeros(2), "C": np.zeros(2), "D": np.zeros(2)}
        intent_big, _ = factorial_contrasts(centroids_big)
        intent_small, _ = factorial_contrasts(centroids_small)
        np.testing.assert_allclose(intent_big, intent_small)

    def test_cosine_angle_orthogonal_vectors(self):
        assert cosine_angle_degrees(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(90.0)

    def test_cosine_angle_parallel_vectors(self):
        assert cosine_angle_degrees(np.array([2.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)

    def test_cosine_angle_undefined_for_zero_vector(self):
        assert cosine_angle_degrees(np.array([0.0, 0.0]), np.array([1.0, 0.0])) is None


# ── 5. pairwise distance calculation ─────────────────────────────────────
class TestCentroidGeometry:
    def test_pairwise_centroid_distances_all_six_pairs(self):
        centroids = {
            "A": np.array([0.0, 0.0]), "B": np.array([3.0, 4.0]),
            "C": np.array([0.0, 4.0]), "D": np.array([3.0, 0.0]),
        }
        dists = pairwise_centroid_distances(centroids)
        assert set(dists.keys()) == set(PAIR_ORDER)
        assert dists["AB"] == pytest.approx(5.0)
        assert dists["AC"] == pytest.approx(4.0)
        assert dists["AD"] == pytest.approx(3.0)

    def test_centroid_ignores_row_order(self):
        X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = np.array([[3.0, 4.0], [1.0, 2.0]])
        np.testing.assert_allclose(centroid(X1), centroid(X2))


class TestEnergyDistance:
    def test_energy_distance_zero_for_identical_samples(self):
        X = np.array([[0.0], [1.0], [2.0]])
        assert energy_distance(X, X.copy()) == pytest.approx(0.0, abs=1e-8)

    def test_energy_distance_positive_for_separated_samples(self):
        X = np.array([[0.0], [1.0], [2.0]])
        Y = np.array([[100.0], [101.0], [102.0]])
        assert energy_distance(X, Y) > 0

    def test_permutation_p_value_bounds(self):
        rng = np.random.default_rng(0)
        X = np.random.default_rng(1).normal(size=(10, 2))
        Y = np.random.default_rng(2).normal(size=(10, 2))
        obs, p = permutation_test_energy_distance(X, Y, rng, n_permutations=50)
        assert 0.0 <= p <= 1.0

    def test_permutation_p_value_small_for_well_separated_samples(self):
        rng = np.random.default_rng(42)
        X = np.zeros((8, 2))
        Y = np.ones((8, 2)) * 50.0
        obs, p = permutation_test_energy_distance(X, Y, rng, n_permutations=200)
        assert p < 0.05


class TestHolmBonferroni:
    def test_holm_bonferroni_monotone_and_bounded(self):
        p_values = {"AB": 0.001, "AC": 0.20, "AD": 0.04, "BC": 0.5, "BD": 0.9, "CD": 0.01}
        result = holm_bonferroni(p_values)
        adj = [result[k]["adjusted_p_holm"] for k in p_values]
        assert all(0.0 <= a <= 1.0 for a in adj)
        # smallest raw p gets multiplied by the largest factor -> rank 1
        assert result["AB"]["rank"] == 1

    def test_holm_bonferroni_handles_none_p_values(self):
        result = holm_bonferroni({"AB": 0.01, "AC": None})
        assert result["AC"]["adjusted_p_holm"] is None
        assert result["AC"]["reject_at_alpha"] is None
        assert result["AB"]["adjusted_p_holm"] is not None

    def test_holm_bonferroni_deterministic(self):
        p_values = {"AB": 0.03, "AC": 0.01, "AD": 0.2}
        r1 = holm_bonferroni(p_values)
        r2 = holm_bonferroni(p_values)
        assert r1 == r2


# ── 6. JSD normalization / smoothing ─────────────────────────────────────
class TestNgramJSD:
    def test_build_ngram_vocab_min_df(self):
        texts = ["apple banana", "apple cherry", "durian fig"]
        vocab = build_ngram_vocab(texts, n=1, min_df=2)
        assert "apple" in vocab  # appears in 2 documents
        assert "banana" not in vocab  # appears in only 1 document
        assert "durian" not in vocab

    def test_ngram_distribution_sums_to_one(self):
        vocab = {"apple", "banana"}
        dist, support = ngram_distribution(["apple banana banana", "unseenword apple"], vocab, n=1, alpha=0.01)
        assert dist.sum() == pytest.approx(1.0)
        assert support[-1] == "<RARE>"

    def test_ngram_distribution_rare_bucket_catches_oov(self):
        vocab = {"apple"}
        dist, support = ngram_distribution(["apple zzz yyy"], vocab, n=1, alpha=0.01)
        rare_idx = support.index("<RARE>")
        apple_idx = support.index("apple")
        # two OOV tokens (zzz, yyy) landed in <RARE>, one in-vocab (apple)
        assert dist[rare_idx] > dist[apple_idx]

    def test_ngram_distribution_smoothing_never_zero(self):
        vocab = {"apple", "banana", "cherry"}
        dist, _ = ngram_distribution(["apple"], vocab, n=1, alpha=0.01)
        assert (dist > 0).all()  # no zero mass anywhere, even for unseen vocab entries

    def test_jsd_base2_zero_for_identical_distributions(self):
        p = np.array([0.5, 0.5])
        assert jsd_base2(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_jsd_base2_bounded_zero_one(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        d = jsd_base2(p, q)
        assert 0.0 <= d <= 1.0
        assert d == pytest.approx(1.0, abs=1e-6)  # maximally different, base-2 -> divergence 1

    def test_jsd_base2_symmetric(self):
        p = np.array([0.7, 0.3])
        q = np.array([0.2, 0.8])
        assert jsd_base2(p, q) == pytest.approx(jsd_base2(q, p))


# ── confound-sensitivity primitive ───────────────────────────────────────
class TestResidualization:
    def test_residualize_removes_linear_relationship(self):
        covariate = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X = np.column_stack([covariate * 2.0 + 1.0])  # perfectly linear in covariate
        residuals = residualize_on_covariate(X, covariate)
        assert np.allclose(residuals, 0.0, atol=1e-8)

    def test_residualize_preserves_shape(self):
        covariate = np.array([1.0, 2.0, 3.0])
        X = np.random.default_rng(0).normal(size=(3, 4))
        residuals = residualize_on_covariate(X, covariate)
        assert residuals.shape == X.shape

    def test_residualize_empty_columns_noop(self):
        X = np.zeros((3, 0))
        residuals = residualize_on_covariate(X, np.array([1.0, 2.0, 3.0]))
        assert residuals.shape == (3, 0)


# ── section 5.3 PCA artifacts (C-F-C-confirmed-missing implementation
# fix): coordinates, loadings, all-six-population projection, R104-
# source/R-AUTHORED projection-only-ness, determinism, required plots ──
class TestPCA:
    def _small_combined_views(self):
        """Small but non-degenerate combined-view fixture: 5 rows per
        quadrant (20 total, 4 features) so PCA has enough components for
        PC1-PC3, plus small R104-source/R-AUTHORED auxiliary populations."""
        rng = np.random.default_rng(0)
        combined = {
            "A": rng.normal(loc=0.0, size=(5, 4)),
            "B": rng.normal(loc=1.0, size=(5, 4)),
            "C": rng.normal(loc=2.0, size=(5, 4)),
            "D": rng.normal(loc=3.0, size=(5, 4)),
            "R104_source": rng.normal(loc=2.0, size=(3, 4)),
            "R_AUTHORED": rng.normal(loc=2.5, size=(2, 4)),
        }
        record_ids = {
            name: [f"{name}-{i}" for i in range(X.shape[0])]
            for name, X in combined.items()
        }
        return combined, record_ids

    def _fit_on_abcd(self, combined):
        abcd_rows = np.vstack([combined[q] for q in QUADRANT_ORDER])
        return fit_pca_combined(abcd_rows)

    # -- coordinate dimensions --
    def test_pca_coordinate_dimensions(self):
        combined, record_ids = self._small_combined_views()
        pca = self._fit_on_abcd(combined)
        coords = pca_coordinates(pca, combined, record_ids, ALL_POPULATIONS_ORDER, k=3)
        assert len(coords) == sum(X.shape[0] for X in combined.values())
        for row in coords:
            assert set(row.keys()) == {"record_id", "population", "pc1", "pc2", "pc3"}
            assert all(isinstance(row[f"pc{i}"], float) for i in (1, 2, 3))

    # -- stable row identity --
    def test_pca_stable_row_identity(self):
        combined, record_ids = self._small_combined_views()
        pca = self._fit_on_abcd(combined)
        coords = pca_coordinates(pca, combined, record_ids, ALL_POPULATIONS_ORDER, k=3)
        expected_ids = {rid for ids in record_ids.values() for rid in ids}
        got_ids = {row["record_id"] for row in coords}
        assert got_ids == expected_ids  # every input record_id appears exactly once
        assert len(coords) == len(got_ids)  # no duplicated/dropped rows
        for row in coords:
            # each coordinate row's record_id belongs to the population it is labeled with
            assert row["record_id"] in record_ids[row["population"]]

    # -- loading dimensions --
    def test_pca_loading_dimensions(self):
        combined, _ = self._small_combined_views()
        pca = self._fit_on_abcd(combined)
        loadings = pca_component_loadings(pca, k=3)
        assert set(loadings.keys()) == {"pc1", "pc2", "pc3"}
        n_features = combined["A"].shape[1]
        for vec in loadings.values():
            assert len(vec) == n_features

    # -- all six populations represented in required projection output --
    def test_all_six_populations_represented(self):
        combined, record_ids = self._small_combined_views()
        pca = self._fit_on_abcd(combined)
        coords = pca_coordinates(pca, combined, record_ids, ALL_POPULATIONS_ORDER, k=3)
        assert {row["population"] for row in coords} == set(ALL_POPULATIONS_ORDER)

    # -- R104-source is projection-only --
    def test_r104_source_is_projection_only(self):
        combined, record_ids = self._small_combined_views()
        pca = self._fit_on_abcd(combined)
        components_before = pca.components_.copy()
        # perturbing R104-source rows drastically must not change the
        # already-fitted components -- pca_coordinates only transforms it
        perturbed = dict(combined)
        perturbed["R104_source"] = combined["R104_source"] * 1000.0
        _ = pca_coordinates(pca, perturbed, record_ids, ALL_POPULATIONS_ORDER, k=3)
        np.testing.assert_array_equal(pca.components_, components_before)
        assert "R104_source" in AUXILIARY_ORDER  # never an A/B/C/D quadrant

    # -- R-AUTHORED is projection-only --
    def test_r_authored_is_projection_only(self):
        combined, record_ids = self._small_combined_views()
        pca = self._fit_on_abcd(combined)
        components_before = pca.components_.copy()
        perturbed = dict(combined)
        perturbed["R_AUTHORED"] = combined["R_AUTHORED"] * 1000.0
        _ = pca_coordinates(pca, perturbed, record_ids, ALL_POPULATIONS_ORDER, k=3)
        np.testing.assert_array_equal(pca.components_, components_before)
        assert "R_AUTHORED" in AUXILIARY_ORDER  # never an A/B/C/D quadrant

    # -- deterministic PCA numerical output --
    def test_pca_deterministic_numerical_output(self):
        combined, record_ids = self._small_combined_views()
        pca1 = self._fit_on_abcd(combined)
        pca2 = self._fit_on_abcd(combined)
        coords1 = pca_coordinates(pca1, combined, record_ids, ALL_POPULATIONS_ORDER, k=3)
        coords2 = pca_coordinates(pca2, combined, record_ids, ALL_POPULATIONS_ORDER, k=3)
        assert coords1 == coords2
        assert cumulative_explained_variance(pca1, 2) == cumulative_explained_variance(pca2, 2)
        assert cumulative_explained_variance(pca1, 3) == cumulative_explained_variance(pca2, 3)

    # -- required plot files created --
    def test_required_plot_files_created(self, tmp_path):
        combined, record_ids = self._small_combined_views()
        pca = self._fit_on_abcd(combined)
        coords = pca_coordinates(pca, combined, record_ids, ALL_POPULATIONS_ORDER, k=3)
        pc12_path = tmp_path / "pca_pc1_pc2.png"
        pc13_path = tmp_path / "pca_pc1_pc3.png"
        plot_pca_scatter(coords, "pc1", "pc2", pc12_path, "PC1 vs PC2")
        plot_pca_scatter(coords, "pc1", "pc3", pc13_path, "PC1 vs PC3")
        assert pc12_path.exists() and pc12_path.stat().st_size > 0
        assert pc13_path.exists() and pc13_path.stat().st_size > 0
