import pytest
import polars as pl
import numpy as np
from polars_stats.multivariate.comparison import (
    ttest_ind, ttest_paired, welch_ttest, mann_whitney, wilcoxon_paired,
    kolmogorov_smirnov_2samp, levene, bartlett,
    cohens_d_2samp, rank_biserial,
    anova_oneway, welch_anova, kruskal_wallis, friedman,
    tukey_hsd, eta_squared, omega_squared,
)


@pytest.fixture
def two_groups_different():
    np.random.seed(42)
    a = np.random.normal(50, 10, 50)
    b = np.random.normal(80, 10, 50)
    groups = ["A"] * 50 + ["B"] * 50
    values = np.concatenate([a, b])
    return pl.DataFrame({"value": values, "group": groups})


@pytest.fixture
def two_groups_same():
    np.random.seed(42)
    a = np.random.normal(50, 10, 50)
    b = np.random.normal(50, 10, 50)
    groups = ["A"] * 50 + ["B"] * 50
    values = np.concatenate([a, b])
    return pl.DataFrame({"value": values, "group": groups})


@pytest.fixture
def three_groups():
    np.random.seed(42)
    a = np.random.normal(50, 10, 40)
    b = np.random.normal(60, 10, 40)
    c = np.random.normal(80, 10, 40)
    groups = ["A"] * 40 + ["B"] * 40 + ["C"] * 40
    values = np.concatenate([a, b, c])
    return pl.DataFrame({"value": values, "group": groups})


@pytest.fixture
def paired_data():
    np.random.seed(42)
    before = np.random.normal(50, 10, 30)
    after = before + np.random.normal(10, 5, 30)  # improvement
    return pl.Series("before", before.tolist()), pl.Series("after", after.tolist())


# ── Two-group independent ────────────────────────────────────

class TestTtestInd:
    def test_different_groups_rejected(self, two_groups_different):
        p = ttest_ind(two_groups_different, "value", "group")
        assert p < 0.05

    def test_same_groups_not_rejected(self, two_groups_same):
        p = ttest_ind(two_groups_same, "value", "group")
        assert p > 0.05

    def test_detail(self, two_groups_different):
        result = ttest_ind(two_groups_different, "value", "group", detail=True)
        assert "t" in result
        assert "mean_a" in result


class TestWelchTtest:
    def test_different_groups_rejected(self, two_groups_different):
        p = welch_ttest(two_groups_different, "value", "group")
        assert p < 0.05

    def test_same_groups_not_rejected(self, two_groups_same):
        p = welch_ttest(two_groups_same, "value", "group")
        assert p > 0.05


class TestMannWhitney:
    def test_different_groups_rejected(self, two_groups_different):
        p = mann_whitney(two_groups_different, "value", "group")
        assert p < 0.05

    def test_same_groups_not_rejected(self, two_groups_same):
        p = mann_whitney(two_groups_same, "value", "group")
        assert p > 0.05


# ── Two-group paired ────────────────────────────────────────

class TestTtestPaired:
    def test_improvement_detected(self, paired_data):
        before, after = paired_data
        p = ttest_paired(before, after)
        assert p < 0.05

    def test_length_mismatch_raises(self):
        a = pl.Series("a", [1, 2, 3])
        b = pl.Series("b", [1, 2])
        with pytest.raises(ValueError):
            ttest_paired(a, b)


class TestWilcoxonPaired:
    def test_improvement_detected(self, paired_data):
        before, after = paired_data
        p = wilcoxon_paired(before, after)
        assert p < 0.05


class TestKS2Samp:
    def test_different_distributions(self):
        a = pl.Series("a", np.random.normal(0, 1, 100).tolist())
        b = pl.Series("b", np.random.exponential(1, 100).tolist())
        p = kolmogorov_smirnov_2samp(a, b)
        assert p < 0.05

    def test_same_distribution(self):
        np.random.seed(42)
        a = pl.Series("a", np.random.normal(0, 1, 100).tolist())
        b = pl.Series("b", np.random.normal(0, 1, 100).tolist())
        p = kolmogorov_smirnov_2samp(a, b)
        assert p > 0.05


# ── Variance equality ───────────────────────────────────────

class TestLevene:
    def test_equal_variances(self, two_groups_same):
        p = levene(two_groups_same, "value", "group")
        assert p > 0.05

    def test_unequal_variances(self):
        np.random.seed(42)
        a = np.random.normal(50, 5, 50)
        b = np.random.normal(50, 30, 50)
        df = pl.DataFrame({"value": np.concatenate([a, b]), "group": ["A"] * 50 + ["B"] * 50})
        p = levene(df, "value", "group")
        assert p < 0.05


class TestBartlett:
    def test_equal_variances(self, two_groups_same):
        p = bartlett(two_groups_same, "value", "group")
        assert p > 0.05


# ── Effect sizes ─────────────────────────────────────────────

class TestCohensD2Samp:
    def test_large_effect(self, two_groups_different):
        result = cohens_d_2samp(two_groups_different, "value", "group")
        assert abs(result["d"]) > 0.8

    def test_small_effect(self, two_groups_same):
        result = cohens_d_2samp(two_groups_same, "value", "group")
        assert abs(result["d"]) < 0.5


class TestRankBiserial:
    def test_returns_r(self, two_groups_different):
        result = rank_biserial(two_groups_different, "value", "group")
        assert "r" in result
        assert -1 <= result["r"] <= 1


# ── K-group comparisons ─────────────────────────────────────

class TestAnovaOneway:
    def test_different_groups_rejected(self, three_groups):
        p = anova_oneway(three_groups, "value", "group")
        assert p < 0.05

    def test_detail(self, three_groups):
        result = anova_oneway(three_groups, "value", "group", detail=True)
        assert "F" in result
        assert "n_groups" in result


class TestWelchAnova:
    def test_different_groups_rejected(self, three_groups):
        p = welch_anova(three_groups, "value", "group")
        assert p < 0.05


class TestKruskalWallis:
    def test_different_groups_rejected(self, three_groups):
        p = kruskal_wallis(three_groups, "value", "group")
        assert p < 0.05


class TestFriedman:
    def test_different_conditions(self):
        np.random.seed(42)
        a = pl.Series("a", np.random.normal(10, 2, 30).tolist())
        b = pl.Series("b", np.random.normal(15, 2, 30).tolist())
        c = pl.Series("c", np.random.normal(20, 2, 30).tolist())
        p = friedman([a, b, c])
        assert p < 0.05


# ── Post-hoc ─────────────────────────────────────────────────

class TestTukeyHSD:
    def test_returns_comparisons(self, three_groups):
        result = tukey_hsd(three_groups, "value", "group")
        assert "comparisons" in result
        assert len(result["comparisons"]) == 3  # 3 pairs from 3 groups


# ── K-group effect sizes ────────────────────────────────────

class TestEtaSquared:
    def test_range(self, three_groups):
        result = eta_squared(three_groups, "value", "group")
        assert 0 <= result <= 1

    def test_large_effect(self, three_groups):
        result = eta_squared(three_groups, "value", "group")
        assert result > 0.14


class TestOmegaSquared:
    def test_range(self, three_groups):
        result = omega_squared(three_groups, "value", "group")
        assert 0 <= result <= 1

    def test_less_than_eta(self, three_groups):
        eta = eta_squared(three_groups, "value", "group")
        omega = omega_squared(three_groups, "value", "group")
        assert omega <= eta