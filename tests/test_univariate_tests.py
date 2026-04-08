import pytest
import polars as pl
import numpy as np
from polars_stats.univariate.tests import (
    shapiro_wilk, dagostino_pearson, kolmogorov_smirnov, normality,
    ttest_1samp, ztest_1samp, wilcoxon_1samp, sign_test, binomial_test,
    outliers_zscore, outliers_mad, outliers_iqr, outliers_grubbs,
    cohens_d, which_test,
)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def normal_data():
    np.random.seed(42)
    return pl.Series("x", np.random.normal(100, 15, 200).tolist())


@pytest.fixture
def skewed_data():
    np.random.seed(42)
    return pl.Series("x", np.random.exponential(10, 200).tolist())


@pytest.fixture
def with_outliers():
    data = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 200]
    return pl.Series("x", data)


@pytest.fixture
def binary():
    return pl.Series("x", [1, 1, 1, 1, 0, 0, 0, 1, 1, 0])


# ── Normality tests ─────────────────────────────────────────

class TestShapiroWilk:
    def test_normal_data_not_rejected(self, normal_data):
        p = shapiro_wilk(normal_data)
        assert p > 0.05

    def test_skewed_data_rejected(self, skewed_data):
        p = shapiro_wilk(skewed_data)
        assert p < 0.05

    def test_detail_returns_dict(self, normal_data):
        result = shapiro_wilk(normal_data, detail=True)
        assert "W" in result
        assert "pvalue" in result

    def test_returns_float_by_default(self, normal_data):
        result = shapiro_wilk(normal_data)
        assert isinstance(result, float)


class TestDagostinoPearson:
    def test_normal_data_not_rejected(self, normal_data):
        p = dagostino_pearson(normal_data)
        assert p > 0.05

    def test_skewed_data_rejected(self, skewed_data):
        p = dagostino_pearson(skewed_data)
        assert p < 0.05

    def test_detail_returns_dict(self, normal_data):
        result = dagostino_pearson(normal_data, detail=True)
        assert "K2" in result
        assert "pvalue" in result


class TestKolmogorovSmirnov:
    def test_normal_fit(self, normal_data):
        p = kolmogorov_smirnov(normal_data, dist="norm")
        assert p > 0.05

    def test_detail_returns_dict(self, normal_data):
        result = kolmogorov_smirnov(normal_data, detail=True)
        assert "D" in result
        assert "pvalue" in result


class TestNormality:
    def test_normal_data_verdict(self, normal_data):
        result = normality(normal_data)
        assert result["verdict"] == "normal"

    def test_skewed_data_verdict(self, skewed_data):
        result = normality(skewed_data)
        assert result["verdict"] == "not normal"

    def test_has_alpha(self, normal_data):
        result = normality(normal_data, alpha=0.01)
        assert result["alpha"] == 0.01

    def test_has_n(self, normal_data):
        result = normality(normal_data)
        assert result["n"] > 0


# ── Hypothesis tests ────────────────────────────────────────

class TestTtest1Samp:
    def test_true_mean_not_rejected(self, normal_data):
        p = ttest_1samp(normal_data, mu=100)
        assert p > 0.05

    def test_wrong_mean_rejected(self, normal_data):
        p = ttest_1samp(normal_data, mu=50)
        assert p < 0.05

    def test_detail(self, normal_data):
        result = ttest_1samp(normal_data, mu=100, detail=True)
        assert "t" in result
        assert "pvalue" in result


class TestZtest1Samp:
    def test_true_mean_not_rejected(self, normal_data):
        p = ztest_1samp(normal_data, mu=100)
        assert p > 0.05

    def test_wrong_mean_rejected(self, normal_data):
        p = ztest_1samp(normal_data, mu=50)
        assert p < 0.05

    def test_with_known_sigma(self, normal_data):
        p = ztest_1samp(normal_data, mu=100, sigma=15)
        assert p > 0.05


class TestWilcoxon1Samp:
    def test_true_median_not_rejected(self, normal_data):
        p = wilcoxon_1samp(normal_data, mu=100)
        assert p > 0.05

    def test_wrong_median_rejected(self, normal_data):
        p = wilcoxon_1samp(normal_data, mu=50)
        assert p < 0.05


class TestSignTest:
    def test_true_median_not_rejected(self, normal_data):
        p = sign_test(normal_data, mu=100)
        assert p > 0.05

    def test_wrong_median_rejected(self, normal_data):
        p = sign_test(normal_data, mu=50)
        assert p < 0.05

    def test_detail(self, normal_data):
        result = sign_test(normal_data, mu=100, detail=True)
        assert "n_above" in result
        assert "n_below" in result


class TestBinomialTest:
    def test_fair_coin(self, binary):
        p = binomial_test(binary, p0=0.5)
        assert p > 0.05  # 6/10 is compatible with p=0.5

    def test_biased_proportion_rejected(self):
        s = pl.Series("x", [1] * 95 + [0] * 5)
        p = binomial_test(s, p0=0.5)
        assert p < 0.05

    def test_detail(self, binary):
        result = binomial_test(binary, p0=0.5, detail=True)
        assert "k" in result
        assert "observed_proportion" in result


# ── Outlier detection ────────────────────────────────────────

class TestOutliersZscore:
    def test_detects_outlier(self, with_outliers):
        result = outliers_zscore(with_outliers)
        assert 200 in result["values"]

    def test_no_outliers_in_clean_data(self):
        s = pl.Series("x", list(range(10)))
        result = outliers_zscore(s)
        assert result["count"] == 0

    def test_threshold(self, with_outliers):
        strict = outliers_zscore(with_outliers, threshold=2.0)
        loose = outliers_zscore(with_outliers, threshold=4.0)
        assert strict["count"] >= loose["count"]


class TestOutliersMad:
    def test_detects_outlier(self, with_outliers):
        result = outliers_mad(with_outliers)
        assert 200 in result["values"]

    def test_handles_zero_mad(self):
        s = pl.Series("x", [5, 5, 5, 5, 5])
        result = outliers_mad(s)
        assert result["mad"] == 0
        assert result["count"] == 0


class TestOutliersIqr:
    def test_detects_outlier(self, with_outliers):
        result = outliers_iqr(with_outliers)
        assert 200 in result["values"]

    def test_has_fences(self, with_outliers):
        result = outliers_iqr(with_outliers)
        assert "lower_fence" in result
        assert "upper_fence" in result


class TestOutliersGrubbs:
    def test_detects_outlier(self, with_outliers):
        result = outliers_grubbs(with_outliers)
        assert result["suspect_value"] == 200
        assert result["is_outlier"]

    def test_no_outlier_in_clean_data(self):
        s = pl.Series("x", [10, 11, 12, 13, 14, 15])
        result = outliers_grubbs(s)
        assert not result["is_outlier"]


# ── Effect size ──────────────────────────────────────────────

class TestCohensD:
    def test_zero_effect(self, normal_data):
        d = cohens_d(normal_data, mu=100)
        assert abs(d) < 0.5

    def test_large_effect(self, normal_data):
        d = cohens_d(normal_data, mu=50)
        assert abs(d) > 0.8

    def test_zero_std_raises(self):
        s = pl.Series("x", [5, 5, 5, 5])
        with pytest.raises(ArithmeticError):
            cohens_d(s, mu=10)


# ── which_test ───────────────────────────────────────────────

class TestWhichTest:
    def test_normality_objective(self, normal_data):
        result = which_test(normal_data, "normality")
        assert "recommended" in result
        assert "reason" in result

    def test_location_objective(self, normal_data):
        result = which_test(normal_data, "location")
        assert "recommended" in result

    def test_outliers_objective(self, normal_data):
        result = which_test(normal_data, "outliers")
        assert "recommended" in result

    def test_unknown_objective_raises(self, normal_data):
        with pytest.raises(ValueError):
            which_test(normal_data, "unknown")