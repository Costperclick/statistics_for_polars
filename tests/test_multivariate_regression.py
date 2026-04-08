import pytest
import polars as pl
import numpy as np
from polars_stats.multivariate.correlation import (
    pearson, spearman, kendall, point_biserial,
    chi2_independence, cramers_v, mutual_information,
)


@pytest.fixture
def correlated():
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = x * 2 + np.random.normal(0, 0.5, 100)
    return pl.Series("x", x.tolist()), pl.Series("y", y.tolist())


@pytest.fixture
def uncorrelated():
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 1, 100)
    return pl.Series("x", x.tolist()), pl.Series("y", y.tolist())


@pytest.fixture
def categorical_df():
    np.random.seed(42)
    plan = np.random.choice(["basic", "pro", "enterprise"], 200)
    churned = np.where(plan == "basic",
                       np.random.choice([0, 1], 200, p=[0.5, 0.5]),
                       np.random.choice([0, 1], 200, p=[0.8, 0.2]))
    return pl.DataFrame({"plan": plan, "churned": churned.astype(str)})


class TestPearson:
    def test_strong_correlation(self, correlated):
        x, y = correlated
        r = pearson(x, y)
        assert r > 0.8

    def test_weak_correlation(self, uncorrelated):
        x, y = uncorrelated
        r = pearson(x, y)
        assert abs(r) < 0.3

    def test_range(self, correlated):
        x, y = correlated
        r = pearson(x, y)
        assert -1 <= r <= 1

    def test_detail(self, correlated):
        x, y = correlated
        result = pearson(x, y, detail=True)
        assert "r" in result
        assert "pvalue" in result


class TestSpearman:
    def test_strong_correlation(self, correlated):
        x, y = correlated
        rho = spearman(x, y)
        assert rho > 0.8

    def test_detail(self, correlated):
        x, y = correlated
        result = spearman(x, y, detail=True)
        assert "rho" in result
        assert "pvalue" in result


class TestKendall:
    def test_strong_correlation(self, correlated):
        x, y = correlated
        tau = kendall(x, y)
        assert tau > 0.5

    def test_detail(self, correlated):
        x, y = correlated
        result = kendall(x, y, detail=True)
        assert "tau" in result


class TestPointBiserial:
    def test_basic(self):
        binary = pl.Series("b", [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        continuous = pl.Series("c", [10, 12, 11, 13, 10, 20, 22, 21, 23, 20])
        r = point_biserial(binary, continuous)
        assert r > 0.5

    def test_no_association(self):
        np.random.seed(42)
        binary = pl.Series("b", np.random.choice([0, 1], 100).tolist())
        continuous = pl.Series("c", np.random.normal(0, 1, 100).tolist())
        r = point_biserial(binary, continuous)
        assert abs(r) < 0.3


class TestChi2Independence:
    def test_returns_float(self, categorical_df):
        p = chi2_independence(categorical_df, "plan", "churned")
        assert isinstance(p, float)

    def test_detail(self, categorical_df):
        result = chi2_independence(categorical_df, "plan", "churned", detail=True)
        assert "chi2" in result
        assert "dof" in result


class TestCramersV:
    def test_range(self, categorical_df):
        v = cramers_v(categorical_df, "plan", "churned")
        assert 0 <= v <= 1


class TestMutualInformation:
    def test_correlated_higher_than_uncorrelated(self, correlated, uncorrelated):
        cx, cy = correlated
        ux, uy = uncorrelated
        mi_corr = mutual_information(cx, cy)
        mi_uncorr = mutual_information(ux, uy)
        assert mi_corr > mi_uncorr

    def test_non_negative(self, correlated):
        x, y = correlated
        mi = mutual_information(x, y)
        assert mi >= 0