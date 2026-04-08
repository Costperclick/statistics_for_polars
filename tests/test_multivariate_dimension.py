import pytest
import polars as pl
import numpy as np
from polars_stats.multivariate.dimension import pca, scree_data


@pytest.fixture
def df():
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = x1 * 0.9 + np.random.normal(0, 0.3, n)  # correlated with x1
    x3 = np.random.normal(0, 1, n)  # independent
    return pl.DataFrame({"x1": x1, "x2": x2, "x3": x3})


class TestPCA:
    def test_returns_scores(self, df):
        result = pca(df, n_components=2)
        assert "scores" in result
        assert len(result["scores"]) == 200
        assert len(result["scores"][0]) == 2

    def test_returns_loadings(self, df):
        result = pca(df, n_components=2)
        assert "loadings" in result
        assert "PC1" in result["loadings"]
        assert "x1" in result["loadings"]["PC1"]

    def test_variance_ratio_sums_to_less_than_one(self, df):
        result = pca(df, n_components=2)
        total = sum(result["explained_variance_ratio"])
        assert total < 1.0

    def test_first_component_captures_most(self, df):
        result = pca(df, n_components=2)
        assert result["explained_variance_ratio"][0] > result["explained_variance_ratio"][1]

    def test_n_components_respected(self, df):
        result = pca(df, n_components=1)
        assert result["n_components"] == 1
        assert len(result["scores"][0]) == 1


class TestScreeData:
    def test_returns_eigenvalues(self, df):
        result = scree_data(df)
        assert "eigenvalues" in result
        assert len(result["eigenvalues"]) == 3

    def test_eigenvalues_descending(self, df):
        result = scree_data(df)
        evs = result["eigenvalues"]
        assert evs == sorted(evs, reverse=True)

    def test_cumulative_increases(self, df):
        result = scree_data(df)
        cumulative = result["cumulative_variance"]
        for i in range(1, len(cumulative)):
            assert cumulative[i] >= cumulative[i - 1]

    def test_cumulative_ends_at_one(self, df):
        result = scree_data(df)
        assert abs(result["cumulative_variance"][-1] - 1.0) < 1e-10

    def test_kaiser_criterion(self, df):
        result = scree_data(df)
        assert result["kaiser_criterion"] >= 1

    def test_components_for_thresholds(self, df):
        result = scree_data(df)
        assert result["components_for_80pct"] <= result["components_for_90pct"]