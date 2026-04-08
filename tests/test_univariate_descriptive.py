import pytest
import polars as pl
import numpy as np
from polars_stats.multivariate.descriptive import (
    mean, cross_summary, correlation_matrix, covariance_matrix, partial_correlation,
)


@pytest.fixture
def df():
    np.random.seed(42)
    n = 100
    x = np.random.normal(50, 10, n)
    y = x * 2 + np.random.normal(0, 5, n)
    z = np.random.normal(30, 8, n)
    return pl.DataFrame({"x": x, "y": y, "z": z})


class TestMean:
    def test_returns_dict(self, df):
        result = mean(df)
        assert isinstance(result, dict)
        assert "x" in result
        assert "y" in result
        assert "z" in result

    def test_values_close_to_expected(self, df):
        result = mean(df)
        assert abs(result["x"] - 50) < 5
        assert abs(result["z"] - 30) < 5

    def test_subset_columns(self, df):
        result = mean(df, columns=["x", "z"])
        assert "x" in result
        assert "z" in result
        assert "y" not in result


class TestCrossSummary:
    def test_returns_dataframe(self, df):
        result = cross_summary(df)
        assert isinstance(result, pl.DataFrame)

    def test_subset_columns(self, df):
        result = cross_summary(df, columns=["x"])
        assert result.shape[1] >= 1


class TestCorrelationMatrix:
    def test_diagonal_is_one(self, df):
        result = correlation_matrix(df)
        for col in ["x", "y", "z"]:
            val = result.filter(pl.col("column") == col)[col].item()
            assert abs(val - 1.0) < 1e-10

    def test_x_y_highly_correlated(self, df):
        result = correlation_matrix(df)
        r = result.filter(pl.col("column") == "x")["y"].item()
        assert r > 0.8

    def test_spearman(self, df):
        result = correlation_matrix(df, method="spearman")
        assert result.shape[0] == 3

    def test_kendall(self, df):
        result = correlation_matrix(df, method="kendall")
        assert result.shape[0] == 3

    def test_unknown_method_raises(self, df):
        with pytest.raises(ValueError):
            correlation_matrix(df, method="unknown")


class TestCovarianceMatrix:
    def test_returns_dataframe(self, df):
        result = covariance_matrix(df)
        assert isinstance(result, pl.DataFrame)

    def test_diagonal_is_positive(self, df):
        result = covariance_matrix(df)
        for col in ["x", "y", "z"]:
            val = result.filter(pl.col("column") == col)[col].item()
            assert val > 0

    def test_symmetric(self, df):
        result = covariance_matrix(df)
        xy = result.filter(pl.col("column") == "x")["y"].item()
        yx = result.filter(pl.col("column") == "y")["x"].item()
        assert abs(xy - yx) < 1e-10


class TestPartialCorrelation:
    def test_controlled_correlation(self, df):
        result = partial_correlation(df, x="x", y="z", controls=["y"])
        assert "partial_r" in result
        assert "pvalue" in result
        assert -1 <= result["partial_r"] <= 1

    def test_x_y_partial_still_correlated(self, df):
        result = partial_correlation(df, x="x", y="y", controls=["z"])
        assert abs(result["partial_r"]) > 0.5