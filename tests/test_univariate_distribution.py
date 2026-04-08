import pytest
import polars as pl
import numpy as np
from polars_stats.univariate.distribution import distribution_fit, qqplot_data, kde


@pytest.fixture
def normal_data():
    np.random.seed(42)
    return pl.Series("x", np.random.normal(50, 10, 500).tolist())


@pytest.fixture
def exponential_data():
    np.random.seed(42)
    return pl.Series("x", np.random.exponential(5, 500).tolist())


class TestDistributionFit:
    def test_normal_fit(self, normal_data):
        result = distribution_fit(normal_data, dist="norm")
        assert result["distribution"] == "norm"
        assert "params" in result
        assert "log_likelihood" in result
        # Fitted mean should be close to 50
        assert abs(result["params"][0] - 50) < 5

    def test_exponential_fit(self, exponential_data):
        result = distribution_fit(exponential_data, dist="expon")
        assert result["distribution"] == "expon"

    def test_unknown_distribution_raises(self, normal_data):
        with pytest.raises(ValueError):
            distribution_fit(normal_data, dist="fake_distribution")


class TestQQPlotData:
    def test_returns_coordinates(self, normal_data):
        result = qqplot_data(normal_data)
        assert "theoretical" in result
        assert "observed" in result
        assert len(result["theoretical"]) == len(result["observed"])

    def test_observed_is_sorted(self, normal_data):
        result = qqplot_data(normal_data)
        observed = result["observed"]
        assert observed == sorted(observed)

    def test_custom_distribution(self, exponential_data):
        result = qqplot_data(exponential_data, dist="expon")
        assert result["distribution"] == "expon"


class TestKDE:
    def test_returns_x_and_density(self, normal_data):
        result = kde(normal_data)
        assert "x" in result
        assert "density" in result
        assert "bandwidth" in result

    def test_length_matches_n_points(self, normal_data):
        result = kde(normal_data, n_points=100)
        assert len(result["x"]) == 100
        assert len(result["density"]) == 100

    def test_density_is_positive(self, normal_data):
        result = kde(normal_data)
        assert all(d >= 0 for d in result["density"])

    def test_custom_n_points(self, normal_data):
        result = kde(normal_data, n_points=50)
        assert len(result["x"]) == 50