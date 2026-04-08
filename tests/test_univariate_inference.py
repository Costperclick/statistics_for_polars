import pytest
import polars as pl
import numpy as np
from polars_stats.univariate.inference import (
    ci_mean_bootstrap, ci_median_bootstrap, ci_mean, ci_proportion, ci_variance,
)


@pytest.fixture
def normal_data():
    np.random.seed(42)
    return pl.Series("x", np.random.normal(100, 15, 200).tolist())


class TestCIMeanBootstrap:
    def test_contains_true_mean(self, normal_data):
        result = ci_mean_bootstrap(normal_data, confidence=0.95, iterations=5000)
        assert result["lower"] < 100 < result["upper"]

    def test_higher_confidence_wider(self, normal_data):
        narrow = ci_mean_bootstrap(normal_data, confidence=0.90, iterations=5000)
        wide = ci_mean_bootstrap(normal_data, confidence=0.99, iterations=5000)
        width_narrow = narrow["upper"] - narrow["lower"]
        width_wide = wide["upper"] - wide["lower"]
        assert width_wide > width_narrow

    def test_has_metadata(self, normal_data):
        result = ci_mean_bootstrap(normal_data)
        assert result["confidence"] == 0.95
        assert result["iterations"] == 10000
        assert result["method"] == "bootstrap"


class TestCIMedianBootstrap:
    def test_contains_true_median(self, normal_data):
        result = ci_median_bootstrap(normal_data, confidence=0.95, iterations=5000)
        assert result["lower"] < 100 < result["upper"]

    def test_has_metadata(self, normal_data):
        result = ci_median_bootstrap(normal_data)
        assert result["method"] == "bootstrap"


class TestCIMean:
    def test_contains_true_mean(self, normal_data):
        result = ci_mean(normal_data, confidence=0.95)
        assert result["lower"] < 100 < result["upper"]

    def test_higher_confidence_wider(self, normal_data):
        narrow = ci_mean(normal_data, confidence=0.90)
        wide = ci_mean(normal_data, confidence=0.99)
        assert (wide["upper"] - wide["lower"]) > (narrow["upper"] - narrow["lower"])

    def test_has_mean_and_margin(self, normal_data):
        result = ci_mean(normal_data)
        assert "mean" in result
        assert "margin_of_error" in result
        assert result["method"] == "student_t"

    def test_symmetric_around_mean(self, normal_data):
        result = ci_mean(normal_data)
        lower_dist = result["mean"] - result["lower"]
        upper_dist = result["upper"] - result["mean"]
        assert abs(lower_dist - upper_dist) < 1e-10


class TestCIProportion:
    def test_wilson_contains_true(self):
        # 50 successes out of 100, true proportion = 0.5
        result = ci_proportion(50, 100, method="wilson")
        assert result["lower"] < 0.5 < result["upper"]

    def test_wald_contains_true(self):
        result = ci_proportion(50, 100, method="wald")
        assert result["lower"] < 0.5 < result["upper"]

    def test_clopper_pearson_contains_true(self):
        result = ci_proportion(50, 100, method="clopper_pearson")
        assert result["lower"] < 0.5 < result["upper"]

    def test_bounds_between_0_and_1(self):
        result = ci_proportion(1, 100, method="wilson")
        assert result["lower"] >= 0
        assert result["upper"] <= 1

    def test_clopper_pearson_wider_than_wald(self):
        cp = ci_proportion(30, 100, method="clopper_pearson")
        wald = ci_proportion(30, 100, method="wald")
        width_cp = cp["upper"] - cp["lower"]
        width_wald = wald["upper"] - wald["lower"]
        assert width_cp >= width_wald

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            ci_proportion(50, 100, method="unknown")

    def test_has_proportion(self):
        result = ci_proportion(45, 100)
        assert abs(result["proportion"] - 0.45) < 1e-10


class TestCIVariance:
    def test_contains_true_variance(self, normal_data):
        result = ci_variance(normal_data, confidence=0.95)
        true_var = 15 ** 2  # 225
        assert result["lower"] < true_var < result["upper"]

    def test_has_sample_variance(self, normal_data):
        result = ci_variance(normal_data)
        assert "variance" in result
        assert result["method"] == "chi_squared"

    def test_higher_confidence_wider(self, normal_data):
        narrow = ci_variance(normal_data, confidence=0.90)
        wide = ci_variance(normal_data, confidence=0.99)
        assert (wide["upper"] - wide["lower"]) > (narrow["upper"] - narrow["lower"])