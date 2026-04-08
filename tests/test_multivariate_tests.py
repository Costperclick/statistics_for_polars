import pytest
import polars as pl
import numpy as np
from polars_stats.multivariate.tests import hotelling_t2, mahalanobis, box_m


@pytest.fixture
def two_groups_different():
    np.random.seed(42)
    n = 50
    a1 = np.random.normal(50, 10, n)
    a2 = np.random.normal(30, 5, n)
    b1 = np.random.normal(80, 10, n)
    b2 = np.random.normal(60, 5, n)
    return pl.DataFrame({
        "x1": np.concatenate([a1, b1]),
        "x2": np.concatenate([a2, b2]),
        "group": ["A"] * n + ["B"] * n,
    })


@pytest.fixture
def two_groups_same():
    np.random.seed(42)
    n = 50
    a1 = np.random.normal(50, 10, n)
    a2 = np.random.normal(30, 5, n)
    b1 = np.random.normal(50, 10, n)
    b2 = np.random.normal(30, 5, n)
    return pl.DataFrame({
        "x1": np.concatenate([a1, b1]),
        "x2": np.concatenate([a2, b2]),
        "group": ["A"] * n + ["B"] * n,
    })


@pytest.fixture
def with_outlier():
    np.random.seed(42)
    n = 50
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    # Add a multivariate outlier
    x1 = np.append(x1, 5)
    x2 = np.append(x2, 5)
    return pl.DataFrame({"x1": x1, "x2": x2})


class TestHotellingT2:
    def test_different_groups_rejected(self, two_groups_different):
        p = hotelling_t2(two_groups_different, ["x1", "x2"], "group")
        assert p < 0.05

    def test_same_groups_not_rejected(self, two_groups_same):
        p = hotelling_t2(two_groups_same, ["x1", "x2"], "group")
        assert p > 0.05

    def test_detail(self, two_groups_different):
        result = hotelling_t2(two_groups_different, ["x1", "x2"], "group", detail=True)
        assert "T2" in result
        assert "F" in result
        assert "pvalue" in result
        assert "mean_a" in result

    def test_wrong_group_count_raises(self):
        df = pl.DataFrame({
            "x1": [1, 2, 3],
            "x2": [4, 5, 6],
            "group": ["A", "B", "C"],
        })
        with pytest.raises(ValueError):
            hotelling_t2(df, ["x1", "x2"], "group")


class TestMahalanobis:
    def test_returns_series(self, with_outlier):
        result = mahalanobis(with_outlier, ["x1", "x2"])
        assert isinstance(result, pl.Series)
        assert len(result) == 51

    def test_outlier_has_highest_distance(self, with_outlier):
        result = mahalanobis(with_outlier, ["x1", "x2"])
        # Last point is the outlier
        assert result[-1] == result.max()

    def test_all_positive(self, with_outlier):
        result = mahalanobis(with_outlier, ["x1", "x2"])
        assert result.min() >= 0


class TestBoxM:
    def test_equal_covariance_not_rejected(self, two_groups_same):
        result = box_m(two_groups_same, ["x1", "x2"], "group")
        assert result["pvalue"] > 0.05
        assert result["equal_covariance"]

    def test_returns_all_fields(self, two_groups_same):
        result = box_m(two_groups_same, ["x1", "x2"], "group")
        assert "M" in result
        assert "chi2" in result
        assert "df" in result
        assert "pvalue" in result
        assert "groups" in result