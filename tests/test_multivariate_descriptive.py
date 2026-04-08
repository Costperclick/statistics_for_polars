import pytest
import math
import polars as pl
from polars_stats.univariate.descriptive import (
    mean, median, weighted_mean, geometric_mean, harmonic_mean,
    mode, variance, standard_deviation, median_absolute_deviation,
    skewness, kurtosis, quantile, iqr, data_range,
    variance_coefficient, s_entropy, gini,
)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def simple():
    return pl.Series("x", [1, 2, 3, 4, 5])


@pytest.fixture
def with_nulls():
    return pl.Series("x", [1, 2, None, 4, 5])


@pytest.fixture
def symmetric():
    return pl.Series("x", [10, 20, 30, 40, 50])


@pytest.fixture
def single():
    return pl.Series("x", [42])


@pytest.fixture
def identical():
    return pl.Series("x", [7, 7, 7, 7, 7])


# ── mean ─────────────────────────────────────────────────────

class TestMean:
    def test_basic(self, simple):
        assert mean(simple) == 3.0

    def test_fast_and_manual_agree(self, simple):
        assert mean(simple, fast=True) == mean(simple, fast=False)

    def test_with_nulls(self, with_nulls):
        result = mean(with_nulls)
        assert result == 3.0  # (1+2+4+5) / 4

    def test_single_value(self, single):
        assert mean(single) == 42


# ── median ───────────────────────────────────────────────────

class TestMedian:
    def test_odd_length(self, simple):
        assert median(simple) == 3.0

    def test_even_length(self):
        s = pl.Series("x", [1, 2, 3, 4])
        assert median(s) == 2.5

    def test_fast_and_manual_agree(self, simple):
        assert median(simple, fast=True) == median(simple, fast=False)

    def test_with_nulls(self, with_nulls):
        result = median(with_nulls)
        assert result == 3.0  # [1, 2, 4, 5] → (2+4)/2

    def test_single_value(self, single):
        assert median(single) == 42


# ── weighted_mean ────────────────────────────────────────────

class TestWeightedMean:
    def test_basic(self):
        values = pl.Series("v", [10, 20, 30])
        weights = pl.Series("w", [1, 1, 1])
        assert weighted_mean(values, weights) == 20.0

    def test_weighted(self):
        values = pl.Series("v", [12, 15, 8])
        weights = pl.Series("w", [1, 3, 2])
        expected = (12 * 1 + 15 * 3 + 8 * 2) / (1 + 3 + 2)
        assert abs(weighted_mean(values, weights) - expected) < 1e-10

    def test_null_weight_ignored(self):
        values = pl.Series("v", [10, 20, 30])
        weights = pl.Series("w", [1, None, 1])
        assert weighted_mean(values, weights) == 20.0

    def test_null_value_ignored(self):
        values = pl.Series("v", [10, None, 30])
        weights = pl.Series("w", [1, 1, 1])
        assert weighted_mean(values, weights) == 20.0

    def test_length_mismatch_raises(self):
        values = pl.Series("v", [1, 2])
        weights = pl.Series("w", [1])
        with pytest.raises(ArithmeticError):
            weighted_mean(values, weights)

    def test_zero_weights_raises(self):
        values = pl.Series("v", [1, 2, 3])
        weights = pl.Series("w", [0, 0, 0])
        with pytest.raises(ArithmeticError):
            weighted_mean(values, weights)


# ── geometric_mean ───────────────────────────────────────────

class TestGeometricMean:
    def test_basic(self):
        s = pl.Series("x", [1, 2, 4])
        expected = (1 * 2 * 4) ** (1 / 3)
        assert abs(geometric_mean(s) - expected) < 1e-10

    def test_log_and_naive_agree(self):
        s = pl.Series("x", [1.1, 0.8, 1.3])
        assert abs(geometric_mean(s, log_transform=True) - geometric_mean(s, log_transform=False)) < 1e-10

    def test_zero_raises(self):
        s = pl.Series("x", [1, 0, 3])
        with pytest.raises(ArithmeticError):
            geometric_mean(s)

    def test_negative_raises(self):
        s = pl.Series("x", [1, -2, 3])
        with pytest.raises(ArithmeticError):
            geometric_mean(s)

    def test_with_nulls(self):
        s = pl.Series("x", [2, None, 8])
        expected = (2 * 8) ** (1 / 2)
        assert abs(geometric_mean(s) - expected) < 1e-10


# ── harmonic_mean ────────────────────────────────────────────

class TestHarmonicMean:
    def test_basic(self):
        s = pl.Series("x", [60, 40])
        expected = 2 / (1 / 60 + 1 / 40)
        assert abs(harmonic_mean(s) - expected) < 1e-10

    def test_zero_raises(self):
        s = pl.Series("x", [1, 0, 3])
        with pytest.raises(ArithmeticError):
            harmonic_mean(s)

    def test_negative_raises(self):
        s = pl.Series("x", [1, -2, 3])
        with pytest.raises(ArithmeticError):
            harmonic_mean(s)

    def test_inequality(self):
        """harmonic <= geometric <= arithmetic for positive values"""
        s = pl.Series("x", [2, 4, 8])
        h = harmonic_mean(s)
        g = geometric_mean(s)
        a = mean(s)
        assert h <= g <= a


# ── mode ─────────────────────────────────────────────────────

class TestMode:
    def test_single_mode(self):
        s = pl.Series("x", [1, 2, 2, 3])
        assert mode(s) == [2]

    def test_multimodal(self):
        s = pl.Series("x", [1, 1, 2, 2, 3])
        result = mode(s)
        assert set(result) == {1, 2}

    def test_all_same_frequency_raises(self):
        s = pl.Series("x", [1, 2, 3, 4])
        with pytest.raises(ArithmeticError):
            mode(s)

    def test_with_nulls(self):
        s = pl.Series("x", [1, 2, 2, None])
        assert mode(s) == [2]


# ── variance ─────────────────────────────────────────────────

class TestVariance:
    def test_basic(self):
        s = pl.Series("x", [2, 4, 6, 8, 10])
        expected = 10.0  # sum of (xi - 6)² / 4
        assert abs(variance(s) - expected) < 1e-10

    def test_identical_values(self, identical):
        assert variance(identical) == 0.0

    def test_with_nulls(self):
        s = pl.Series("x", [2, 4, None, 8, 10])
        clean = pl.Series("x", [2, 4, 8, 10])
        assert abs(variance(s) - variance(clean)) < 1e-10


# ── standard_deviation ───────────────────────────────────────

class TestStd:
    def test_basic(self):
        s = pl.Series("x", [2, 4, 6, 8, 10])
        assert abs(standard_deviation(s) - math.sqrt(10)) < 1e-10

    def test_identical_values(self, identical):
        assert standard_deviation(identical) == 0.0


# ── median_absolute_deviation ────────────────────────────────

class TestMAD:
    def test_basic(self):
        s = pl.Series("x", [1, 2, 3, 4, 5])
        # median = 3, deviations = [2, 1, 0, 1, 2], median of deviations = 1
        assert median_absolute_deviation(s) == 1.0

    def test_with_outlier(self):
        s = pl.Series("x", [1, 2, 3, 4, 100])
        # MAD should be robust to the outlier
        result = median_absolute_deviation(s)
        assert result < 10  # std would be ~43

    def test_identical_values(self, identical):
        assert median_absolute_deviation(identical) == 0.0


# ── skewness ─────────────────────────────────────────────────

class TestSkewness:
    def test_symmetric(self):
        s = pl.Series("x", [1, 2, 3, 4, 5])
        assert abs(skewness(s)) < 0.01

    def test_right_skewed(self):
        s = pl.Series("x", [1, 1, 1, 1, 1, 100])
        assert skewness(s) > 0

    def test_left_skewed(self):
        s = pl.Series("x", [100, 99, 99, 99, 99, 1])
        assert skewness(s) < 0


# ── kurtosis ─────────────────────────────────────────────────

class TestKurtosis:
    def test_normal_like(self):
        # A perfectly uniform distribution has kurtosis < 0
        s = pl.Series("x", list(range(1, 101)))
        result = kurtosis(s)
        assert result < 0  # platykurtic

    def test_heavy_tails(self):
        s = pl.Series("x", [0, 0, 0, 0, 0, 0, 0, 0, 100, -100])
        assert kurtosis(s) > 0  # leptokurtic


# ── quantile ─────────────────────────────────────────────────

class TestQuantile:
    def test_median_is_q50(self, simple):
        assert quantile(simple, 0.5) == median(simple)

    def test_min_is_q0(self, simple):
        assert quantile(simple, 0.0) == 1

    def test_max_is_q1(self, simple):
        assert quantile(simple, 1.0) == 5

    def test_invalid_q_raises(self, simple):
        with pytest.raises(ArithmeticError):
            quantile(simple, 1.5)

        with pytest.raises(ArithmeticError):
            quantile(simple, -0.1)


# ── iqr ──────────────────────────────────────────────────────

class TestIQR:
    def test_basic(self, simple):
        result = iqr(simple)
        q1 = quantile(simple, 0.25)
        q3 = quantile(simple, 0.75)
        assert result == q3 - q1

    def test_identical_values(self, identical):
        assert iqr(identical) == 0.0


# ── data_range ───────────────────────────────────────────────

class TestDataRange:
    def test_basic(self, simple):
        assert data_range(simple) == 4  # 5 - 1

    def test_identical_values(self, identical):
        assert data_range(identical) == 0

    def test_with_nulls(self, with_nulls):
        assert data_range(with_nulls) == 4  # 5 - 1


# ── variance_coefficient ─────────────────────────────────────

class TestCV:
    def test_basic(self):
        s = pl.Series("x", [10, 10, 10, 10])
        assert variance_coefficient(s) == 0.0

    def test_positive(self, simple):
        result = variance_coefficient(simple)
        assert result == standard_deviation(simple) / mean(simple)

    def test_zero_mean_raises(self):
        s = pl.Series("x", [-1, 0, 1])
        with pytest.raises(ArithmeticError):
            variance_coefficient(s)


# ── s_entropy ────────────────────────────────────────────────

class TestEntropy:
    def test_single_category(self):
        s = pl.Series("x", [1, 1, 1, 1])
        assert s_entropy(s) == 0.0

    def test_max_entropy(self):
        s = pl.Series("x", [1, 2, 3, 4])
        expected = math.log(4)
        assert abs(s_entropy(s) - expected) < 1e-10

    def test_two_categories_equal(self):
        s = pl.Series("x", [0, 0, 1, 1])
        expected = math.log(2)
        assert abs(s_entropy(s) - expected) < 1e-10

    def test_entropy_increases_with_diversity(self):
        low = pl.Series("x", [1, 1, 1, 1, 2])
        high = pl.Series("x", [1, 1, 2, 2, 3])
        assert s_entropy(low) < s_entropy(high)


# ── gini ─────────────────────────────────────────────────────

class TestGini:
    def test_perfect_equality(self):
        s = pl.Series("x", [10, 10, 10, 10])
        assert abs(gini(s)) < 1e-10

    def test_high_inequality(self):
        s = pl.Series("x", [0, 0, 0, 100])
        result = gini(s)
        assert result > 0.5

    def test_range(self):
        s = pl.Series("x", [1, 2, 3, 4, 5])
        result = gini(s)
        assert 0 <= result <= 1

    def test_zero_sum_raises(self):
        s = pl.Series("x", [0, 0, 0])
        with pytest.raises(ArithmeticError):
            gini(s)

    def test_inequality_increases(self):
        equal = pl.Series("x", [25, 25, 25, 25])
        unequal = pl.Series("x", [1, 1, 1, 97])
        assert gini(equal) < gini(unequal)