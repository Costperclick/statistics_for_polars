import math
from typing import Any
import polars as pl


def mean(n: pl.Series, fast=True) -> Any:
    """
    Calculate the arithmetic mean of a series.

    The arithmetic mean is the sum of all values divided by the number of values.
    Use it when your data is additive (revenues, temperatures, durations).
    Sensitive to outliers — consider median() for skewed distributions.

    Formula: x̄ = (1/n) × Σxi

    :param n: the series
    :param fast: Default is True. Uses the built-in mean.
    :return: the mean value (float)
    """
    if not fast:
        n_sum = n.sum()
        n_length = len(n)
        return float(n_sum / n_length)
    else:
        return n.mean()


def median(n: pl.Series, fast=True) -> Any:
    """
    Calculate the median of a series.

    The median is the middle value when the data is sorted.
    Unlike the mean, it is robust to outliers — a single extreme value
    won't affect the result. Use it when your distribution is skewed
    (e.g. salaries, house prices, session durations).

    :param n: the series
    :param fast: Default is True. Uses the built-in median.
    :return: the median value (float)
    """
    if not fast:
        clean = n.drop_nulls()
        n_len = len(clean)
        x = sorted(clean)
        if n_len % 2 == 0:
            return (x[n_len // 2] + x[n_len // 2 - 1]) / 2
        else:
            return x[n_len // 2]
    else:
        return n.median()


def weighted_mean(n: pl.Series, weights: pl.Series) -> Any:
    """
    Calculate the weighted mean of a series.

    Each value is multiplied by its associated weight before averaging.
    Use it when some observations matter more than others:
    - Averaging grades with different coefficients
    - Computing a portfolio return (weighted by asset allocation)
    - Aggregating survey results with sampling weights

    Null weights or null values cause the (value, weight) pair to be ignored.
    Raises if the sum of weights is zero.

    Formula: x̄w = Σ(wi × xi) / Σwi

    :param n: the series of values
    :param weights: the series of weights (same length as n)
    :return: the weighted mean value (float)
    """
    if len(n) != len(weights):
        raise ArithmeticError("Series and weights must have the same length")

    sigma = 0
    w_sum = 0
    for i, x in enumerate(n):
        if x is None or weights[i] is None:
            continue
        sigma += weights[i] * x
        w_sum += weights[i]

    if w_sum == 0:
        raise ArithmeticError("Sum of weights is zero")

    return sigma / w_sum


def geometric_mean(n: pl.Series, log_transform=True) -> Any:
    """
    Calculate the geometric mean of a series.

    The geometric mean is the nth root of the product of all values.
    Use it when your data is multiplicative — values that compound
    or interact as ratios:
    - Average growth rate over multiple periods
    - Average investment return across years
    - Averaging ratios or indices

    Example: returns of +10%, -20%, +30% → geometric_mean([1.1, 0.8, 1.3])
    gives the true average multiplier per period.

    All values must be strictly positive. Null values are ignored.

    Formula: x̄g = exp((1/n) × Σln(xi))

    :param n: the series (strictly positive values)
    :param log_transform: Default is True. Uses log transformation to avoid
        overflow on large series. Set to False for the naive product approach.
    :return: the geometric mean (float)
    """
    clean = n.drop_nulls()
    if clean.min() <= 0:
        raise ArithmeticError("Series must contain strictly positive values")

    if not log_transform:
        product = 1
        for x in clean:
            product *= x

        return product ** (1 / len(clean))
    else:
        return math.exp(clean.log().mean())


def harmonic_mean(n: pl.Series) -> Any:
    """
    Calculate the harmonic mean of a series.

    The harmonic mean is the reciprocal of the arithmetic mean of reciprocals.
    Use it when averaging rates or ratios where the numerator is fixed:
    - Average speed over a fixed distance (e.g. 60 km/h out, 40 km/h back → 48 km/h, not 50)
    - Average price per unit when spending a fixed budget
    - The F1-score in ML is the harmonic mean of precision and recall

    Always less than or equal to the arithmetic mean. All values must be
    strictly positive. Null values are ignored.

    Formula: x̄h = n / Σ(1/xi)

    :param n: the series (strictly positive values)
    :return: the harmonic mean (float)
    """
    clean = n.drop_nulls()
    if clean.min() <= 0:
        raise ArithmeticError("Series must contain strictly positive values")

    length = len(clean)
    sigma = 0

    for x in clean:
        sigma += 1 / x

    return length / sigma


def mode(n: pl.Series) -> list:
    """
    Find the mode(s) of a series — the value(s) with the highest frequency.

    Unlike mean and median, the mode works on categorical data too
    (e.g. most popular product, most common error code).
    Can return multiple values if the series is multimodal.

    Use cases:
    - Finding the most common category in a column
    - Detecting dominant patterns in discrete data
    - Imputing missing categorical values with the most frequent one

    Raises if all values have the same frequency (no meaningful mode).
    Null values are ignored.

    :param n: the series
    :return: list of mode value(s)
    """
    clean = n.drop_nulls()
    count = {}
    for x in clean:
        count[x] = count.get(x, 0) + 1

    max_count = max(count.values())
    output = [k for k, v in count.items() if v == max_count]

    if len(output) == len(clean):
        raise ArithmeticError("All values have the same frequency")
    else:
        return output


def variance(n: pl.Series) -> Any:
    """
    Calculate the sample variance of a series.

    Variance measures how spread out the data is around the mean.
    A low variance means values are clustered near the mean;
    a high variance means they are spread out.

    Uses Bessel's correction (n-1 denominator) for unbiased estimation
    from a sample. Null values are ignored.

    Use standard_deviation() for a result in the same unit as the data —
    variance is in squared units, which is harder to interpret directly.

    Formula: s² = Σ(xi - x̄)² / (n - 1)

    :param n: the series
    :return: the sample variance (float)
    """
    clean = n.drop_nulls()
    avg = mean(clean)
    deltas = []

    for x in clean:
        deltas.append((x - avg) ** 2)

    return sum(deltas) / (len(clean) - 1)


def standard_deviation(n: pl.Series) -> Any:
    """
    Calculate the sample standard deviation of a series.

    The standard deviation is the square root of the variance.
    It measures dispersion in the same unit as the original data,
    making it easier to interpret than variance.

    Rule of thumb for normally distributed data:
    - ~68% of values fall within ±1 std of the mean
    - ~95% within ±2 std
    - ~99.7% within ±3 std

    Null values are ignored.

    Formula: s = √(variance)

    :param n: the series
    :return: the sample standard deviation (float)
    """
    return math.sqrt(variance(n))


def median_absolute_deviation(n: pl.Series) -> Any:
    """
    Calculate the Median Absolute Deviation (MAD) of a series.

    MAD is a robust measure of dispersion — the median of absolute
    deviations from the median. Unlike standard deviation, it is
    virtually immune to outliers.

    Use it when:
    - Your data contains outliers or extreme values
    - You need a robust alternative to standard deviation
    - You want to detect outliers (values beyond k × MAD from the median)

    Example: [1, 2, 3, 4, 100] → std is inflated by 100, MAD is not.

    Null values are ignored.

    Formula: MAD = median(|xi - median(x)|)

    :param n: the series
    :return: the MAD value (float)
    """
    clean = n.drop_nulls()
    med = median(clean)
    deltas = []
    for x in clean:
        deltas.append(abs(x - med))

    return median(pl.Series(deltas))


def skewness(n: pl.Series) -> Any:
    """
    Calculate the skewness (Fisher's coefficient of asymmetry) of a series.

    Skewness measures how asymmetric the distribution is:
    - = 0 : symmetric (like a normal distribution)
    - > 0 : right-skewed, tail extends to the right (many small values, few large ones)
    - < 0 : left-skewed, tail extends to the left (many large values, few small ones)

    Rules of thumb:
    - Between -0.5 and 0.5 : approximately symmetric
    - Beyond ±1 : significantly skewed

    Use it to:
    - Decide between mean and median as central tendency measure
    - Check assumptions before parametric tests (t-test, ANOVA)
    - Detect asymmetric patterns in financial returns, response times, etc.

    Null values are ignored.

    Formula: g1 = (1/n × Σ(xi - x̄)³) / s³

    :param n: the series
    :return: the skewness coefficient (float)
    """
    clean = n.drop_nulls()

    avg = mean(clean)
    std = standard_deviation(clean)
    n_len = len(clean)
    deltas = []
    for x in clean:
        deltas.append((x - avg) ** 3)

    return (sum(deltas) / n_len) / (std ** 3)


def kurtosis(n: pl.Series) -> Any:
    """
    Calculate the excess kurtosis of a series.

    Kurtosis measures the heaviness of the tails of a distribution —
    how likely extreme values (outliers) are compared to a normal distribution:
    - = 0 : tails similar to a normal distribution
    - > 0 : heavy tails, extreme events more frequent than expected (leptokurtic)
    - < 0 : light tails, data is tightly concentrated (platykurtic)

    This is excess kurtosis (subtract 3 from raw kurtosis) so that the
    normal distribution gives 0.

    Use it in:
    - Risk assessment: high kurtosis in financial returns means more "black swan" events
    - Normality checks: kurtosis far from 0 suggests non-normal data
    - Quality control: unexpected kurtosis changes can signal process shifts

    Null values are ignored.

    Formula: g2 = ((1/n × Σ(xi - x̄)⁴) / s⁴) - 3

    :param n: the series
    :return: the excess kurtosis coefficient (float)
    """
    clean = n.drop_nulls()

    avg = mean(clean)
    std = standard_deviation(clean)
    n_len = len(clean)
    deltas = []
    for x in clean:
        deltas.append((x - avg) ** 4)

    return ((sum(deltas) / n_len) / (std ** 4)) - 3


def quantile(n: pl.Series, q: float) -> Any:
    """
    Calculate an arbitrary quantile of a series.

    The quantile at q is the value below which q×100% of the data falls.
    Common quantiles:
    - q=0.25 → first quartile (Q1)
    - q=0.50 → median (Q2)
    - q=0.75 → third quartile (Q3)
    - q=0.90 → 90th percentile

    Use it to understand the distribution shape, set thresholds,
    or define SLA targets (e.g. "p99 response time under 500ms").

    Null values are ignored by Polars.

    :param n: the series
    :param q: quantile level, must be between 0 and 1
    :return: the quantile value (float)
    """
    if not 0 <= q <= 1:
        raise ArithmeticError("q must be between 0 and 1")

    return n.quantile(q)


def iqr(n: pl.Series) -> Any:
    """
    Calculate the Interquartile Range (IQR) of a series.

    IQR = Q3 - Q1, the range of the middle 50% of the data.
    It measures dispersion without being affected by extreme values.

    Commonly used for:
    - Outlier detection: values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR
      are considered outliers (the "Tukey fence" method, used in boxplots)
    - Comparing spread between groups without outlier sensitivity

    :param n: the series
    :return: the IQR value (float)
    """
    q_one = quantile(n, 0.25)
    q_three = quantile(n, 0.75)

    return q_three - q_one


def data_range(n: pl.Series) -> Any:
    """
    Calculate the range of a series (max - min).

    The simplest measure of spread. Highly sensitive to outliers since it
    only depends on the two most extreme values. Prefer IQR or standard
    deviation for robust dispersion measurement.

    Useful for quick sanity checks:
    - Detecting data entry errors (e.g. an age of 999)
    - Understanding the scale of a variable at a glance

    Null values are ignored.

    :param n: the series
    :return: the range (max - min)
    """
    clean = n.drop_nulls()

    return clean.max() - clean.min()


def variance_coefficient(n: pl.Series) -> Any:
    """
    Calculate the Coefficient of Variation (CV) of a series.

    CV = std / mean, expressed as a unitless ratio. It measures relative
    variability — how large the standard deviation is compared to the mean.

    Use it to compare dispersion between series with different scales or units:
    - Revenue in euros (mean=10M, std=2M) → CV = 0.2
    - Number of orders (mean=500, std=100) → CV = 0.2
    - Same relative variability despite very different scales.

    Raises if the mean is zero (division by zero).
    Null values are ignored.

    Formula: CV = s / x̄

    :param n: the series
    :return: the coefficient of variation (float)
    """
    clean = n.drop_nulls()
    std = standard_deviation(clean)
    avg = mean(clean)

    if avg == 0:
        raise ArithmeticError("The mean value of the series can't be 0")
    else:
        return std / avg


def s_entropy(n: pl.Series) -> Any:
    """
    Calculate the Shannon Entropy of a series.

    Shannon entropy measures the level of "surprise" or diversity in the data.
    High entropy = values are spread evenly across many categories.
    Low entropy = values are concentrated on a few categories.

    Use it to:
    - Measure diversity of a categorical column (products, error codes, user types)
    - Feature selection in ML: a column with entropy ≈ 0 carries no information
    - Anomaly detection: a sudden change in entropy signals a distributional shift
    - Compare how "balanced" different categorical distributions are

    Example:
    - [A, A, A, A] → entropy = 0 (perfectly predictable)
    - [A, B, C, D] → entropy = ln(4) ≈ 1.39 (maximum surprise)

    Null values are ignored.

    Formula: H = -Σ(pi × ln(pi)) where pi is the relative frequency of each distinct value

    :param n: the series
    :return: the Shannon entropy (float, ≥ 0)
    """
    clean = n.drop_nulls()

    n_len = len(clean)
    counts = {}
    entropy = []

    for x in clean:
        counts[x] = counts.get(x, 0) + 1

    for _, v in counts.items():
        entropy.append((v / n_len) * math.log(v / n_len))

    return -sum(entropy)


def gini(n: pl.Series) -> Any:
    """
    Calculate the Gini coefficient of a series.

    The Gini coefficient measures inequality in the distribution of values:
    - 0 = perfect equality (all values are identical)
    - 1 = maximum inequality (one value holds everything)

    Originally used to measure income inequality between countries
    (France ≈ 0.32, USA ≈ 0.39), but applicable to any concentration analysis:
    - Revenue by client: high Gini → a few clients drive all revenue
    - Traffic by page: high Gini → a few pages get all the visits
    - Sales by product: high Gini → long tail, most products sell little
    - Server load: high Gini → unbalanced distribution across nodes

    All values must be non-negative. Raises if the sum is zero.
    Null values are ignored.

    Formula: G = (2 × Σ(i × x(i))) / (n × Σx(i)) - (n+1)/n
    where x(i) are values sorted in ascending order and i is the rank (1 to n).

    :param n: the series (non-negative values)
    :return: the Gini coefficient (float, between 0 and 1)
    """
    clean = n.drop_nulls()

    n_sorted = sorted(clean)
    n_length = len(n_sorted)

    if sum(n_sorted) == 0:
        raise ArithmeticError("Sum of the series can't be zero")

    numerator = 0
    for i, x in enumerate(n_sorted, 1):
        numerator += (x * i)

    denominator = len(n_sorted) * sum(n_sorted)

    fraction = (numerator * 2) / denominator
    correction = (n_length + 1) / n_length

    return fraction - correction