import polars as pl
from _utils import _require
import warnings


# ============================================================
# Normality tests
# ============================================================


def which_test(n: pl.Series, objective: str = "normality") -> dict:
    """
    Recommend the appropriate statistical test based on the series
    characteristics (size, normality, symmetry, outliers).

    Supports objectives:
    - "normality" : which normality test to use
    - "location"  : which test to compare the mean/median to a value
    - "outliers"  : which outlier detection method to use

    :param n: the series
    :param objective: what you want to test (default "normality")
    :return: dict with recommended test, reason, and series diagnostics
    """
    clean = n.drop_nulls()
    n_len = len(clean)

    diagnostics = {
        "n": n_len,
        "has_outliers": outliers_iqr(clean)["count"] > 0,
    }

    if objective == "normality":
        if n_len < 20:
            return {
                "recommended": "shapiro_wilk",
                "reason": "Sample too small for D'Agostino (n < 20)",
                **diagnostics,
            }
        elif n_len < 5000:
            return {
                "recommended": "shapiro_wilk",
                "reason": "Best overall normality test for n < 5000",
                "alternative": "dagostino_pearson",
                **diagnostics,
            }
        else:
            return {
                "recommended": "dagostino_pearson",
                "reason": "Shapiro-Wilk unreliable for n > 5000",
                **diagnostics,
            }

    elif objective == "location":
        is_normal = shapiro_wilk(clean) > 0.05 if n_len < 5000 else dagostino_pearson(clean) > 0.05

        diagnostics["is_normal"] = is_normal

        if is_normal:
            return {
                "recommended": "ttest_1samp",
                "reason": "Data appears normal, parametric test is appropriate",
                **diagnostics,
            }
        elif n_len >= 20:
            return {
                "recommended": "wilcoxon_1samp",
                "reason": "Data not normal, using non-parametric test (assumes symmetry)",
                "alternative": "sign_test",
                **diagnostics,
            }
        else:
            return {
                "recommended": "sign_test",
                "reason": "Small non-normal sample, safest non-parametric test",
                **diagnostics,
            }

    elif objective == "outliers":
        if diagnostics["has_outliers"]:
            return {
                "recommended": "outliers_mad",
                "reason": "Outliers detected — MAD-based method is robust to existing outliers",
                "alternative": "outliers_iqr",
                **diagnostics,
            }
        else:
            return {
                "recommended": "outliers_zscore",
                "reason": "No obvious outliers — z-score method is sufficient",
                **diagnostics,
            }

    else:
        raise ValueError(f"Unknown objective: '{objective}'. Use 'normality', 'location', or 'outliers'.")


def shapiro_wilk(n: pl.Series, detail=False) -> float | dict:
    """
    Shapiro-Wilk test for normality.

    The most reliable normality test for small samples (n < 5000).
    Tests if the data comes from a normal distribution by measuring
    how well the ordered values match the expected order statistics
    of a normal distribution.

    Returns the p-value by default. If p < 0.05, reject normality.

    Use it as a first-line normality check before parametric tests
    (t-test, ANOVA, linear regression).

    :param n: the series
    :param detail: if True, returns W statistic + p-value
    :return: p-value (float) or dict with W and p-value
    """
    stats = _require("scipy.stats")
    clean = n.drop_nulls().to_numpy()
    results = stats.shapiro(clean)
    if not detail:
        return results.pvalue
    else:
        return {
            "W": results.statistic,
            "pvalue": results.pvalue,
        }


def dagostino_pearson(n: pl.Series, detail=False) -> float | dict:
    """
    D'Agostino-Pearson K² test for normality.

    Combines skewness and kurtosis into a single test statistic.
    More reliable than Shapiro-Wilk for larger samples (n > 20).
    Requires at least 20 observations.

    Returns the p-value by default. If p < 0.05, reject normality.

    Use it when your dataset is too large for Shapiro-Wilk or when
    you want a test specifically sensitive to asymmetry and tail weight.

    :param n: the series
    :param detail: if True, returns K² statistic + p-value
    :return: p-value (float) or dict with K2 and p-value
    """
    stats = _require("scipy.stats")
    clean = n.drop_nulls().to_numpy()

    if len(clean) < 20:
        warnings.warn(
            "D'Agostino-Pearson requires at least 20 observations. "
            "Consider shapiro_wilk() instead.",
            UserWarning,
        )

    results = stats.normaltest(clean)
    if not detail:
        return results.pvalue
    else:
        return {
            "K2": results.statistic,
            "pvalue": results.pvalue,
        }


def kolmogorov_smirnov(n: pl.Series, dist="norm", detail=False) -> float | dict:
    """
    Kolmogorov-Smirnov one-sample test.

    Unlike Shapiro and D'Agostino, this test works against any theoretical
    distribution — not just the normal. It measures the maximum distance
    between the empirical distribution of the data and the theoretical one.

    By default tests against a normal distribution fitted to the data
    (using the sample mean and std). Pass a different distribution name
    (e.g. "expon", "uniform", "poisson") to test against other distributions.
    See scipy.stats for available distribution names.

    Returns the p-value by default. If p < 0.05, reject the fit.

    Use it when:
    - You want to test against a non-normal distribution
    - You need a general-purpose goodness-of-fit test

    :param n: the series
    :param dist: distribution to test against (default "norm")
    :param detail: if True, returns D statistic + p-value
    :return: p-value (float) or dict with D and p-value
    """
    stats = _require("scipy.stats")
    np = _require("numpy")
    clean = n.drop_nulls().to_numpy()
    results = stats.kstest(clean, dist, args=(np.mean(clean), np.std(clean)))
    if not detail:
        return results.pvalue
    else:
        return {
            "D": results.statistic,
            "pvalue": results.pvalue,
        }


def normality(n: pl.Series, alpha=0.05) -> dict:
    """
    Meta-function: runs multiple normality tests and gives a synthetic verdict.

    Runs Shapiro-Wilk and D'Agostino-Pearson, compares their p-values
    against the significance level alpha, and returns a summary with
    individual results and an overall verdict.

    The verdict is "normal" if all tests fail to reject, "not normal"
    if all reject, and "inconclusive" if they disagree.

    Use it when you want a quick normality diagnostic without choosing
    which test to run.

    :param n: the series
    :param alpha: significance level (default 0.05)
    :return: dict with individual test results and overall verdict
    """
    clean = n.drop_nulls()
    n_len = len(clean)

    results = {}

    # Shapiro-Wilk (reliable for n < 5000)
    if n_len < 5000:
        sw = shapiro_wilk(clean, detail=True)
        results["shapiro_wilk"] = sw
        results["shapiro_wilk"]["reject"] = sw["pvalue"] < alpha

    # D'Agostino-Pearson (needs n >= 20)
    if n_len >= 20:
        dp = dagostino_pearson(clean, detail=True)
        results["dagostino_pearson"] = dp
        results["dagostino_pearson"]["reject"] = dp["pvalue"] < alpha

    # Verdict
    rejections = [v["reject"] for v in results.values()]
    if all(rejections):
        verdict = "not normal"
    elif not any(rejections):
        verdict = "normal"
    else:
        verdict = "inconclusive"

    return {
        "tests": results,
        "verdict": verdict,
        "alpha": alpha,
        "n": n_len,
    }


# ============================================================
# Hypothesis tests (one-sample)
# ============================================================


def ttest_1samp(n: pl.Series, mu: float, detail=False) -> float | dict:
    """
    One-sample t-test.

    Tests whether the mean of the series is significantly different
    from a hypothesized value mu.
    H0: the true mean equals mu.

    Assumes the data is approximately normal (check with normality()).
    For non-normal data, use wilcoxon_1samp() or sign_test().

    Example: "Is the average response time different from 200ms?"
    → ttest_1samp(response_times, mu=200)

    Returns the p-value by default. If p < 0.05, the mean is
    significantly different from mu.

    :param n: the series
    :param mu: the hypothesized mean value
    :param detail: if True, returns t statistic + p-value
    :return: p-value (float) or dict with t and p-value
    """
    stats = _require("scipy.stats")
    clean = n.drop_nulls().to_numpy()
    results = stats.ttest_1samp(clean, mu)
    if not detail:
        return results.pvalue
    else:
        return {
            "t": results.statistic,
            "pvalue": results.pvalue,
        }


def ztest_1samp(n: pl.Series, mu: float, sigma: float = None, detail=False) -> float | dict:
    """
    One-sample z-test.

    Same purpose as ttest_1samp but uses the normal distribution
    instead of Student's t. Use it when:
    - The population standard deviation (sigma) is known, or
    - The sample is large (n > 30), where t and z converge

    In practice, the t-test is almost always preferred. The z-test
    exists mainly for educational purposes and specific cases where
    sigma is truly known (e.g. standardized manufacturing processes).

    :param n: the series
    :param mu: the hypothesized mean value
    :param sigma: known population std. If None, uses sample std.
    :param detail: if True, returns z statistic + p-value
    :return: p-value (float) or dict with z and p-value
    """
    stats = _require("scipy.stats")
    np = _require("numpy")
    clean = n.drop_nulls().to_numpy()

    n_len = len(clean)
    s = sigma if sigma is not None else np.std(clean, ddof=1)
    z = (np.mean(clean) - mu) / (s / np.sqrt(n_len))
    pvalue = 2 * stats.norm.sf(abs(z))

    if not detail:
        return pvalue
    else:
        return {
            "z": z,
            "pvalue": pvalue,
        }


def wilcoxon_1samp(n: pl.Series, mu: float = 0, detail=False) -> float | dict:
    """
    Wilcoxon signed-rank test (one-sample).

    Non-parametric alternative to ttest_1samp. Does not assume normality.
    Tests whether the median of the series is significantly different from mu.
    Assumes the distribution is symmetric around the median.

    Use it when:
    - Your data fails the normality test
    - You have ordinal data
    - You want a more robust test against outliers

    For data that is also not symmetric, use sign_test() instead.

    :param n: the series
    :param mu: the hypothesized median value (default 0)
    :param detail: if True, returns W statistic + p-value
    :return: p-value (float) or dict with W and p-value
    """
    stats = _require("scipy.stats")
    clean = n.drop_nulls().to_numpy()
    results = stats.wilcoxon(clean - mu)
    if not detail:
        return results.pvalue
    else:
        return {
            "W": results.statistic,
            "pvalue": results.pvalue,
        }


def sign_test(n: pl.Series, mu: float = 0, detail=False) -> float | dict:
    """
    Sign test (one-sample).

    The most robust one-sample test. No assumptions about the shape
    of the distribution — simply counts how many values are above vs
    below mu and tests if the split is significantly different from 50/50.

    Less powerful than Wilcoxon (needs more data to detect a real effect)
    but works on any distribution, including asymmetric ones.

    Use it as a last resort when:
    - Data is not normal AND not symmetric
    - You have very small samples with unknown distribution
    - You need the safest possible test

    :param n: the series
    :param mu: the hypothesized median value (default 0)
    :param detail: if True, returns n_above, n_below + p-value
    :return: p-value (float) or dict with counts and p-value
    """
    stats = _require("scipy.stats")
    clean = n.drop_nulls().to_numpy()

    differences = clean - mu
    n_positive = int((differences > 0).sum())
    n_negative = int((differences < 0).sum())
    n_nonzero = n_positive + n_negative

    results = stats.binomtest(n_positive, n_nonzero, 0.5)

    if not detail:
        return results.pvalue
    else:
        return {
            "n_above": n_positive,
            "n_below": n_negative,
            "n_zero": len(clean) - n_nonzero,
            "pvalue": results.pvalue,
        }


def binomial_test(n: pl.Series, p0: float = 0.5, detail=False) -> float | dict:
    """
    Exact binomial test.

    Tests whether the proportion of successes in a binary series
    is significantly different from a hypothesized proportion p0.
    H0: the true proportion equals p0.

    The series must contain binary values (0/1, True/False).

    Example: "Is the conversion rate significantly different from 5%?"
    → binomial_test(conversions, p0=0.05)

    :param n: the series (binary: 0/1 or True/False)
    :param p0: the hypothesized proportion (default 0.5)
    :param detail: if True, returns observed proportion, k, n + p-value
    :return: p-value (float) or dict with details and p-value
    """
    stats = _require("scipy.stats")
    clean = n.drop_nulls().to_numpy()

    k = int(clean.sum())
    n_len = len(clean)
    results = stats.binomtest(k, n_len, p0)

    if not detail:
        return results.pvalue
    else:
        return {
            "k": k,
            "n": n_len,
            "observed_proportion": k / n_len,
            "expected_proportion": p0,
            "pvalue": results.pvalue,
        }


# ============================================================
# Outlier detection
# ============================================================


def outliers_zscore(n: pl.Series, threshold: float = 3.0) -> dict:
    """
    Detect outliers using the z-score method.

    Marks any value more than `threshold` standard deviations away
    from the mean as an outlier. Default threshold is 3 (99.7% of
    a normal distribution falls within ±3σ).

    Simple and fast, but sensitive to the outliers themselves — extreme
    values inflate the mean and std, potentially masking other outliers.
    For a robust alternative, use outliers_mad().

    :param n: the series
    :param threshold: number of standard deviations (default 3.0)
    :return: dict with outlier values, indices, and count
    """
    clean = n.drop_nulls()
    avg = clean.mean()
    std = clean.std()

    outlier_indices = []
    outlier_values = []
    for i, x in enumerate(clean):
        z = abs(x - avg) / std
        if z > threshold:
            outlier_indices.append(i)
            outlier_values.append(x)

    return {
        "indices": outlier_indices,
        "values": outlier_values,
        "count": len(outlier_values),
        "threshold": threshold,
        "mean": avg,
        "std": std,
    }


def outliers_mad(n: pl.Series, threshold: float = 3.5) -> dict:
    """
    Detect outliers using the MAD-based modified z-score.

    Same principle as outliers_zscore but uses the median and MAD
    instead of mean and std. This makes it robust — the outliers
    themselves don't influence the detection thresholds.

    The modified z-score is: 0.6745 * (xi - median) / MAD
    The constant 0.6745 makes it comparable to standard z-scores.
    Default threshold is 3.5 (recommended by Iglewicz & Hoaglin).

    Use it when:
    - Your data has heavy tails or known outliers
    - outliers_zscore misses outliers because they inflated the std
    - You want a robust first-pass detection

    :param n: the series
    :param threshold: modified z-score threshold (default 3.5)
    :return: dict with outlier values, indices, and count
    """
    clean = n.drop_nulls()
    med = clean.median()
    mad = (clean - med).abs().median()

    if mad == 0:
        return {
            "indices": [],
            "values": [],
            "count": 0,
            "threshold": threshold,
            "median": med,
            "mad": 0,
            "note": "MAD is zero — most values are identical",
        }

    outlier_indices = []
    outlier_values = []
    for i, x in enumerate(clean):
        modified_z = 0.6745 * abs(x - med) / mad
        if modified_z > threshold:
            outlier_indices.append(i)
            outlier_values.append(x)

    return {
        "indices": outlier_indices,
        "values": outlier_values,
        "count": len(outlier_values),
        "threshold": threshold,
        "median": med,
        "mad": mad,
    }


def outliers_iqr(n: pl.Series, k: float = 1.5) -> dict:
    """
    Detect outliers using the IQR (Tukey fence) method.

    A value is an outlier if it falls below Q1 - k×IQR or above
    Q3 + k×IQR. This is the method used by boxplots.

    - k=1.5 (default) detects "mild" outliers
    - k=3.0 detects "extreme" outliers

    Non-parametric: no assumptions about the distribution shape.
    Works well for skewed data where zscore methods fail.

    :param n: the series
    :param k: fence multiplier (default 1.5)
    :return: dict with outlier values, indices, bounds, and count
    """
    clean = n.drop_nulls()
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr_val = q3 - q1

    lower = q1 - k * iqr_val
    upper = q3 + k * iqr_val

    outlier_indices = []
    outlier_values = []
    for i, x in enumerate(clean):
        if x < lower or x > upper:
            outlier_indices.append(i)
            outlier_values.append(x)

    return {
        "indices": outlier_indices,
        "values": outlier_values,
        "count": len(outlier_values),
        "lower_fence": lower,
        "upper_fence": upper,
        "q1": q1,
        "q3": q3,
        "iqr": iqr_val,
    }


def outliers_grubbs(n: pl.Series, alpha: float = 0.05) -> dict:
    """
    Grubbs' test for outliers.

    A formal statistical test that detects one outlier at a time.
    Tests the most extreme value (furthest from mean) and returns
    whether it is a significant outlier at the given alpha level.

    Run iteratively: remove the detected outlier and test again
    until no more outliers are found.

    Assumes the data (without outliers) is approximately normal.

    :param n: the series
    :param alpha: significance level (default 0.05)
    :return: dict with the suspected outlier, test statistic, critical value,
             and whether it is a significant outlier
    """
    stats = _require("scipy.stats")
    np = _require("numpy")
    clean = n.drop_nulls().to_numpy()

    n_len = len(clean)
    avg = np.mean(clean)
    std = np.std(clean, ddof=1)

    abs_deviations = np.abs(clean - avg)
    max_idx = int(np.argmax(abs_deviations))
    suspect = clean[max_idx]
    G = abs(suspect - avg) / std

    t_crit = stats.t.ppf(1 - alpha / (2 * n_len), n_len - 2)
    G_crit = ((n_len - 1) / np.sqrt(n_len)) * np.sqrt(t_crit ** 2 / (n_len - 2 + t_crit ** 2))

    return {
        "suspect_value": suspect,
        "suspect_index": max_idx,
        "G": G,
        "G_critical": G_crit,
        "is_outlier": G > G_crit,
        "alpha": alpha,
    }


# ============================================================
# Effect size
# ============================================================


def cohens_d(n: pl.Series, mu: float) -> float:
    """
    Cohen's d for a one-sample comparison.

    Measures the distance between the sample mean and a hypothesized
    value mu, expressed in standard deviation units. It tells you
    not just whether the difference is significant (that's the t-test)
    but how *large* the effect is.

    Interpretation:
    - |d| ≈ 0.2 : small effect
    - |d| ≈ 0.5 : medium effect
    - |d| ≈ 0.8 : large effect

    Always report effect size alongside p-values. A tiny effect can
    be "significant" with enough data, and a large effect can be
    "non-significant" with too little data.

    Example: "Response time mean is 215ms vs target 200ms.
    Is that a meaningful difference or just noise?"
    → cohens_d(response_times, mu=200)

    :param n: the series
    :param mu: the reference value to compare against
    :return: Cohen's d (float)
    """
    clean = n.drop_nulls()
    avg = clean.mean()
    std = clean.std()

    if std == 0:
        raise ArithmeticError("Standard deviation is zero, cannot compute Cohen's d")

    return (avg - mu) / std
