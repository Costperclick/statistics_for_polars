import polars as pl
import random
from polars_stats._utils import _require


def ci_mean_bootstrap(n: pl.Series, confidence: float = 0.95, iterations: int = 10000) -> dict:
    """
    Confidence interval for the mean using bootstrap resampling.

    No assumptions about the distribution shape. The bootstrap simulates
    thousands of alternative samples by drawing with replacement from your
    data, computes the mean of each, and uses the percentiles of those
    means as the confidence interval bounds.

    Use it when:
    - Your data is not normal
    - You don't know the distribution
    - You want a robust, assumption-free interval

    More iterations = more precise bounds (but slower).
    10 000 is a good default for most cases.

    :param n: the series
    :param confidence: confidence level (default 0.95)
    :param iterations: number of bootstrap samples (default 10000)
    :return: dict with lower and upper bounds
    """
    clean = n.drop_nulls()
    means = []
    for i in range(iterations):
        rand = random.choices(clean, k=len(clean))
        means.append(pl.Series(rand).mean())
    sorted_means = sorted(means)
    c = (1 - confidence) / 2
    return {
        "lower": sorted_means[int(len(sorted_means) * c)],
        "upper": sorted_means[int(len(sorted_means) * (1 - c))],
        "confidence": confidence,
        "iterations": iterations,
        "method": "bootstrap",
    }


def ci_median_bootstrap(n: pl.Series, confidence: float = 0.95, iterations: int = 10000) -> dict:
    """
    Confidence interval for the median using bootstrap resampling.

    Same principle as ci_mean_bootstrap but for the median.
    Particularly useful because there is no simple parametric formula
    for a median confidence interval.

    Use it when:
    - You want an interval for the median (not the mean)
    - Your data has outliers (the median is more robust)
    - You don't know the distribution

    :param n: the series
    :param confidence: confidence level (default 0.95)
    :param iterations: number of bootstrap samples (default 10000)
    :return: dict with lower and upper bounds
    """
    clean = n.drop_nulls()
    medians = []
    for i in range(iterations):
        rand = random.choices(clean, k=len(clean))
        medians.append(pl.Series(rand).median())
    sorted_medians = sorted(medians)
    c = (1 - confidence) / 2
    return {
        "lower": sorted_medians[int(len(sorted_medians) * c)],
        "upper": sorted_medians[int(len(sorted_medians) * (1 - c))],
        "confidence": confidence,
        "iterations": iterations,
        "method": "bootstrap",
    }


def ci_mean(n: pl.Series, confidence: float = 0.95) -> dict:
    """
    Parametric confidence interval for the mean using Student's t distribution.

    Formula: mean ± t_critical × (std / √n)

    Assumes the data is approximately normal. For non-normal data,
    use ci_mean_bootstrap() instead.

    Fast and exact when the normality assumption holds. The t distribution
    accounts for the extra uncertainty from estimating the standard deviation
    (unlike a z-based interval which assumes sigma is known).

    :param n: the series
    :param confidence: confidence level (default 0.95)
    :return: dict with lower bound, upper bound, mean, and margin of error
    """
    stats = _require("scipy.stats")
    np = _require("numpy")
    clean = n.drop_nulls().to_numpy()

    n_len = len(clean)
    avg = np.mean(clean)
    std = np.std(clean, ddof=1)
    se = std / np.sqrt(n_len)

    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, df=n_len - 1)
    margin = t_critical * se

    return {
        "lower": float(avg - margin),
        "upper": float(avg + margin),
        "mean": float(avg),
        "margin_of_error": float(margin),
        "confidence": confidence,
        "method": "student_t",
    }


def ci_proportion(successes: int, total: int, confidence: float = 0.95, method: str = "wilson") -> dict:
    """
    Confidence interval for a proportion (e.g. conversion rate, error rate).

    Three methods available:
    - "wilson" (default): best general-purpose method, works well even for
      small samples and extreme proportions. Recommended.
    - "wald": the classic normal approximation (p ± z × √(p(1-p)/n)).
      Simple but unreliable for small n or p close to 0 or 1.
    - "clopper_pearson": exact method based on the binomial distribution.
      Conservative (intervals are wider than necessary) but guaranteed coverage.

    Example: 45 conversions out of 1000 visitors
    → ci_proportion(45, 1000) → interval around 4.5%

    :param successes: number of successes (e.g. conversions)
    :param total: total number of trials (e.g. visitors)
    :param confidence: confidence level (default 0.95)
    :param method: "wilson" (default), "wald", or "clopper_pearson"
    :return: dict with lower bound, upper bound, and observed proportion
    """
    stats = _require("scipy.stats")
    np = _require("numpy")

    p = successes / total
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    if method == "wald":
        se = np.sqrt(p * (1 - p) / total)
        lower = p - z * se
        upper = p + z * se

    elif method == "wilson":
        denominator = 1 + z ** 2 / total
        centre = (p + z ** 2 / (2 * total)) / denominator
        margin = (z / denominator) * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))
        lower = centre - margin
        upper = centre + margin

    elif method == "clopper_pearson":
        lower = stats.beta.ppf(alpha / 2, successes, total - successes + 1)
        upper = stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes)

    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'wilson', 'wald', or 'clopper_pearson'.")

    return {
        "lower": float(max(0, lower)),
        "upper": float(min(1, upper)),
        "proportion": float(p),
        "confidence": confidence,
        "method": method,
    }


def ci_variance(n: pl.Series, confidence: float = 0.95) -> dict:
    """
    Confidence interval for the variance using the chi-squared distribution.

    Formula:
    - lower = (n-1) × s² / χ²_upper
    - upper = (n-1) × s² / χ²_lower

    Assumes the data is normal. The chi-squared distribution describes
    how the sample variance fluctuates around the true variance.

    Use it when:
    - You need to quantify uncertainty on dispersion, not just central tendency
    - Comparing process variability (quality control, manufacturing)
    - Checking if variance has changed between periods

    :param n: the series
    :param confidence: confidence level (default 0.95)
    :return: dict with lower bound, upper bound, and sample variance
    """
    stats = _require("scipy.stats")
    np = _require("numpy")
    clean = n.drop_nulls().to_numpy()

    n_len = len(clean)
    var = np.var(clean, ddof=1)
    df = n_len - 1

    alpha = 1 - confidence
    chi2_lower = stats.chi2.ppf(alpha / 2, df)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)

    return {
        "lower": float(df * var / chi2_upper),
        "upper": float(df * var / chi2_lower),
        "variance": float(var),
        "confidence": confidence,
        "method": "chi_squared",
    }