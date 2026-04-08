import polars as pl
from polars_stats._utils import _require

def hotelling_t2(df: pl.DataFrame, features: list, group: str, detail: bool = False) -> float | dict:
    """
    Hotelling's T² test — multivariate generalization of the t-test.

    Tests whether two groups have significantly different mean VECTORS.
    A regular t-test compares one variable at a time. Hotelling T² compares
    all variables simultaneously, accounting for correlations between them.

    Example: "Do male and female patients differ across blood pressure,
    cholesterol, AND heart rate simultaneously?"
    A series of t-tests might miss a pattern that emerges only when
    looking at all variables together.

    Assumes:
    - Multivariate normality in both groups
    - Equal covariance matrices (check with box_m())

    :param df: the DataFrame
    :param features: list of feature column names to compare
    :param group: column name containing the group labels (must have exactly 2 groups)
    :param detail: if True, returns T², F statistic, and p-value
    :return: p-value (float) or dict with full results
    """
    np = _require("numpy")
    stats = _require("scipy.stats")

    groups = df[group].unique().to_list()
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}: {groups}")

    data_a = df.filter(pl.col(group) == groups[0]).select(features).drop_nulls().to_numpy()
    data_b = df.filter(pl.col(group) == groups[1]).select(features).drop_nulls().to_numpy()

    n_a = data_a.shape[0]
    n_b = data_b.shape[0]
    p = len(features)

    mean_a = np.mean(data_a, axis=0)
    mean_b = np.mean(data_b, axis=0)
    diff = mean_a - mean_b

    # Pooled covariance matrix
    cov_a = np.cov(data_a, rowvar=False)
    cov_b = np.cov(data_b, rowvar=False)
    cov_pooled = ((n_a - 1) * cov_a + (n_b - 1) * cov_b) / (n_a + n_b - 2)

    # T² statistic
    t2 = (n_a * n_b) / (n_a + n_b) * diff @ np.linalg.inv(cov_pooled) @ diff

    # Convert to F distribution
    f_stat = t2 * (n_a + n_b - p - 1) / (p * (n_a + n_b - 2))
    df1 = p
    df2 = n_a + n_b - p - 1
    pvalue = 1 - stats.f.cdf(f_stat, df1, df2)

    if not detail:
        return float(pvalue)
    else:
        return {
            "T2": float(t2),
            "F": float(f_stat),
            "df1": df1,
            "df2": df2,
            "pvalue": float(pvalue),
            "group_a": groups[0],
            "group_b": groups[1],
            "mean_a": mean_a.tolist(),
            "mean_b": mean_b.tolist(),
        }


def mahalanobis(df: pl.DataFrame, features: list = None) -> pl.Series:
    """
    Mahalanobis distance for each observation — multivariate outlier detection.

    The Mahalanobis distance measures how far each point is from the center
    of the data cloud, taking into account the shape (correlations) of the
    distribution. Unlike Euclidean distance, it accounts for the fact that
    some directions have more spread than others.

    A point can be an outlier in multivariate space even if it looks normal
    on each variable individually. Mahalanobis catches these cases.

    Interpretation:
    - For p features, the squared Mahalanobis distance follows approximately
      a chi-squared distribution with p degrees of freedom.
    - Values above chi2.ppf(0.975, p) can be considered outliers.

    Returns a Series of Mahalanobis distances (same length as the DataFrame).

    :param df: the DataFrame
    :param features: list of feature column names (default: all numeric columns)
    :return: pl.Series of Mahalanobis distances
    """
    np = _require("numpy")

    if features is None:
        features = df.select(pl.col(pl.NUMERIC_DTYPES)).columns

    data = df.select(features).drop_nulls().to_numpy()

    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    cov_inv = np.linalg.inv(cov)

    distances = []
    for row in data:
        diff = row - mean
        d = float(np.sqrt(diff @ cov_inv @ diff))
        distances.append(d)

    return pl.Series("mahalanobis", distances)


def box_m(df: pl.DataFrame, features: list, group: str) -> dict:
    """
    Box's M test — tests equality of covariance matrices across groups.

    This is an assumption of Hotelling T² and discriminant analysis.
    If Box M rejects (p < 0.05), the covariance structures differ
    between groups, and Hotelling T² results may be unreliable.

    Box's M is known to be very sensitive — it often rejects with large
    samples even when the differences are practically negligible.
    Consider it a warning, not a hard veto.

    :param df: the DataFrame
    :param features: list of feature column names
    :param group: column name containing the group labels
    :return: dict with M statistic, chi-squared approximation, and p-value
    """
    np = _require("numpy")
    stats = _require("scipy.stats")

    groups = df[group].unique().to_list()
    k = len(groups)
    p = len(features)

    samples = []
    cov_matrices = []
    ns = []

    for g in groups:
        data = df.filter(pl.col(group) == g).select(features).drop_nulls().to_numpy()
        samples.append(data)
        cov_matrices.append(np.cov(data, rowvar=False))
        ns.append(data.shape[0])

    n_total = sum(ns)

    # Pooled covariance
    cov_pooled = sum((n - 1) * cov for n, cov in zip(ns, cov_matrices)) / (n_total - k)

    # Box's M statistic
    log_det_pooled = np.linalg.slogdet(cov_pooled)[1]
    M = (n_total - k) * log_det_pooled - sum(
        (n - 1) * np.linalg.slogdet(cov)[1]
        for n, cov in zip(ns, cov_matrices)
    )

    # Chi-squared approximation
    c = (sum(1 / (n - 1) for n in ns) - 1 / (n_total - k)) * (2 * p ** 2 + 3 * p - 1) / (6 * (p + 1) * (k - 1))
    df_val = p * (p + 1) * (k - 1) / 2
    chi2 = (1 - c) * M
    pvalue = 1 - stats.chi2.cdf(chi2, df_val)

    return {
        "M": float(M),
        "chi2": float(chi2),
        "df": int(df_val),
        "pvalue": float(pvalue),
        "groups": groups,
        "equal_covariance": pvalue >= 0.05,
    }