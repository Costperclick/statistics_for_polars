import polars as pl
from polars_stats._utils import _require


def mean(df: pl.DataFrame, columns: list = None) -> dict:
    """
    Calculate the mean of each column (the centroid of the data).

    In multivariate analysis, the mean is a vector — one value per variable.
    This vector represents the center of the data cloud in n-dimensional space.

    Used as a building block for many multivariate methods (Mahalanobis distance,
    Hotelling T², PCA centering).

    :param df: the DataFrame
    :param columns: list of column names (default: all columns)
    :return: dict mapping column names to their mean
    """
    cols = columns or df.columns
    output = {}
    for col in cols:
        output[col] = df[col].mean()
    return output


def cross_summary(df: pl.DataFrame, columns: list = None) -> pl.DataFrame:
    """
    Descriptive summary across multiple columns.

    Returns count, null_count, mean, std, min, q25, median, q75, max
    for each selected column in a single DataFrame. This is the multivariate
    equivalent of calling describe() on each column.

    Use it as a first step when exploring a new dataset — one glance tells you
    the scale, spread, and completeness of every variable.

    :param df: the DataFrame
    :param columns: list of column names (default: all numeric columns)
    :return: DataFrame with summary statistics per column
    """
    if columns:
        subset = df.select(columns)
    else:
        subset = df.select(pl.col(pl.NUMERIC_DTYPES))

    return subset.describe()


def correlation_matrix(df: pl.DataFrame, columns: list = None, method: str = "pearson") -> pl.DataFrame:
    """
    Compute the correlation matrix between multiple columns.

    Correlation measures the strength and direction of the linear (Pearson)
    or monotonic (Spearman, Kendall) relationship between pairs of variables.
    Values range from -1 (perfect inverse) to +1 (perfect direct), 0 = no relationship.

    Methods:
    - "pearson" (default): linear correlation. Fast, assumes normality.
      Use when relationships are roughly linear.
    - "spearman": rank correlation. Robust to outliers and non-linear monotonic
      relationships. Use when data is skewed or ordinal.
    - "kendall": rank correlation. More robust than Spearman for small samples
      and tied values. Slower on large datasets.

    Interpretation guidelines:
    - |r| < 0.3  : weak
    - |r| 0.3-0.7: moderate
    - |r| > 0.7  : strong

    :param df: the DataFrame
    :param columns: list of column names (default: all numeric columns)
    :param method: "pearson", "spearman", or "kendall" (default "pearson")
    :return: DataFrame correlation matrix
    """
    np = _require("numpy")

    if columns:
        subset = df.select(columns)
    else:
        subset = df.select(pl.col(pl.NUMERIC_DTYPES))

    cols = subset.columns
    data = subset.drop_nulls().to_numpy()

    if method == "pearson":
        corr = np.corrcoef(data, rowvar=False)
    elif method == "spearman":
        stats = _require("scipy.stats")
        corr, _ = stats.spearmanr(data)
        if len(cols) == 2:
            corr = np.array([[1, corr], [corr, 1]])
    elif method == "kendall":
        stats = _require("scipy.stats")
        n_cols = len(cols)
        corr = np.ones((n_cols, n_cols))
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                tau, _ = stats.kendalltau(data[:, i], data[:, j])
                corr[i, j] = tau
                corr[j, i] = tau
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'pearson', 'spearman', or 'kendall'.")

    result = pl.DataFrame({
        cols[i]: corr[:, i] for i in range(len(cols))
    })

    return result.with_columns(pl.Series("column", cols)).select(["column"] + cols)


def covariance_matrix(df: pl.DataFrame, columns: list = None) -> pl.DataFrame:
    """
    Compute the variance-covariance matrix.

    Covariance measures how two variables move together:
    - Positive: when one increases, the other tends to increase
    - Negative: when one increases, the other tends to decrease
    - Zero: no linear relationship

    Unlike correlation, covariance is not normalized — its magnitude depends
    on the scale of the variables. Use correlation_matrix() for comparisons,
    covariance_matrix() when you need the raw values (e.g. for Mahalanobis
    distance, PCA, or portfolio variance).

    :param df: the DataFrame
    :param columns: list of column names (default: all numeric columns)
    :return: DataFrame covariance matrix
    """
    np = _require("numpy")

    if columns:
        subset = df.select(columns)
    else:
        subset = df.select(pl.col(pl.NUMERIC_DTYPES))

    cols = subset.columns
    data = subset.drop_nulls().to_numpy()
    cov = np.cov(data, rowvar=False)

    result = pl.DataFrame({
        cols[i]: cov[:, i] for i in range(len(cols))
    })

    return result.with_columns(pl.Series("column", cols)).select(["column"] + cols)


def partial_correlation(df: pl.DataFrame, x: str, y: str, controls: list) -> dict:
    """
    Compute the partial correlation between x and y, controlling for other variables.

    Regular correlation between revenue and ice cream sales might be 0.8.
    But both are driven by temperature. Partial correlation removes the effect
    of temperature, revealing the true direct relationship between revenue
    and ice cream sales.

    Use it to:
    - Disentangle confounded relationships
    - Find direct vs indirect associations
    - Identify spurious correlations driven by a third variable

    :param df: the DataFrame
    :param x: first variable name
    :param y: second variable name
    :param controls: list of variable names to control for
    :return: dict with partial correlation, p-value, and controlled variables
    """
    np = _require("numpy")
    stats = _require("scipy.stats")

    all_cols = [x, y] + controls
    data = df.select(all_cols).drop_nulls().to_numpy()

    n = data.shape[0]
    k = len(controls)

    # Residualize x and y against control variables
    controls_data = data[:, 2:]
    x_data = data[:, 0]
    y_data = data[:, 1]

    # Add intercept
    controls_with_intercept = np.column_stack([np.ones(n), controls_data])

    # Compute residuals via OLS: residual = value - predicted
    beta_x = np.linalg.lstsq(controls_with_intercept, x_data, rcond=None)[0]
    beta_y = np.linalg.lstsq(controls_with_intercept, y_data, rcond=None)[0]

    resid_x = x_data - controls_with_intercept @ beta_x
    resid_y = y_data - controls_with_intercept @ beta_y

    # Correlation of residuals
    r, _ = stats.pearsonr(resid_x, resid_y)

    # Compute p-value using t-distribution
    df_val = n - k - 2
    t_stat = r * np.sqrt(df_val / (1 - r ** 2))
    pvalue = 2 * stats.t.sf(abs(t_stat), df_val)

    return {
        "partial_r": float(r),
        "pvalue": float(pvalue),
        "x": x,
        "y": y,
        "controls": controls,
        "n": n,
    }