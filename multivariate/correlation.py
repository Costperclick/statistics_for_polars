import polars as pl
from _utils import _require


def pearson(a: pl.Series, b: pl.Series, detail: bool = False) -> float | dict:
    """
    Pearson correlation coefficient between two variables.

    Measures the strength and direction of the LINEAR relationship.
    Ranges from -1 (perfect inverse) to +1 (perfect direct), 0 = no linear relationship.

    Assumes:
    - Both variables are continuous
    - The relationship is approximately linear
    - Both variables are approximately normal (for the p-value to be valid)

    Interpretation:
    - |r| < 0.3  : weak
    - |r| 0.3-0.7: moderate
    - |r| > 0.7  : strong

    :param a: first series
    :param b: second series
    :param detail: if True, returns r + p-value
    :return: correlation coefficient (float) or dict with r and p-value
    """
    stats = _require("scipy.stats")
    a_clean = a.drop_nulls().to_numpy()
    b_clean = b.drop_nulls().to_numpy()

    r, pvalue = stats.pearsonr(a_clean, b_clean)
    if not detail:
        return float(r)
    else:
        return {
            "r": float(r),
            "pvalue": float(pvalue),
        }


def spearman(a: pl.Series, b: pl.Series, detail: bool = False) -> float | dict:
    """
    Spearman rank correlation between two variables.

    Measures the strength of a MONOTONIC relationship (not just linear).
    Works on ranks — if one variable increases, does the other consistently
    increase (or decrease), even if not at a constant rate?

    Use instead of Pearson when:
    - The relationship is monotonic but not linear
    - Data is ordinal (ratings, rankings)
    - Data has outliers
    - Data is not normal

    :param a: first series
    :param b: second series
    :param detail: if True, returns rho + p-value
    :return: correlation coefficient (float) or dict with rho and p-value
    """
    stats = _require("scipy.stats")
    a_clean = a.drop_nulls().to_numpy()
    b_clean = b.drop_nulls().to_numpy()

    rho, pvalue = stats.spearmanr(a_clean, b_clean)
    if not detail:
        return float(rho)
    else:
        return {
            "rho": float(rho),
            "pvalue": float(pvalue),
        }


def kendall(a: pl.Series, b: pl.Series, detail: bool = False) -> float | dict:
    """
    Kendall's tau correlation between two variables.

    Another rank correlation, more robust than Spearman for:
    - Small samples
    - Many tied values
    - More conservative (less likely to overestimate association)

    Interpretation is similar to Spearman but values are typically smaller
    in magnitude. Don't compare Kendall tau to Pearson r directly.

    :param a: first series
    :param b: second series
    :param detail: if True, returns tau + p-value
    :return: correlation coefficient (float) or dict with tau and p-value
    """
    stats = _require("scipy.stats")
    a_clean = a.drop_nulls().to_numpy()
    b_clean = b.drop_nulls().to_numpy()

    tau, pvalue = stats.kendalltau(a_clean, b_clean)
    if not detail:
        return float(tau)
    else:
        return {
            "tau": float(tau),
            "pvalue": float(pvalue),
        }


def point_biserial(binary: pl.Series, continuous: pl.Series, detail: bool = False) -> float | dict:
    """
    Point-biserial correlation (one binary variable + one continuous variable).

    A special case of Pearson correlation where one variable is binary (0/1).
    Measures the association between group membership and a continuous outcome.

    Example: correlation between gender (0/1) and salary.

    Equivalent to computing the t-test and converting to a correlation.

    :param binary: binary series (0/1 or True/False)
    :param continuous: continuous series
    :param detail: if True, returns r + p-value
    :return: correlation coefficient (float) or dict with r and p-value
    """
    stats = _require("scipy.stats")
    b_clean = binary.drop_nulls().to_numpy()
    c_clean = continuous.drop_nulls().to_numpy()

    r, pvalue = stats.pointbiserialr(b_clean, c_clean)
    if not detail:
        return float(r)
    else:
        return {
            "r": float(r),
            "pvalue": float(pvalue),
        }


def chi2_independence(df: pl.DataFrame, col_a: str, col_b: str, detail: bool = False) -> float | dict:
    """
    Chi-squared test of independence between two categorical variables.

    Tests whether two categorical variables are independent (unrelated)
    or associated. Works by comparing observed frequencies in a contingency
    table to the frequencies expected under independence.

    Example: "Is there an association between subscription plan and churn?"
    → chi2_independence(df, "plan", "churned")

    Assumes:
    - Both variables are categorical
    - Expected frequencies ≥ 5 in each cell (for large-sample approximation)

    :param df: the DataFrame
    :param col_a: first categorical column
    :param col_b: second categorical column
    :param detail: if True, returns chi2, p-value, degrees of freedom, and expected frequencies
    :return: p-value (float) or dict with full results
    """
    stats = _require("scipy.stats")
    np = _require("numpy")

    contingency = df.group_by([col_a, col_b]).len().pivot(
        on=col_b, index=col_a, values="len"
    ).drop(col_a).fill_null(0).to_numpy()

    chi2, pvalue, dof, expected = stats.chi2_contingency(contingency)

    if not detail:
        return float(pvalue)
    else:
        return {
            "chi2": float(chi2),
            "pvalue": float(pvalue),
            "dof": int(dof),
            "expected": expected.tolist(),
        }


def cramers_v(df: pl.DataFrame, col_a: str, col_b: str) -> float:
    """
    Cramér's V — effect size for chi-squared test of independence.

    Measures the strength of association between two categorical variables.
    Ranges from 0 (no association) to 1 (perfect association).

    Interpretation:
    - V ≈ 0.1 : small association
    - V ≈ 0.3 : medium association
    - V ≈ 0.5 : large association

    Always report alongside chi2_independence p-value.

    :param df: the DataFrame
    :param col_a: first categorical column
    :param col_b: second categorical column
    :return: Cramér's V (float between 0 and 1)
    """
    stats = _require("scipy.stats")
    np = _require("numpy")

    contingency = df.group_by([col_a, col_b]).len().pivot(
        on=col_b, index=col_a, values="len"
    ).drop(col_a).fill_null(0).to_numpy()

    chi2, _, _, _ = stats.chi2_contingency(contingency)
    n = contingency.sum()
    min_dim = min(contingency.shape) - 1

    if min_dim == 0:
        return 0.0

    return float(np.sqrt(chi2 / (n * min_dim)))


def mutual_information(a: pl.Series, b: pl.Series, n_bins: int = 10) -> float:
    """
    Mutual information between two continuous variables.

    Measures the amount of information shared between two variables.
    Unlike correlation, it captures ANY kind of relationship (linear,
    non-linear, non-monotonic).

    - 0 = variables are independent (knowing one tells nothing about the other)
    - Higher = more information shared

    Unlike correlation, mutual information has no upper bound and is always ≥ 0.
    It is not directly comparable across different datasets or bin sizes.

    Use it when:
    - You suspect non-linear relationships
    - Feature selection in ML (identifies informative features regardless of relationship shape)
    - Comparing association strength without assuming linearity

    Note: continuous values are discretized into bins for computation.
    The n_bins parameter controls the resolution.

    :param a: first series
    :param b: second series
    :param n_bins: number of bins for discretization (default 10)
    :return: mutual information in nats (float ≥ 0)
    """
    np = _require("numpy")

    a_clean = a.drop_nulls().to_numpy()
    b_clean = b.drop_nulls().to_numpy()

    # Discretize
    a_binned = np.digitize(a_clean, np.linspace(a_clean.min(), a_clean.max(), n_bins))
    b_binned = np.digitize(b_clean, np.linspace(b_clean.min(), b_clean.max(), n_bins))

    # Joint and marginal distributions
    joint_hist = np.histogram2d(a_binned, b_binned, bins=n_bins)[0]
    joint_prob = joint_hist / joint_hist.sum()

    marginal_a = joint_prob.sum(axis=1)
    marginal_b = joint_prob.sum(axis=0)

    # Mutual information
    mi = 0.0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if joint_prob[i, j] > 0 and marginal_a[i] > 0 and marginal_b[j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (marginal_a[i] * marginal_b[j]))

    return float(mi)