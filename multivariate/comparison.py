import polars as pl
from _utils import _require


# ============================================================
# Two-group comparisons
# ============================================================


def ttest_ind(df: pl.DataFrame, values: str, group: str, detail: bool = False) -> float | dict:
    """
    Independent samples t-test.

    Tests whether two groups have significantly different means.
    H0: the two group means are equal.

    Assumes:
    - Both groups are approximately normal
    - Equal variances (use welch_ttest if variances differ)

    Example: "Do users on plan A spend more than users on plan B?"
    → ttest_ind(df, values="revenue", group="plan")

    :param df: the DataFrame
    :param values: column name containing the values
    :param group: column name containing the group labels (must have exactly 2 groups)
    :param detail: if True, returns t statistic + p-value
    :return: p-value (float) or dict with t and p-value
    """
    stats = _require("scipy.stats")
    groups = df[group].unique().to_list()
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}: {groups}")

    a = df.filter(pl.col(group) == groups[0])[values].drop_nulls().to_numpy()
    b = df.filter(pl.col(group) == groups[1])[values].drop_nulls().to_numpy()

    results = stats.ttest_ind(a, b)
    if not detail:
        return results.pvalue
    else:
        return {
            "t": results.statistic,
            "pvalue": results.pvalue,
            "group_a": groups[0],
            "group_b": groups[1],
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
        }


def ttest_paired(a: pl.Series, b: pl.Series, detail: bool = False) -> float | dict:
    """
    Paired samples t-test.

    Tests whether the mean difference between paired observations is zero.
    Use it when the same subjects are measured twice (before/after, pre/post).

    Example: "Did the training program improve test scores?"
    → ttest_paired(scores_before, scores_after)

    Assumes the differences are approximately normal.

    :param a: first measurement series
    :param b: second measurement series (same length, same order)
    :param detail: if True, returns t statistic + p-value
    :return: p-value (float) or dict with t and p-value
    """
    stats = _require("scipy.stats")
    a_clean = a.drop_nulls().to_numpy()
    b_clean = b.drop_nulls().to_numpy()

    if len(a_clean) != len(b_clean):
        raise ValueError("Both series must have the same length for paired test")

    results = stats.ttest_rel(a_clean, b_clean)
    if not detail:
        return results.pvalue
    else:
        return {
            "t": results.statistic,
            "pvalue": results.pvalue,
            "mean_diff": float((a_clean - b_clean).mean()),
        }


def welch_ttest(df: pl.DataFrame, values: str, group: str, detail: bool = False) -> float | dict:
    """
    Welch's t-test for two independent samples with unequal variances.

    Same purpose as ttest_ind but does NOT assume equal variances.
    In practice, Welch's test is almost always preferred over the classic
    t-test — it performs equally well when variances are equal, and better
    when they're not.

    :param df: the DataFrame
    :param values: column name containing the values
    :param group: column name containing the group labels (must have exactly 2 groups)
    :param detail: if True, returns t statistic + p-value
    :return: p-value (float) or dict with t and p-value
    """
    stats = _require("scipy.stats")
    groups = df[group].unique().to_list()
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}: {groups}")

    a = df.filter(pl.col(group) == groups[0])[values].drop_nulls().to_numpy()
    b = df.filter(pl.col(group) == groups[1])[values].drop_nulls().to_numpy()

    results = stats.ttest_ind(a, b, equal_var=False)
    if not detail:
        return results.pvalue
    else:
        return {
            "t": results.statistic,
            "pvalue": results.pvalue,
            "group_a": groups[0],
            "group_b": groups[1],
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
        }


def mann_whitney(df: pl.DataFrame, values: str, group: str, detail: bool = False) -> float | dict:
    """
    Mann-Whitney U test (non-parametric alternative to ttest_ind).

    Tests whether two groups come from the same distribution.
    Does not assume normality — works on ranks instead of raw values.

    Use it when:
    - Data is not normal
    - Data is ordinal (ratings, rankings)
    - You have outliers that would bias a t-test

    :param df: the DataFrame
    :param values: column name containing the values
    :param group: column name containing the group labels (must have exactly 2 groups)
    :param detail: if True, returns U statistic + p-value
    :return: p-value (float) or dict with U and p-value
    """
    stats = _require("scipy.stats")
    groups = df[group].unique().to_list()
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}: {groups}")

    a = df.filter(pl.col(group) == groups[0])[values].drop_nulls().to_numpy()
    b = df.filter(pl.col(group) == groups[1])[values].drop_nulls().to_numpy()

    results = stats.mannwhitneyu(a, b, alternative="two-sided")
    if not detail:
        return results.pvalue
    else:
        return {
            "U": results.statistic,
            "pvalue": results.pvalue,
            "group_a": groups[0],
            "group_b": groups[1],
        }


def wilcoxon_paired(a: pl.Series, b: pl.Series, detail: bool = False) -> float | dict:
    """
    Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to ttest_paired. Does not assume the
    differences are normal, only that they are symmetric.

    Use it when:
    - You have before/after data
    - The differences are not normally distributed
    - You want robustness against outliers

    :param a: first measurement series
    :param b: second measurement series (same length, same order)
    :param detail: if True, returns W statistic + p-value
    :return: p-value (float) or dict with W and p-value
    """
    stats = _require("scipy.stats")
    a_clean = a.drop_nulls().to_numpy()
    b_clean = b.drop_nulls().to_numpy()

    if len(a_clean) != len(b_clean):
        raise ValueError("Both series must have the same length for paired test")

    results = stats.wilcoxon(a_clean - b_clean)
    if not detail:
        return results.pvalue
    else:
        return {
            "W": results.statistic,
            "pvalue": results.pvalue,
        }


def kolmogorov_smirnov_2samp(a: pl.Series, b: pl.Series, detail: bool = False) -> float | dict:
    """
    Two-sample Kolmogorov-Smirnov test.

    Tests whether two samples come from the same distribution.
    Unlike t-tests which compare means, KS compares the entire shape
    of the two distributions.

    Use it when:
    - You want to compare distributions, not just means
    - Detecting any kind of difference (location, spread, shape)
    - Checking if two datasets were generated by the same process

    :param a: first series
    :param b: second series
    :param detail: if True, returns D statistic + p-value
    :return: p-value (float) or dict with D and p-value
    """
    stats = _require("scipy.stats")
    a_clean = a.drop_nulls().to_numpy()
    b_clean = b.drop_nulls().to_numpy()

    results = stats.ks_2samp(a_clean, b_clean)
    if not detail:
        return results.pvalue
    else:
        return {
            "D": results.statistic,
            "pvalue": results.pvalue,
        }


def levene(df: pl.DataFrame, values: str, group: str, detail: bool = False) -> float | dict:
    """
    Levene's test for equality of variances.

    Tests whether two or more groups have equal variances (homoscedasticity).
    This is an assumption of the classic t-test and ANOVA.

    More robust than Bartlett's test — works even when data is not normal.
    Run this before ttest_ind. If it rejects, use welch_ttest instead.

    :param df: the DataFrame
    :param values: column name containing the values
    :param group: column name containing the group labels
    :param detail: if True, returns test statistic + p-value
    :return: p-value (float) or dict with statistic and p-value
    """
    stats = _require("scipy.stats")
    groups = df[group].unique().to_list()
    samples = [
        df.filter(pl.col(group) == g)[values].drop_nulls().to_numpy()
        for g in groups
    ]

    results = stats.levene(*samples)
    if not detail:
        return results.pvalue
    else:
        return {
            "statistic": results.statistic,
            "pvalue": results.pvalue,
            "groups": groups,
        }


def bartlett(df: pl.DataFrame, values: str, group: str, detail: bool = False) -> float | dict:
    """
    Bartlett's test for equality of variances.

    Same purpose as Levene's test but assumes data is normal.
    More powerful than Levene when normality holds, but sensitive
    to non-normality (will reject even if variances are equal
    but data is skewed).

    Use Levene unless you've confirmed normality.

    :param df: the DataFrame
    :param values: column name containing the values
    :param group: column name containing the group labels
    :param detail: if True, returns test statistic + p-value
    :return: p-value (float) or dict with statistic and p-value
    """
    stats = _require("scipy.stats")
    groups = df[group].unique().to_list()
    samples = [
        df.filter(pl.col(group) == g)[values].drop_nulls().to_numpy()
        for g in groups
    ]

    results = stats.bartlett(*samples)
    if not detail:
        return results.pvalue
    else:
        return {
            "statistic": results.statistic,
            "pvalue": results.pvalue,
            "groups": groups,
        }


# ============================================================
# Effect sizes (two groups)
# ============================================================


def cohens_d_2samp(df: pl.DataFrame, values: str, group: str) -> dict:
    """
    Cohen's d for two independent samples.

    Measures how far apart the two group means are, in pooled
    standard deviation units. Tells you the practical significance
    of a difference, not just statistical significance.

    Interpretation:
    - |d| ≈ 0.2 : small effect
    - |d| ≈ 0.5 : medium effect
    - |d| ≈ 0.8 : large effect

    Always report alongside t-test p-values.

    :param df: the DataFrame
    :param values: column name containing the values
    :param group: column name containing the group labels (must have exactly 2 groups)
    :return: dict with Cohen's d, group means, and pooled std
    """
    np = _require("numpy")
    groups = df[group].unique().to_list()
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}: {groups}")

    a = df.filter(pl.col(group) == groups[0])[values].drop_nulls().to_numpy()
    b = df.filter(pl.col(group) == groups[1])[values].drop_nulls().to_numpy()

    pooled_std = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))

    d = (np.mean(a) - np.mean(b)) / pooled_std

    return {
        "d": float(d),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "pooled_std": float(pooled_std),
        "group_a": groups[0],
        "group_b": groups[1],
    }


def rank_biserial(df: pl.DataFrame, values: str, group: str) -> dict:
    """
    Rank-biserial correlation (effect size for Mann-Whitney U).

    The non-parametric equivalent of Cohen's d. Ranges from -1 to +1.
    Tells you the probability that a random observation from group A
    is larger than a random observation from group B (rescaled to [-1, 1]).

    Interpretation:
    - |r| ≈ 0.1 : small effect
    - |r| ≈ 0.3 : medium effect
    - |r| ≈ 0.5 : large effect

    Always report alongside Mann-Whitney p-values.

    :param df: the DataFrame
    :param values: column name containing the values
    :param group: column name containing the group labels (must have exactly 2 groups)
    :return: dict with rank-biserial r and group info
    """
    stats = _require("scipy.stats")
    groups = df[group].unique().to_list()
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}: {groups}")

    a = df.filter(pl.col(group) == groups[0])[values].drop_nulls().to_numpy()
    b = df.filter(pl.col(group) == groups[1])[values].drop_nulls().to_numpy()

    U = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    n_a = len(a)
    n_b = len(b)
    r = 1 - (2 * U) / (n_a * n_b)

    return {
        "r": float(r),
        "U": float(U),
        "group_a": groups[0],
        "group_b": groups[1],
    }


# ============================================================
# K-group comparisons
# ============================================================


def anova_oneway(df: pl.DataFrame, values: str, group: str, detail: bool = False) -> float | dict:
    """
    One-way ANOVA.

    Tests whether three or more groups have significantly different means.
    The generalization of the t-test to k groups.
    H0: all group means are equal.

    Assumes:
    - Each group is approximately normal
    - Equal variances across groups (check with levene())
    - Independent observations

    If it rejects, it tells you "at least one group differs" but not which.
    Use tukey_hsd() as a post-hoc test to find which pairs differ.

    :param df: the DataFrame
    :param values: column name containing the values
    :param group: column name containing the group labels
    :param detail: if True, returns F statistic + p-value
    :return: p-value (float) or dict with F and p-value
    """
    stats = _require("scipy.stats")
    groups = df[group].unique().to_list()
    samples = [
        df.filter(pl.col(group) == g)[values].drop_nulls().to_numpy()
        for g in groups
    ]

    results = stats.f_oneway(*samples)
    if not detail:
        return results.pvalue
    else:
        return {
            "F": results.statistic,
            "pvalue": results.pvalue,
            "groups": groups,
            "n_groups": len(groups),
        }


def welch_anova(df: pl.DataFrame, values: str, group: str, detail: bool = False) -> float | dict:
    """
    Welch's ANOVA for groups with unequal variances.

    Same purpose as anova_oneway but does NOT assume equal variances.
    Use when levene() rejects the equality of variances.

    Uses scipy's Alexander-Govern approximation.

    :param df: the DataFrame
    :param values: column name containing the values
    :param group: column name containing the group labels
    :param detail: if True, returns statistic + p-value
    :return: p-value (float) or dict with statistic and p-value
    """
    stats = _require("scipy.stats")
    groups = df[group].unique().to_list()
    samples = [
        df.filter(pl.col(group) == g)[values].drop_nulls().to_numpy()
        for g in groups
    ]

    results = stats.alexandergovern(*samples)
    if not detail:
        return results.pvalue
    else:
        return {
            "statistic": results.statistic,
            "pvalue": results.pvalue,
            "groups": groups,
        }


def kruskal_wallis(df: pl.DataFrame, values: str, group: str, detail: bool = False) -> float | dict:
    """
    Kruskal-Wallis H test (non-parametric alternative to ANOVA).

    Tests whether the distributions of k groups are identical.
    Works on ranks — no normality assumption needed.

    Use it when:
    - Data is not normal
    - Data is ordinal
    - You have outliers

    If it rejects, use dunn() as a post-hoc test.

    :param df: the DataFrame
    :param values: column name containing the values
    :param group: column name containing the group labels
    :param detail: if True, returns H statistic + p-value
    :return: p-value (float) or dict with H and p-value
    """
    stats = _require("scipy.stats")
    groups = df[group].unique().to_list()
    samples = [
        df.filter(pl.col(group) == g)[values].drop_nulls().to_numpy()
        for g in groups
    ]

    results = stats.kruskal(*samples)
    if not detail:
        return results.pvalue
    else:
        return {
            "H": results.statistic,
            "pvalue": results.pvalue,
            "groups": groups,
        }


def friedman(samples: list, detail: bool = False) -> float | dict:
    """
    Friedman test (non-parametric repeated measures ANOVA).

    Tests whether k related groups have different distributions.
    The non-parametric alternative to repeated measures ANOVA.

    Use it when:
    - Same subjects measured under k conditions
    - Data is ordinal or not normal
    - Example: rating the same product under 3 different packaging designs

    :param samples: list of pl.Series (one per condition, same length and order)
    :param detail: if True, returns chi-squared statistic + p-value
    :return: p-value (float) or dict with statistic and p-value
    """
    stats = _require("scipy.stats")
    arrays = [s.drop_nulls().to_numpy() for s in samples]

    results = stats.friedmanchisquare(*arrays)
    if not detail:
        return results.pvalue
    else:
        return {
            "chi2": results.statistic,
            "pvalue": results.pvalue,
            "k": len(samples),
        }


# ============================================================
# Post-hoc tests
# ============================================================


def tukey_hsd(df: pl.DataFrame, values: str, group: str) -> dict:
    """
    Tukey's Honestly Significant Difference post-hoc test.

    After ANOVA tells you "at least one group differs", Tukey HSD
    tells you WHICH pairs of groups are significantly different.

    Controls for multiple comparisons — the family-wise error rate
    stays at alpha even when testing many pairs.

    :param df: the DataFrame
    :param values: column name containing the values
    :param group: column name containing the group labels
    :return: dict with pairwise comparisons, each with mean difference and p-value
    """
    stats = _require("scipy.stats")
    groups = df[group].unique().sort().to_list()
    samples = [
        df.filter(pl.col(group) == g)[values].drop_nulls().to_numpy()
        for g in groups
    ]

    result = stats.tukey_hsd(*samples)

    comparisons = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            comparisons.append({
                "group_a": groups[i],
                "group_b": groups[j],
                "pvalue": float(result.pvalue[i][j]),
                "statistic": float(result.statistic[i][j]),
            })

    return {
        "comparisons": comparisons,
        "n_groups": len(groups),
    }


def dunn(df: pl.DataFrame, values: str, group: str) -> dict:
    """
    Dunn's post-hoc test (after Kruskal-Wallis).

    The non-parametric equivalent of Tukey HSD. Tests all pairwise
    group comparisons after a significant Kruskal-Wallis result.

    Uses Bonferroni correction for multiple comparisons.

    Requires the 'scikit_posthocs' package.

    :param df: the DataFrame
    :param values: column name containing the values
    :param group: column name containing the group labels
    :return: dict with pairwise comparisons and p-values
    """
    sp = _require("scikit_posthocs")
    np = _require("numpy")

    groups = df[group].unique().sort().to_list()
    samples = [
        df.filter(pl.col(group) == g)[values].drop_nulls().to_numpy()
        for g in groups
    ]

    # Build a flat array + group labels for scikit_posthocs
    all_values = np.concatenate(samples)
    all_groups = np.concatenate([[g] * len(s) for g, s in zip(groups, samples)])

    pval_matrix = sp.posthoc_dunn([all_values], group=all_groups, p_adjust="bonferroni")

    comparisons = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            comparisons.append({
                "group_a": groups[i],
                "group_b": groups[j],
                "pvalue": float(pval_matrix.loc[groups[i], groups[j]]),
            })

    return {
        "comparisons": comparisons,
        "n_groups": len(groups),
        "correction": "bonferroni",
    }


# ============================================================
# Effect sizes (k groups)
# ============================================================


def eta_squared(df: pl.DataFrame, values: str, group: str) -> float:
    """
    Eta squared (η²) — effect size for one-way ANOVA.

    Represents the proportion of total variance explained by the grouping.
    Analogous to R² in regression.

    Interpretation:
    - η² ≈ 0.01 : small effect
    - η² ≈ 0.06 : medium effect
    - η² ≈ 0.14 : large effect

    Tends to overestimate the true effect in the population.
    Use omega_squared() for a less biased estimate.

    :param df: the DataFrame
    :param values: column name containing the values
    :param group: column name containing the group labels
    :return: eta squared (float between 0 and 1)
    """
    np = _require("numpy")
    groups = df[group].unique().to_list()

    all_values = df[values].drop_nulls().to_numpy()
    grand_mean = np.mean(all_values)

    ss_between = sum(
        len(df.filter(pl.col(group) == g)[values].drop_nulls()) *
        (df.filter(pl.col(group) == g)[values].mean() - grand_mean) ** 2
        for g in groups
    )

    ss_total = np.sum((all_values - grand_mean) ** 2)

    return float(ss_between / ss_total)


def omega_squared(df: pl.DataFrame, values: str, group: str) -> float:
    """
    Omega squared (ω²) — less biased effect size for ANOVA.

    Same purpose as eta_squared but corrects for the positive bias
    of η². Preferred in publications and when comparing across studies.

    Interpretation:
    - ω² ≈ 0.01 : small effect
    - ω² ≈ 0.06 : medium effect
    - ω² ≈ 0.14 : large effect

    :param df: the DataFrame
    :param values: column name containing the values
    :param group: column name containing the group labels
    :return: omega squared (float between 0 and 1)
    """
    np = _require("numpy")
    groups = df[group].unique().to_list()
    k = len(groups)

    all_values = df[values].drop_nulls().to_numpy()
    n_total = len(all_values)
    grand_mean = np.mean(all_values)

    ss_between = sum(
        len(df.filter(pl.col(group) == g)[values].drop_nulls()) *
        (df.filter(pl.col(group) == g)[values].mean() - grand_mean) ** 2
        for g in groups
    )

    ss_total = np.sum((all_values - grand_mean) ** 2)
    ms_within = (ss_total - ss_between) / (n_total - k)

    return float((ss_between - (k - 1) * ms_within) / (ss_total + ms_within))
