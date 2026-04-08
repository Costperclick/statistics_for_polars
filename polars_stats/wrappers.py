"""
Class-based wrappers for interactive exploration.

These classes delegate to the pure functions in each submodule.
They add no logic — just ergonomic method access on a bound Series or DataFrame.
"""

import polars as pl

from polars_stats.univariate import descriptive as uv_desc
from polars_stats.univariate import tests as uv_tests
from polars_stats.univariate import distribution as uv_dist
from polars_stats.univariate import inference as uv_inf

from polars_stats.multivariate import descriptive as mv_desc
from polars_stats.multivariate import comparison as mv_comp
from polars_stats.multivariate import correlation as mv_corr
from polars_stats.multivariate import regression as mv_reg
from polars_stats.multivariate import dimension as mv_dim
from polars_stats.multivariate import tests as mv_tests


class Univariate:
    """
    Wraps a Polars Series for interactive statistical exploration.

    Usage:
        uv = Univariate(df["revenue"])
        uv.mean()
        uv.shapiro_wilk()
        uv.ci_mean_bootstrap()
    """

    def __init__(self, s: pl.Series):
        self._s = s

    def __repr__(self):
        return f"Univariate(name='{self._s.name}', len={len(self._s)})"

    # ── Descriptive ──────────────────────────────────────────

    def mean(self, **kwargs):
        return uv_desc.mean(self._s, **kwargs)

    def median(self, **kwargs):
        return uv_desc.median(self._s, **kwargs)

    def weighted_mean(self, weights: pl.Series):
        return uv_desc.weighted_mean(self._s, weights)

    def geometric_mean(self, **kwargs):
        return uv_desc.geometric_mean(self._s, **kwargs)

    def harmonic_mean(self):
        return uv_desc.harmonic_mean(self._s)

    def mode(self):
        return uv_desc.mode(self._s)

    def variance(self):
        return uv_desc.variance(self._s)

    def standard_deviation(self):
        return uv_desc.standard_deviation(self._s)

    def median_absolute_deviation(self):
        return uv_desc.median_absolute_deviation(self._s)

    def skewness(self):
        return uv_desc.skewness(self._s)

    def kurtosis(self):
        return uv_desc.kurtosis(self._s)

    def quantile(self, q: float):
        return uv_desc.quantile(self._s, q)

    def iqr(self):
        return uv_desc.iqr(self._s)

    def data_range(self):
        return uv_desc.data_range(self._s)

    def variance_coefficient(self):
        return uv_desc.variance_coefficient(self._s)

    def s_entropy(self):
        return uv_desc.s_entropy(self._s)

    def gini(self):
        return uv_desc.gini(self._s)

    # ── Tests ────────────────────────────────────────────────

    def shapiro_wilk(self, **kwargs):
        return uv_tests.shapiro_wilk(self._s, **kwargs)

    def dagostino_pearson(self, **kwargs):
        return uv_tests.dagostino_pearson(self._s, **kwargs)

    def kolmogorov_smirnov(self, **kwargs):
        return uv_tests.kolmogorov_smirnov(self._s, **kwargs)

    def normality(self, **kwargs):
        return uv_tests.normality(self._s, **kwargs)

    def ttest_1samp(self, mu: float, **kwargs):
        return uv_tests.ttest_1samp(self._s, mu, **kwargs)

    def ztest_1samp(self, mu: float, **kwargs):
        return uv_tests.ztest_1samp(self._s, mu, **kwargs)

    def wilcoxon_1samp(self, **kwargs):
        return uv_tests.wilcoxon_1samp(self._s, **kwargs)

    def sign_test(self, **kwargs):
        return uv_tests.sign_test(self._s, **kwargs)

    def binomial_test(self, **kwargs):
        return uv_tests.binomial_test(self._s, **kwargs)

    def outliers_zscore(self, **kwargs):
        return uv_tests.outliers_zscore(self._s, **kwargs)

    def outliers_mad(self, **kwargs):
        return uv_tests.outliers_mad(self._s, **kwargs)

    def outliers_iqr(self, **kwargs):
        return uv_tests.outliers_iqr(self._s, **kwargs)

    def outliers_grubbs(self, **kwargs):
        return uv_tests.outliers_grubbs(self._s, **kwargs)

    def cohens_d(self, mu: float):
        return uv_tests.cohens_d(self._s, mu)

    def which_test(self, objective: str = "normality"):
        return uv_tests.which_test(self._s, objective)

    # ── Distribution ─────────────────────────────────────────

    def distribution_fit(self, **kwargs):
        return uv_dist.distribution_fit(self._s, **kwargs)

    def qqplot_data(self, **kwargs):
        return uv_dist.qqplot_data(self._s, **kwargs)

    def kde(self, **kwargs):
        return uv_dist.kde(self._s, **kwargs)

    # ── Inference ────────────────────────────────────────────

    def ci_mean_bootstrap(self, **kwargs):
        return uv_inf.ci_mean_bootstrap(self._s, **kwargs)

    def ci_median_bootstrap(self, **kwargs):
        return uv_inf.ci_median_bootstrap(self._s, **kwargs)

    def ci_mean(self, **kwargs):
        return uv_inf.ci_mean(self._s, **kwargs)

    def ci_variance(self, **kwargs):
        return uv_inf.ci_variance(self._s, **kwargs)


class Multivariate:
    """
    Wraps a Polars DataFrame for interactive statistical exploration.

    Usage:
        mv = Multivariate(df, ["revenue", "users", "sessions"])
        mv.correlation_matrix()
        mv.ols(target="revenue")
        mv.pca(n_components=2)
    """

    def __init__(self, df: pl.DataFrame, columns: list = None):
        self._df = df
        self._columns = columns

    def __repr__(self):
        cols = self._columns or self._df.columns
        return f"Multivariate(columns={cols}, rows={len(self._df)})"

    # ── Descriptive ──────────────────────────────────────────

    def mean(self):
        return mv_desc.mean(self._df, self._columns)

    def cross_summary(self):
        return mv_desc.cross_summary(self._df, self._columns)

    def correlation_matrix(self, **kwargs):
        return mv_desc.correlation_matrix(self._df, self._columns, **kwargs)

    def covariance_matrix(self):
        return mv_desc.covariance_matrix(self._df, self._columns)

    def partial_correlation(self, x: str, y: str, controls: list):
        return mv_desc.partial_correlation(self._df, x, y, controls)

    # ── Comparison ───────────────────────────────────────────

    def ttest_ind(self, values: str, group: str, **kwargs):
        return mv_comp.ttest_ind(self._df, values, group, **kwargs)

    def ttest_paired(self, a: str, b: str, **kwargs):
        return mv_comp.ttest_paired(self._df[a], self._df[b], **kwargs)

    def welch_ttest(self, values: str, group: str, **kwargs):
        return mv_comp.welch_ttest(self._df, values, group, **kwargs)

    def mann_whitney(self, values: str, group: str, **kwargs):
        return mv_comp.mann_whitney(self._df, values, group, **kwargs)

    def wilcoxon_paired(self, a: str, b: str, **kwargs):
        return mv_comp.wilcoxon_paired(self._df[a], self._df[b], **kwargs)

    def kolmogorov_smirnov_2samp(self, a: str, b: str, **kwargs):
        return mv_comp.kolmogorov_smirnov_2samp(self._df[a], self._df[b], **kwargs)

    def levene(self, values: str, group: str, **kwargs):
        return mv_comp.levene(self._df, values, group, **kwargs)

    def bartlett(self, values: str, group: str, **kwargs):
        return mv_comp.bartlett(self._df, values, group, **kwargs)

    def cohens_d_2samp(self, values: str, group: str):
        return mv_comp.cohens_d_2samp(self._df, values, group)

    def rank_biserial(self, values: str, group: str):
        return mv_comp.rank_biserial(self._df, values, group)

    def anova_oneway(self, values: str, group: str, **kwargs):
        return mv_comp.anova_oneway(self._df, values, group, **kwargs)

    def welch_anova(self, values: str, group: str, **kwargs):
        return mv_comp.welch_anova(self._df, values, group, **kwargs)

    def kruskal_wallis(self, values: str, group: str, **kwargs):
        return mv_comp.kruskal_wallis(self._df, values, group, **kwargs)

    def tukey_hsd(self, values: str, group: str):
        return mv_comp.tukey_hsd(self._df, values, group)

    def dunn(self, values: str, group: str):
        return mv_comp.dunn(self._df, values, group)

    def eta_squared(self, values: str, group: str):
        return mv_comp.eta_squared(self._df, values, group)

    def omega_squared(self, values: str, group: str):
        return mv_comp.omega_squared(self._df, values, group)

    # ── Correlation ──────────────────────────────────────────

    def pearson(self, a: str, b: str, **kwargs):
        return mv_corr.pearson(self._df[a], self._df[b], **kwargs)

    def spearman(self, a: str, b: str, **kwargs):
        return mv_corr.spearman(self._df[a], self._df[b], **kwargs)

    def kendall(self, a: str, b: str, **kwargs):
        return mv_corr.kendall(self._df[a], self._df[b], **kwargs)

    def point_biserial(self, binary: str, continuous: str, **kwargs):
        return mv_corr.point_biserial(self._df[binary], self._df[continuous], **kwargs)

    def chi2_independence(self, col_a: str, col_b: str, **kwargs):
        return mv_corr.chi2_independence(self._df, col_a, col_b, **kwargs)

    def cramers_v(self, col_a: str, col_b: str):
        return mv_corr.cramers_v(self._df, col_a, col_b)

    def mutual_information(self, a: str, b: str, **kwargs):
        return mv_corr.mutual_information(self._df[a], self._df[b], **kwargs)

    # ── Regression ───────────────────────────────────────────

    def ols(self, target: str, features: list = None):
        return mv_reg.ols(self._df, target, features or self._columns)

    def logistic(self, target: str, features: list = None):
        return mv_reg.logistic(self._df, target, features or self._columns)

    def ridge(self, target: str, features: list = None, **kwargs):
        return mv_reg.ridge(self._df, target, features or self._columns, **kwargs)

    def lasso(self, target: str, features: list = None, **kwargs):
        return mv_reg.lasso(self._df, target, features or self._columns, **kwargs)

    def vif(self, features: list = None):
        return mv_reg.vif(self._df, features or self._columns)

    def residual_diagnostics(self, target: str, features: list = None):
        return mv_reg.residual_diagnostics(self._df, target, features or self._columns)

    # ── Dimension ────────────────────────────────────────────

    def pca(self, **kwargs):
        return mv_dim.pca(self._df, self._columns, **kwargs)

    def scree_data(self):
        return mv_dim.scree_data(self._df, self._columns)

    # ── Tests ────────────────────────────────────────────────

    def hotelling_t2(self, features: list, group: str, **kwargs):
        return mv_tests.hotelling_t2(self._df, features, group, **kwargs)

    def mahalanobis(self, features: list = None):
        return mv_tests.mahalanobis(self._df, features or self._columns)

    def box_m(self, features: list, group: str):
        return mv_tests.box_m(self._df, features, group)