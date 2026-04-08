"""
polars_stats — Statistical analysis toolkit for Polars.

Provides two main interfaces:

1. Function-based (for scripts and pipelines):
    from polars_stats.univariate.descriptive import mean, median
    from polars_stats.multivariate.comparison import ttest_ind

2. Class-based (for interactive exploration):
    from polars_stats import Univariate, Multivariate

    uv = Univariate(series)
    uv.mean()
    uv.shapiro_wilk()

    mv = Multivariate(df, ["col_a", "col_b", "col_c"])
    mv.correlation_matrix()
    mv.ols(target="col_a")
"""

from polars_stats.wrappers import Univariate, Multivariate

__version__ = "0.1.0"
__all__ = ["Univariate", "Multivariate"]