"""
polars_stats.multivariate — Statistical analysis on multiple variables.

Submodules:
    descriptive — Cross-summary, correlation matrix, covariance matrix, partial correlation
    comparison  — t-tests, ANOVA, post-hoc tests, effect sizes
    correlation — Pearson, Spearman, Kendall, chi-squared, mutual information
    regression  — OLS, logistic, ridge, lasso, VIF, residual diagnostics
    dimension   — PCA, scree data
    tests       — Hotelling T², Mahalanobis distance, Box M
"""

from polars_stats.multivariate import descriptive
from polars_stats.multivariate import comparison
from polars_stats.multivariate import correlation
from polars_stats.multivariate import regression
from polars_stats.multivariate import dimension
from polars_stats.multivariate import tests

__all__ = ["descriptive", "comparison", "correlation", "regression", "dimension", "tests"]