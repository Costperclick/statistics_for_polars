"""
polars_stats.univariate — Statistical analysis on single variables.

Submodules:
    descriptive  — Measures of central tendency, dispersion, shape, diversity
    tests        — Normality tests, hypothesis tests, outlier detection, effect size
    distribution — Distribution fitting, QQ-plot data, KDE
    inference    — Confidence intervals (parametric and bootstrap)
"""

from polars_stats.univariate import descriptive
from polars_stats.univariate import tests
from polars_stats.univariate import distribution
from polars_stats.univariate import inference

__all__ = ["descriptive", "tests", "distribution", "inference"]