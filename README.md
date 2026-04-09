# polars-stats

!! -- STILL WORKING ON -- !!
Statistical analysis toolkit for [Polars](https://pola.rs/). Input Polars, output clean results.

## Why ? 

Polars is great but it stops at data manipulation. The moment you need a t-test or a confidence interval, you're back to importing scipy, converting to numpy, and googling the same function signatures you googled last month.
This package wraps all of that. Pass a pl.Series or a pl.DataFrame, get a result. 84 functions covering descriptive stats, hypothesis tests, regression, PCA...
The docs double as a stats refresher. Every function tells you when to use it, not just what it computes. Handy if, like me, you were too busy being drunk in your twenties to get a master degree in statistics.

## Installation

```bash
# Base (descriptive stats only, no dependencies beyond Polars)
pip install "polars-stats @ git+https://github.com/costperclick/statistics_for_polars.git"

# With hypothesis tests and regression
pip install "polars-stats[tests] @ git+https://github.com/costperclick/statistics_for_polars.git"

# Everything
pip install "polars-stats[full] @ git+https://github.com/costperclick/statistics_for_polars.git"
```

## Quick start

### Function-based (for scripts and pipelines)

```python
import polars as pl
from polars_stats.univariate.descriptive import mean, median, skewness
from polars_stats.univariate.tests import shapiro_wilk, ttest_1samp

s = pl.Series("revenue", [120, 150, 130, 200, 145, 160, 180])

mean(s)             # 169.3
skewness(s)         # 0.42
shapiro_wilk(s)     # 0.87 (p-value)
ttest_1samp(s, 150) # 0.12 (p-value)
```

### Class-based (for interactive exploration)

```python
from polars_stats import Univariate, Multivariate

uv = Univariate(df["revenue"])
uv.mean()
uv.shapiro_wilk()
uv.ci_mean_bootstrap()
uv.which_test("normality")  # tells you which test to use !!

mv = Multivariate(df, ["revenue", "users", "sessions"])
mv.correlation_matrix()
mv.ols(target="revenue")
mv.pca(n_components=2)
```

## Modules

### Univariate (single variable)

| Module | Functions | Dependencies |
|---|---|---|
| `descriptive` | mean, median, mode, variance, std, skewness, kurtosis, gini, entropy... (18) | None |
| `tests` | shapiro_wilk, ttest_1samp, wilcoxon, outliers_zscore, cohens_d... (17) | scipy |
| `distribution` | distribution_fit, qqplot_data, kde (3) | scipy |
| `inference` | ci_mean, ci_mean_bootstrap, ci_proportion... (5) | scipy (parametric) / None (bootstrap) |

### Multivariate (multiple variables)

| Module | Functions | Dependencies |
|---|---|---|
| `descriptive` | mean, cross_summary, correlation_matrix, covariance_matrix, partial_correlation (5) | scipy |
| `comparison` | ttest_ind, welch_ttest, mann_whitney, anova, tukey_hsd, cohens_d_2samp... (18) | scipy |
| `correlation` | pearson, spearman, kendall, chi2_independence, cramers_v, mutual_information... (7) | scipy |
| `regression` | ols, logistic, ridge, lasso, vif, residual_diagnostics (6) | statsmodels, sklearn |
| `dimension` | pca, scree_data (2) | sklearn |
| `tests` | hotelling_t2, mahalanobis, box_m (3) | scipy |

**Total: 84 functions.**

## Guides - French only so far

- [Statistics Guide (Univariate)](univarié_guide.md) — Concepts, workflow, decision trees
- [Multivariate Guide](MULTIVARIATE_GUIDE.md) — Correlation, comparison, regression, PCA

## Project structure

```
polars_stats/
├── __init__.py                  # Exports Univariate, Multivariate
├── _utils.py                    # _require(), to_clean_array()
├── wrappers.py                  # Univariate and Multivariate classes
├── univariate/
│   ├── __init__.py
│   ├── descriptive.py           
│   ├── tests.py                 
│   ├── distribution.py          
│   └── inference.py             
└── multivariate/
    ├── __init__.py
    ├── descriptive.py          
    ├── comparison.py            
    ├── correlation.py
    ├── regression.py
    ├── dimension.py
    └── tests.py
```

## Tests : 

```bash
pip install -e ".[full]"
pip install pytest
pytest tests/ -v
```