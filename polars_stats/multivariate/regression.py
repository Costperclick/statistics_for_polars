import polars as pl
from polars_stats._utils import _require


def ols(df: pl.DataFrame, target: str, features: list = None) -> dict:
    """
    Ordinary Least Squares regression.

    Fits a linear model: target = β₀ + β₁×x₁ + β₂×x₂ + ... + ε

    Returns a complete summary: coefficients, standard errors, p-values,
    R², adjusted R², and F-statistic. This is the workhorse of statistical
    modeling — use it to understand how variables relate to an outcome.

    Example: "How do price, location, and size affect apartment rent?"
    → ols(df, target="rent", features=["price_sqm", "district", "area"])

    Assumptions:
    - Linear relationship between features and target
    - Residuals are normal and homoscedastic
    - No severe multicollinearity (check with vif())

    Use residual_diagnostics() after fitting to verify assumptions.

    :param df: the DataFrame
    :param target: column name of the dependent variable
    :param features: list of feature column names (default: all columns except target)
    :return: dict with coefficients, p-values, R², and full model summary
    """
    sm = _require("statsmodels.api")
    np = _require("numpy")

    if features is None:
        features = [c for c in df.columns if c != target]

    data = df.select([target] + features).drop_nulls()
    y = data[target].to_numpy()
    X = sm.add_constant(data.select(features).to_numpy())

    model = sm.OLS(y, X).fit()

    coef_names = ["intercept"] + features

    return {
        "coefficients": {name: float(coef) for name, coef in zip(coef_names, model.params)},
        "std_errors": {name: float(se) for name, se in zip(coef_names, model.bse)},
        "pvalues": {name: float(p) for name, p in zip(coef_names, model.pvalues)},
        "r_squared": float(model.rsquared),
        "r_squared_adj": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "n": int(model.nobs),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "summary": str(model.summary()),
    }


def logistic(df: pl.DataFrame, target: str, features: list = None) -> dict:
    """
    Logistic regression for binary classification.

    Models the probability of the target being 1 (or True) as a function
    of the features. The target must be binary (0/1).

    Example: "What factors predict customer churn?"
    → logistic(df, target="churned", features=["tenure", "monthly_charge", "support_calls"])

    Returns coefficients as log-odds. To interpret:
    - Positive coefficient → increases the probability of target=1
    - exp(coefficient) → odds ratio (e.g. exp(0.5) = 1.65× more likely)

    :param df: the DataFrame
    :param target: column name of the binary dependent variable (0/1)
    :param features: list of feature column names (default: all columns except target)
    :return: dict with coefficients, odds ratios, p-values, and model summary
    """
    sm = _require("statsmodels.api")
    np = _require("numpy")

    if features is None:
        features = [c for c in df.columns if c != target]

    data = df.select([target] + features).drop_nulls()
    y = data[target].to_numpy()
    X = sm.add_constant(data.select(features).to_numpy())

    model = sm.Logit(y, X).fit(disp=0)

    coef_names = ["intercept"] + features

    return {
        "coefficients": {name: float(coef) for name, coef in zip(coef_names, model.params)},
        "odds_ratios": {name: float(np.exp(coef)) for name, coef in zip(coef_names, model.params)},
        "pvalues": {name: float(p) for name, p in zip(coef_names, model.pvalues)},
        "pseudo_r_squared": float(model.prsquared),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "n": int(model.nobs),
        "summary": str(model.summary()),
    }


def ridge(df: pl.DataFrame, target: str, features: list = None, alpha: float = 1.0) -> dict:
    """
    Ridge regression (L2 regularization).

    Like OLS but adds a penalty on large coefficients to prevent overfitting.
    Useful when:
    - You have many features relative to observations
    - Features are correlated (multicollinearity)
    - You want to keep all features but shrink their effects

    The alpha parameter controls the strength of regularization:
    - alpha=0 → equivalent to OLS
    - alpha→∞ → all coefficients shrink toward zero

    Does NOT produce p-values (regularized models don't have clean inferential
    properties). Use OLS for inference, ridge for prediction.

    Requires scikit-learn.

    :param df: the DataFrame
    :param target: column name of the dependent variable
    :param features: list of feature column names (default: all columns except target)
    :param alpha: regularization strength (default 1.0)
    :return: dict with coefficients, intercept, R², and alpha
    """
    sklearn_linear = _require("sklearn.linear_model")
    np = _require("numpy")

    if features is None:
        features = [c for c in df.columns if c != target]

    data = df.select([target] + features).drop_nulls()
    y = data[target].to_numpy()
    X = data.select(features).to_numpy()

    model = sklearn_linear.Ridge(alpha=alpha)
    model.fit(X, y)

    return {
        "coefficients": {name: float(coef) for name, coef in zip(features, model.coef_)},
        "intercept": float(model.intercept_),
        "r_squared": float(model.score(X, y)),
        "alpha": alpha,
    }


def lasso(df: pl.DataFrame, target: str, features: list = None, alpha: float = 1.0) -> dict:
    """
    Lasso regression (L1 regularization).

    Like ridge but can shrink coefficients to exactly zero, effectively
    performing feature selection. Features with zero coefficients are
    excluded from the model.

    Useful when:
    - You suspect many features are irrelevant
    - You want an interpretable model with fewer variables
    - You need automatic feature selection

    The alpha parameter controls the sparsity:
    - alpha=0 → equivalent to OLS (all features kept)
    - alpha→∞ → all coefficients become zero (no features)

    Requires scikit-learn.

    :param df: the DataFrame
    :param target: column name of the dependent variable
    :param features: list of feature column names (default: all columns except target)
    :param alpha: regularization strength (default 1.0)
    :return: dict with coefficients (including zeros), intercept, R², selected features
    """
    sklearn_linear = _require("sklearn.linear_model")
    np = _require("numpy")

    if features is None:
        features = [c for c in df.columns if c != target]

    data = df.select([target] + features).drop_nulls()
    y = data[target].to_numpy()
    X = data.select(features).to_numpy()

    model = sklearn_linear.Lasso(alpha=alpha)
    model.fit(X, y)

    coefficients = {name: float(coef) for name, coef in zip(features, model.coef_)}
    selected = [name for name, coef in coefficients.items() if coef != 0]

    return {
        "coefficients": coefficients,
        "intercept": float(model.intercept_),
        "r_squared": float(model.score(X, y)),
        "alpha": alpha,
        "selected_features": selected,
        "n_selected": len(selected),
    }


def vif(df: pl.DataFrame, features: list = None) -> dict:
    """
    Variance Inflation Factor for multicollinearity detection.

    VIF measures how much the variance of a regression coefficient is
    inflated due to correlation with other features. High VIF means
    a feature is redundant — it can be predicted from the others.

    Interpretation:
    - VIF = 1   : no multicollinearity
    - VIF 1-5   : moderate (usually acceptable)
    - VIF 5-10  : high (concerning)
    - VIF > 10  : severe (this feature is almost entirely redundant)

    Run this BEFORE regression. If VIF is high, consider:
    - Removing one of the correlated features
    - Combining them (PCA)
    - Using ridge regression (tolerates multicollinearity)

    :param df: the DataFrame
    :param features: list of feature column names (default: all numeric columns)
    :return: dict mapping feature names to their VIF values
    """
    np = _require("numpy")
    sm = _require("statsmodels.api")

    if features is None:
        features = df.select(pl.col(pl.NUMERIC_DTYPES)).columns

    data = df.select(features).drop_nulls().to_numpy()
    data_with_const = sm.add_constant(data)

    vif_values = {}
    for i, feature in enumerate(features):
        col_idx = i + 1  # +1 because of the added constant
        y = data_with_const[:, col_idx]
        X = np.delete(data_with_const, col_idx, axis=1)
        r_squared = sm.OLS(y, X).fit().rsquared
        vif_values[feature] = float(1 / (1 - r_squared))

    return vif_values


def residual_diagnostics(df: pl.DataFrame, target: str, features: list = None) -> dict:
    """
    Diagnostic tests on OLS regression residuals.

    After fitting a regression, you need to verify its assumptions:
    1. Residuals are normally distributed
    2. Residuals have constant variance (homoscedasticity)
    3. Residuals are not autocorrelated

    This function runs three tests:
    - Shapiro-Wilk on residuals → normality
    - Breusch-Pagan → homoscedasticity (constant variance)
    - Durbin-Watson → autocorrelation (important for time series)

    Run this after ols() to validate your model.

    :param df: the DataFrame
    :param target: column name of the dependent variable
    :param features: list of feature column names (default: all columns except target)
    :return: dict with test results and interpretation
    """
    sm = _require("statsmodels.api")
    stats = _require("scipy.stats")
    sms = _require("statsmodels.stats.stattools")
    sm_diag = _require("statsmodels.stats.diagnostic")
    np = _require("numpy")

    if features is None:
        features = [c for c in df.columns if c != target]

    data = df.select([target] + features).drop_nulls()
    y = data[target].to_numpy()
    X = sm.add_constant(data.select(features).to_numpy())

    model = sm.OLS(y, X).fit()
    residuals = model.resid

    # Normality of residuals
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])

    # Homoscedasticity (Breusch-Pagan)
    bp_stat, bp_p, _, _ = sm_diag.het_breuschpagan(residuals, X)

    # Autocorrelation (Durbin-Watson)
    dw = float(sms.durbin_watson(residuals))

    return {
        "normality": {
            "test": "shapiro_wilk",
            "statistic": float(shapiro_stat),
            "pvalue": float(shapiro_p),
            "normal": shapiro_p >= 0.05,
        },
        "homoscedasticity": {
            "test": "breusch_pagan",
            "statistic": float(bp_stat),
            "pvalue": float(bp_p),
            "homoscedastic": bp_p >= 0.05,
        },
        "autocorrelation": {
            "test": "durbin_watson",
            "statistic": dw,
            "interpretation": "no autocorrelation" if 1.5 < dw < 2.5 else "possible autocorrelation",
        },
    }