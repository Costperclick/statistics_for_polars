import polars as pl
from _utils import _require


def distribution_fit(n: pl.Series, dist: str = "norm") -> dict:
    """
    Fit a parametric distribution to the data using Maximum Likelihood Estimation.

    Estimates the best parameters for a given scipy distribution.
    Use it when you already have a hypothesis about which distribution
    your data follows and want to find its parameters.

    Example: you suspect response times follow an exponential distribution.
    → distribution_fit(response_times, dist="expon")
    → returns {"params": (loc, scale), "distribution": "expon"}

    Common distributions:
    - "norm"    : normal (μ, σ)
    - "expon"   : exponential (λ)
    - "lognorm" : log-normal
    - "gamma"   : gamma
    - "poisson" : Poisson (discrete)
    - "uniform" : uniform (a, b)

    See scipy.stats for the full list of available distributions.

    Null values are ignored.

    :param n: the series
    :param dist: scipy distribution name (default "norm")
    :return: dict with distribution name, fitted parameters, and log-likelihood
    """
    stats = _require("scipy.stats")
    clean = n.drop_nulls().to_numpy()

    distribution = getattr(stats, dist, None)
    if distribution is None:
        raise ValueError(f"Unknown distribution: '{dist}'. See scipy.stats for available names.")

    params = distribution.fit(clean)
    log_likelihood = distribution.logpdf(clean, *params).sum()

    return {
        "distribution": dist,
        "params": params,
        "log_likelihood": log_likelihood,
    }


def qqplot_data(n: pl.Series, dist: str = "norm") -> dict:
    """
    Generate QQ-plot coordinates (theoretical quantiles vs observed quantiles).

    A QQ-plot is the fastest visual diagnostic for checking if your data
    follows a given distribution. If the points fall on the diagonal,
    the fit is good. Deviations reveal where and how the data differs
    from the theoretical model.

    This function does NOT plot anything — it returns the coordinates
    so you can plot them with your preferred tool (matplotlib, plotly, etc.).

    Reading the output:
    - Points on the diagonal → data matches the distribution
    - Points curving away at the ends → heavy or light tails
    - Points curving to one side → skewness
    - S-shaped curve → different tail behavior than expected

    Null values are ignored.

    :param n: the series
    :param dist: scipy distribution to compare against (default "norm")
    :return: dict with theoretical quantiles, observed quantiles, and distribution info
    """
    stats = _require("scipy.stats")
    np = _require("numpy")
    clean = n.drop_nulls().to_numpy()

    distribution = getattr(stats, dist, None)
    if distribution is None:
        raise ValueError(f"Unknown distribution: '{dist}'. See scipy.stats for available names.")

    # Fit the distribution to get parameters
    params = distribution.fit(clean)

    # Sort observed values
    observed = np.sort(clean)

    # Generate theoretical quantiles
    n_len = len(observed)
    probabilities = [(i - 0.5) / n_len for i in range(1, n_len + 1)]
    theoretical = distribution.ppf(probabilities, *params)

    return {
        "theoretical": theoretical.tolist(),
        "observed": observed.tolist(),
        "distribution": dist,
        "params": params,
    }


def kde(n: pl.Series, n_points: int = 200) -> dict:
    """
    Estimate the probability density function using Kernel Density Estimation.

    KDE is a smoothed histogram — it shows the shape of your distribution
    without assuming any parametric model. Use it for exploratory analysis
    before deciding which distribution to fit.

    Use cases:
    - Visualizing the shape of your data (symmetric? bimodal? skewed?)
    - Comparing distributions between groups (overlay two KDEs)
    - Spotting patterns hidden by histograms (bimodality, gaps)

    Returns x values and their corresponding density estimates,
    ready to plot with your preferred tool.

    The bandwidth (smoothing parameter) is selected automatically
    using Scott's rule. More points = smoother curve.

    Null values are ignored.

    :param n: the series
    :param n_points: number of points to evaluate the density at (default 200)
    :return: dict with x values, density values, and bandwidth
    """
    stats = _require("scipy.stats")
    np = _require("numpy")
    clean = n.drop_nulls().to_numpy()

    kernel = stats.gaussian_kde(clean)

    x_min = clean.min()
    x_max = clean.max()
    padding = (x_max - x_min) * 0.1
    x = np.linspace(x_min - padding, x_max + padding, n_points)

    density = kernel(x)

    return {
        "x": x.tolist(),
        "density": density.tolist(),
        "bandwidth": float(kernel.factor),
    }