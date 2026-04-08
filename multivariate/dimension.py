import polars as pl
from _utils import _require


def pca(df: pl.DataFrame, features: list = None, n_components: int = 2) -> dict:
    """
    Principal Component Analysis (PCA).

    Reduces the dimensionality of your data by finding new axes (components)
    that capture the most variance. The first component captures the most
    variance, the second captures the most remaining variance orthogonal
    to the first, and so on.

    Use it to:
    - Visualize high-dimensional data in 2D or 3D
    - Remove multicollinearity before regression
    - Compress features while keeping most information
    - Identify which original variables drive the main patterns

    Reading the output:
    - explained_variance_ratio: how much each component captures
      (e.g. [0.72, 0.18] → first two components capture 90% of the info)
    - loadings: how much each original variable contributes to each component
      (high absolute loading = important variable for that component)
    - scores: the transformed data points in the new coordinate system

    Requires scikit-learn.

    :param df: the DataFrame
    :param features: list of feature column names (default: all numeric columns)
    :param n_components: number of components to keep (default 2)
    :return: dict with scores, loadings, explained variance, and component info
    """
    sklearn_decomp = _require("sklearn.decomposition")
    sklearn_preproc = _require("sklearn.preprocessing")
    np = _require("numpy")

    if features is None:
        features = df.select(pl.col(pl.NUMERIC_DTYPES)).columns

    data = df.select(features).drop_nulls().to_numpy()

    # Standardize (PCA is sensitive to scale)
    scaler = sklearn_preproc.StandardScaler()
    data_scaled = scaler.fit_transform(data)

    model = sklearn_decomp.PCA(n_components=n_components)
    scores = model.fit_transform(data_scaled)

    component_names = [f"PC{i+1}" for i in range(n_components)]

    loadings = {
        component_names[i]: {
            feature: float(loading)
            for feature, loading in zip(features, model.components_[i])
        }
        for i in range(n_components)
    }

    return {
        "scores": scores.tolist(),
        "loadings": loadings,
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        "cumulative_variance": float(model.explained_variance_ratio_.sum()),
        "n_components": n_components,
        "n_features": len(features),
        "features": features,
    }


def scree_data(df: pl.DataFrame, features: list = None) -> dict:
    """
    Generate scree plot data (eigenvalues per component).

    A scree plot shows the explained variance of each principal component.
    Use it to decide how many components to keep in PCA.

    Rules of thumb:
    - Kaiser criterion: keep components with eigenvalue > 1
    - Elbow method: keep components before the "elbow" where the curve flattens
    - Cumulative threshold: keep enough components to reach 80-90% cumulative variance

    This function does NOT plot — it returns the data so you can plot
    with your preferred tool.

    Requires scikit-learn.

    :param df: the DataFrame
    :param features: list of feature column names (default: all numeric columns)
    :return: dict with eigenvalues, explained variance ratio, and cumulative variance
    """
    sklearn_decomp = _require("sklearn.decomposition")
    sklearn_preproc = _require("sklearn.preprocessing")
    np = _require("numpy")

    if features is None:
        features = df.select(pl.col(pl.NUMERIC_DTYPES)).columns

    data = df.select(features).drop_nulls().to_numpy()

    scaler = sklearn_preproc.StandardScaler()
    data_scaled = scaler.fit_transform(data)

    model = sklearn_decomp.PCA()
    model.fit(data_scaled)

    cumulative = np.cumsum(model.explained_variance_ratio_)

    return {
        "eigenvalues": model.explained_variance_.tolist(),
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumulative.tolist(),
        "n_components": len(features),
        "kaiser_criterion": int((model.explained_variance_ > 1).sum()),
        "components_for_80pct": int(np.argmax(cumulative >= 0.80) + 1),
        "components_for_90pct": int(np.argmax(cumulative >= 0.90) + 1),
    }