"""
Internal utilities for polars_stats.
Not part of the public API.
"""

import importlib
import polars as pl



def _require(module_name: str):
    """
    Lazy import: loads a module at call time instead of import time.

    If the module is not installed, raises a clear error message
    telling the user which extra to install.

    :param module_name: dotted module path (e.g. "scipy.stats")
    :return: the imported module
    """
    extras_map = {
        "scipy": "tests",
        "statsmodels": "tests",
        "sklearn": "full",
        "scikit_posthocs": "full",
        "pingouin": "full",
    }

    try:
        return importlib.import_module(module_name)
    except ImportError:
        root = module_name.split(".")[0]
        extra = extras_map.get(root, "full")
        raise ImportError(
            f"'{module_name}' is required for this function but is not installed.\n"
            f"Install it with: pip install polars-stats[{extra}]"
        )


def to_clean_array(s: pl.Series):
    """
    Convert a Polars Series to a clean NumPy array with nulls removed.

    Centralizes null handling so individual functions don't have to
    repeat drop_nulls().to_numpy() everywhere.

    :param s: the Polars Series
    :return: NumPy array with no null values
    """
    return s.drop_nulls().to_numpy()