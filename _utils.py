def _require(module_name: str):
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"{module_name} is required for this function. "
            f"Install it with: pip install polars-stats[tests]"
        )