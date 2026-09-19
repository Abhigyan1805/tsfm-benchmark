"""tsbench: reproducible, cost-aware benchmarking of time-series forecasters."""

from .base import RESULT_COLUMNS, Forecaster, ModelInfo

__all__ = ["RESULT_COLUMNS", "Forecaster", "ModelInfo", "__version__"]

__version__ = "0.1.0"
