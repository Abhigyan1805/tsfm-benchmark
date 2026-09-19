"""Forecasting models for tsfm-benchmark.

Concrete model classes live in sibling modules and implement the frozen
``Forecaster`` contract owned by ``tsbench.base``. This package re-exports that
contract alongside the shared input-validation helpers used by every model
module.
"""

from __future__ import annotations

import importlib
import operator
from typing import Any

import numpy as np

from tsbench.base import Forecaster, ModelInfo


def as_float_1d(y: Any) -> np.ndarray:
    """Return a defensive, finite, float64 copy of a 1-D series."""
    array = np.asarray(y, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("y must contain at least one observation")
    if not np.all(np.isfinite(array)):
        raise ValueError("y must contain only finite values")
    return array.astype(np.float64, copy=True)


def check_horizon(h: Any) -> int:
    """Validate and normalise a forecast horizon."""
    if isinstance(h, (bool, np.bool_)):
        raise ValueError("h must be a positive integer")
    try:
        horizon = operator.index(h)
    except TypeError as exc:
        raise ValueError("h must be a positive integer") from exc
    if horizon < 1:
        raise ValueError("h must be a positive integer")
    return int(horizon)


_LAZY_EXPORTS = {
    "Naive": ("tsbench.models.naive", "Naive"),
    "SeasonalNaive": ("tsbench.models.naive", "SeasonalNaive"),
    "AutoETS": ("tsbench.models.stats", "AutoETS"),
    "AutoARIMA": ("tsbench.models.stats", "AutoARIMA"),
    "XGBoostLags": ("tsbench.models.gbdt", "XGBoostLags"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


__all__ = [
    "Forecaster",
    "ModelInfo",
    "as_float_1d",
    "check_horizon",
    "Naive",
    "SeasonalNaive",
    "AutoETS",
    "AutoARIMA",
    "XGBoostLags",
]
