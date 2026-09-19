"""Frozen-interface access for the data/evaluation slices.

``tsbench.base`` is owned by the foundation slice and is the single source of
truth for ``Forecaster``, ``ModelInfo`` and the results schema. While that
module is being built in parallel it may not exist on this branch yet, so this
module resolves it exactly the way the model slice does (``tsbench.base`` first,
then a structurally identical local fallback) and exposes one import site for
the rest of the evaluation code. Once ``tsbench.base`` lands, the fallback is
never used and the imported names are byte-identical to the foundation ones.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "CONTRACT_SOURCE",
    "Forecaster",
    "ModelInfo",
    "RESULT_COLUMNS",
    "as_float_1d",
    "check_horizon",
    "result_columns",
]

_CONTRACT_MODULE = "tsbench.base"

# The schema is pinned here literally so this module works when the foundation
# contract is unavailable. When ``tsbench.base`` is importable its columns are
# used directly, so the fallback only matters off the foundation branch.
_FALLBACK_RESULT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "dataset",
    "series_id",
    "model",
    "family",
    "context_length",
    "horizon",
    "window_index",
    "mae",
    "rmse",
    "mase",
    "smape",
    "latency_ms",
    "peak_mem_mb",
    "train_seconds",
    "params",
    "zero_shot",
    "git_sha",
    "config_hash",
    "measured",
    "timestamp",
)


@dataclass(frozen=True)
class _FallbackModelInfo:
    name: str
    family: str
    zero_shot: bool
    params: int | float | str | None
    license: str
    revision: str
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class _FallbackForecaster(Protocol):
    name: str

    def fit(self, y: np.ndarray) -> _FallbackForecaster: ...

    def predict(self, h: int) -> np.ndarray: ...

    def info(self) -> _FallbackModelInfo: ...


def _load_contract() -> Any | None:
    try:
        if importlib.util.find_spec(_CONTRACT_MODULE) is None:
            return None
    except (ImportError, ValueError):
        return None
    try:
        module = importlib.import_module(_CONTRACT_MODULE)
    except ImportError:
        return None
    if hasattr(module, "Forecaster") and hasattr(module, "ModelInfo"):
        return module
    return None


_contract = _load_contract()
if _contract is not None:
    CONTRACT_SOURCE = _CONTRACT_MODULE
    Forecaster = _contract.Forecaster
    ModelInfo = _contract.ModelInfo
    _base_columns = getattr(_contract, "RESULT_COLUMNS", None)
    RESULT_COLUMNS: tuple[str, ...] = (
        tuple(_base_columns) if _base_columns is not None else _FALLBACK_RESULT_COLUMNS
    )
else:  # pragma: no cover - exercised only before the foundation branch lands
    CONTRACT_SOURCE = "tsbench.evaluation.contract:fallback"
    Forecaster = _FallbackForecaster
    ModelInfo = _FallbackModelInfo
    RESULT_COLUMNS = _FALLBACK_RESULT_COLUMNS


def result_columns() -> tuple[str, ...]:
    """Return the authoritative results schema."""
    return RESULT_COLUMNS


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
        horizon = int(h)
        if horizon != h:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("h must be a positive integer") from exc
    if horizon < 1:
        raise ValueError("h must be a positive integer")
    return horizon
