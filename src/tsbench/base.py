"""Frozen interfaces shared by every tsbench component.

This module is the fleet-wide contract. The data, model, and evaluation
slices import these names and must not redefine them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["RESULT_COLUMNS", "Forecaster", "ModelInfo"]


@dataclass(frozen=True)
class ModelInfo:
    """Static description of a forecaster, recorded with every result row."""

    name: str
    family: str
    zero_shot: bool
    params: int | float | str | None
    license: str
    revision: str
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Forecaster(Protocol):
    """Contract every model implementation satisfies.

    ``fit`` returns ``self`` so callers can chain ``model.fit(y).predict(h)``.
    ``predict`` returns a 1-D array of length ``h``.
    """

    name: str

    def fit(self, y: np.ndarray) -> Forecaster: ...

    def predict(self, h: int) -> np.ndarray: ...

    def info(self) -> ModelInfo: ...


RESULT_COLUMNS: tuple[str, ...] = (
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
