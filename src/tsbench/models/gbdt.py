"""Gradient-boosted trees on lag and calendar features.

Features for a target time are built only from values strictly before that
time, and multi-step forecasts are produced recursively so no actual future
value ever enters the feature matrix. ``use_calendar`` toggles the ablation
between lags-only and lags+calendar features.
"""

from __future__ import annotations

import importlib.metadata
import operator
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from tsbench.models import Forecaster, ModelInfo, as_float_1d, check_horizon


def normalise_lags(lags: Any) -> tuple[int, ...]:
    if isinstance(lags, (bool, np.bool_)):
        raise TypeError("lags must be an int or a sequence of ints")
    if isinstance(lags, (int, np.integer)):
        sequence = list(range(1, operator.index(lags) + 1))
    else:
        try:
            sequence = [operator.index(value) for value in lags]
        except TypeError as exc:
            raise TypeError("lags must be an int or a sequence of ints") from exc
    if not sequence:
        raise ValueError("lags must not be empty")
    if any(lag < 1 for lag in sequence):
        raise ValueError("lags must be >= 1")
    return tuple(sorted(set(sequence)))


class _CalendarIndex:
    """Deterministic calendar features for past and future integer positions."""

    def __init__(self, index: Any, size: int) -> None:
        if isinstance(index, pd.DatetimeIndex) and len(index) == size:
            self.index: pd.DatetimeIndex | None = pd.DatetimeIndex(index)
        else:
            self.index = None
        self.size = int(size)
        self.freq = self._infer_freq()
        self.include_time_of_day = self._detect_time_of_day()

    @property
    def mode(self) -> str:
        if self.index is None or self.freq is None:
            return "position"
        return "datetime"

    def _detect_time_of_day(self) -> bool:
        if self.index is None or self.freq is None:
            return False
        hour = self.index.hour.to_numpy()
        minute = self.index.minute.to_numpy()
        return bool(np.any(hour != 0)) or bool(np.any(minute != 0))

    def _infer_freq(self) -> Any:
        if self.index is None or self.size < 2:
            return None
        try:
            inferred = pd.infer_freq(self.index)
        except ValueError:
            inferred = None
        if inferred is not None:
            return pd.tseries.frequencies.to_offset(inferred)
        deltas = pd.Series(self.index).diff().dropna()
        if len(deltas) > 0 and deltas.nunique() == 1:
            try:
                return pd.tseries.frequencies.to_offset(deltas.iloc[0])
            except ValueError:
                return None
        return None

    def _timestamps(self, positions: np.ndarray) -> pd.DatetimeIndex:
        assert self.index is not None and self.freq is not None
        last = self.index[self.size - 1]
        stamps = [
            self.index[int(position)]
            if position < self.size
            else last + self.freq * int(position - (self.size - 1))
            for position in positions
        ]
        return pd.DatetimeIndex(stamps)

    def matrix(self, positions: np.ndarray) -> tuple[np.ndarray, list[str]]:
        positions = np.asarray(positions, dtype=np.int64)
        if self.mode == "position":
            return positions.astype(np.float64).reshape(-1, 1), ["t"]

        timestamps = self._timestamps(positions)
        day_of_year = timestamps.dayofyear.to_numpy(dtype=np.float64)
        angle = 2.0 * np.pi * day_of_year / 365.25
        columns = [
            np.sin(angle),
            np.cos(angle),
            timestamps.dayofweek.to_numpy(dtype=np.float64),
            timestamps.month.to_numpy(dtype=np.float64),
        ]
        names = ["doy_sin", "doy_cos", "dow", "month"]
        if self.include_time_of_day:
            hour = timestamps.hour.to_numpy(dtype=np.float64)
            hour_angle = 2.0 * np.pi * hour / 24.0
            columns.extend([np.sin(hour_angle), np.cos(hour_angle)])
            names.extend(["hod_sin", "hod_cos"])
        return np.column_stack(columns), names


def _xgboost_version() -> str:
    try:
        return importlib.metadata.version("xgboost")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class XGBoostLags(Forecaster):
    """XGBoost regressor over lagged and calendar features."""

    name = "xgboost_lags"
    family = "ml"

    def __init__(
        self,
        lags: int | Sequence[int] = (1, 2, 3, 7, 14),
        use_calendar: bool = True,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        random_state: int = 0,
        n_jobs: int = 1,
    ) -> None:
        self.lags = normalise_lags(lags)
        self.max_lag = max(self.lags)
        self.use_calendar = bool(use_calendar)
        self.n_estimators = operator.index(n_estimators)
        self.max_depth = operator.index(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.min_child_weight = float(min_child_weight)
        self.reg_lambda = float(reg_lambda)
        self.random_state = operator.index(random_state)
        self.n_jobs = operator.index(n_jobs)
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not 0 < self.subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")
        if not 0 < self.colsample_bytree <= 1:
            raise ValueError("colsample_bytree must be in (0, 1]")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be non-zero")
        self._history: np.ndarray | None = None
        self._calendar: _CalendarIndex | None = None
        self._model: Any = None
        self._fallback: str | None = None

    def _design(self, history: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, list[str]]:
        positions = np.asarray(positions, dtype=np.int64)
        columns = [history[positions - lag] for lag in self.lags]
        names = [f"lag_{lag}" for lag in self.lags]
        if self.use_calendar:
            assert self._calendar is not None
            calendar, calendar_names = self._calendar.matrix(positions)
            columns.extend(calendar[:, column] for column in range(calendar.shape[1]))
            names.extend(calendar_names)
        return np.column_stack(columns), names

    def fit(self, y) -> XGBoostLags:
        history = as_float_1d(y)
        index = getattr(y, "index", None)
        self._history = history
        self._calendar = _CalendarIndex(index, history.size)
        self._fallback = None
        self._model = None
        if history.size <= self.max_lag:
            self._fallback = "insufficient_history_for_lags"
            return self

        positions = np.arange(self.max_lag, history.size)
        features, _ = self._design(history, positions)
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",
            objective="reg:squarederror",
            verbosity=0,
        )
        model.fit(features, history[positions])
        self._model = model
        return self

    def predict(self, h) -> np.ndarray:
        horizon = check_horizon(h)
        if self._history is None:
            raise RuntimeError("XGBoostLags must be fitted before predict")
        if self._model is None:
            return np.full(horizon, float(self._history[-1]), dtype=np.float64)

        size = self._history.size
        buffer = np.empty(size + horizon, dtype=np.float64)
        buffer[:size] = self._history
        for step in range(horizon):
            position = size + step
            features, _ = self._design(buffer[:position], np.array([position]))
            buffer[position] = float(self._model.predict(features)[0])
        return buffer[size:].copy()

    def info(self) -> ModelInfo:
        extra: dict[str, Any] = {
            "lags": list(self.lags),
            "use_calendar": self.use_calendar,
            "calendar_mode": self._calendar.mode if self._calendar is not None else None,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "fallback": self._fallback,
        }
        return ModelInfo(
            name=self.name,
            family=self.family,
            zero_shot=False,
            params=None,
            license="Apache-2.0",
            revision=_xgboost_version(),
            extra=extra,
        )
