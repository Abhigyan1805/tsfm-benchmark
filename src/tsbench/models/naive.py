"""Random-walk reference floors: naive and seasonal naive."""

from __future__ import annotations

import operator

import numpy as np

from tsbench.models import Forecaster, ModelInfo, as_float_1d, check_horizon


class Naive(Forecaster):
    """Repeat the last observed value for every forecast step."""

    name = "naive"
    family = "baseline"

    def __init__(self) -> None:
        self._last_value: float | None = None

    def fit(self, y) -> Naive:
        series = as_float_1d(y)
        self._last_value = float(series[-1])
        return self

    def _require_fitted(self) -> float:
        if self._last_value is None:
            raise RuntimeError("Naive must be fitted before predict")
        return self._last_value

    def predict(self, h) -> np.ndarray:
        horizon = check_horizon(h)
        last_value = self._require_fitted()
        return np.full(horizon, last_value, dtype=np.float64)

    def info(self) -> ModelInfo:
        extra = {"method": "last_value", "season_length": 1}
        if self._last_value is not None:
            extra["last_value"] = self._last_value
        return ModelInfo(
            name=self.name,
            family=self.family,
            zero_shot=True,
            params=0,
            license="Apache-2.0",
            revision="repo-local:0.0.1",
            extra=extra,
        )


class SeasonalNaive(Forecaster):
    """Repeat the last observed season of ``season_length`` steps."""

    name = "seasonal_naive"
    family = "baseline"

    def __init__(self, season_length: int = 1) -> None:
        try:
            season_length = operator.index(season_length)
        except TypeError as exc:
            raise TypeError("season_length must be an integer") from exc
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.season_length = int(season_length)
        self._season: np.ndarray | None = None

    def fit(self, y) -> SeasonalNaive:
        series = as_float_1d(y)
        window = min(self.season_length, series.size)
        self._season = series[-window:].astype(np.float64, copy=True)
        return self

    def _require_fitted(self) -> np.ndarray:
        if self._season is None:
            raise RuntimeError("SeasonalNaive must be fitted before predict")
        return self._season

    def predict(self, h) -> np.ndarray:
        horizon = check_horizon(h)
        season = self._require_fitted()
        repeats = int(np.ceil(horizon / season.size))
        return np.tile(season, repeats)[:horizon].astype(np.float64, copy=False)

    def info(self) -> ModelInfo:
        extra = {
            "method": "seasonal_last_value",
            "season_length": self.season_length,
            "effective_season_length": None if self._season is None else int(self._season.size),
        }
        return ModelInfo(
            name=self.name,
            family=self.family,
            zero_shot=True,
            params=0,
            license="Apache-2.0",
            revision="repo-local:0.0.1",
            extra=extra,
        )
