"""Contract-compatible test models.

These stand in for the model slice's real forecasters. They satisfy the
documented ``Forecaster`` protocol (``name``, ``fit(y) -> self``,
``predict(h) -> ndarray``, ``info() -> ModelInfo``) and are built on whatever
contract module is importable, so they work both before and after
``tsbench.base`` lands.
"""

from __future__ import annotations

import numpy as np

from tsbench.evaluation.contract import ModelInfo

__all__ = [
    "BrokenForecaster",
    "ConstantForecaster",
    "DriftForecaster",
    "NaiveStub",
    "SeasonalNaiveStub",
]


class _BaseStub:
    name = "stub"
    family = "baseline"
    zero_shot = True

    def __init__(self, **extra) -> None:
        self.extra = dict(extra)
        self._context: np.ndarray | None = None

    def fit(self, y):
        self._context = np.asarray(y, dtype=np.float64).reshape(-1)
        if self._context.size == 0:
            raise ValueError("cannot fit on an empty series")
        return self

    def _require_context(self) -> np.ndarray:
        if self._context is None:
            raise RuntimeError("forecaster must be fitted before predict")
        return self._context

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            family=self.family,
            zero_shot=self.zero_shot,
            params=0,
            license="Apache-2.0",
            revision="test-stub",
            extra=dict(self.extra),
        )


class NaiveStub(_BaseStub):
    name = "naive"

    def predict(self, h) -> np.ndarray:
        context = self._require_context()
        return np.full(int(h), float(context[-1]), dtype=np.float64)


class SeasonalNaiveStub(_BaseStub):
    name = "seasonal_naive"

    def __init__(self, season_length: int = 1, **extra) -> None:
        super().__init__(**extra)
        self.season_length = int(season_length)

    def predict(self, h) -> np.ndarray:
        context = self._require_context()
        season = self.season_length
        if context.size < season:
            value = float(context[-1])
            return np.full(int(h), value, dtype=np.float64)
        pattern = context[-season:]
        reps = int(np.ceil(int(h) / season))
        return np.tile(pattern, reps)[: int(h)].astype(np.float64)


class ConstantForecaster(_BaseStub):
    name = "constant"

    def __init__(self, value: float = 0.0, **extra) -> None:
        super().__init__(**extra)
        self.value = float(value)

    def predict(self, h) -> np.ndarray:
        self._require_context()
        return np.full(int(h), self.value, dtype=np.float64)


class DriftForecaster(_BaseStub):
    name = "drift"

    def predict(self, h) -> np.ndarray:
        context = self._require_context()
        if context.size < 2:
            return np.full(int(h), float(context[-1]), dtype=np.float64)
        slope = (context[-1] - context[0]) / (context.size - 1)
        steps = np.arange(1, int(h) + 1, dtype=np.float64)
        return context[-1] + slope * steps


class BrokenForecaster(_BaseStub):
    name = "broken"

    def __init__(self, **extra) -> None:
        raise RuntimeError("broken forecaster cannot be constructed")
