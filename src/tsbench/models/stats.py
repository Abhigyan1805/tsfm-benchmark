"""Classical ETS and ARIMA forecasters.

``statsforecast`` is preferred when installed; otherwise a documented
``statsmodels`` fallback selects ETS parameters by AIC and ARIMA orders by AIC
over a small grid with the integration order chosen by KPSS. The engine that
actually ran is recorded in ``info().params["engine"]``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import operator
import warnings
from typing import Any

import numpy as np

from tsbench.models import Forecaster, ModelInfo, as_float_1d, check_horizon

_ENGINES = ("auto", "statsforecast", "statsmodels")
_ENGINE_LICENSES = {
    "statsforecast": "Apache-2.0",
    "statsmodels": "BSD-3-Clause",
    "naive": "Apache-2.0",
    "unavailable": "unknown",
}
_ENGINE_PACKAGES = {"statsforecast": "statsforecast", "statsmodels": "statsmodels"}


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _engine_version(engine: str) -> str:
    package = _ENGINE_PACKAGES.get(engine)
    if package is None:
        return "1"
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class _FittedModel:
    engine = "unavailable"
    detail: dict[str, Any]

    def forecast(self, h: int) -> np.ndarray:
        raise NotImplementedError


class _StatsforecastFit(_FittedModel):
    engine = "statsforecast"

    def __init__(self, model: Any) -> None:
        self.model = model
        self.detail = {}

    def forecast(self, h: int) -> np.ndarray:
        output = self.model.predict(h=h)
        return np.asarray(output["mean"], dtype=np.float64).reshape(-1)


class _StatsmodelsFit(_FittedModel):
    engine = "statsmodels"

    def __init__(self, results: Any, detail: dict[str, Any] | None = None) -> None:
        self.results = results
        self.detail = dict(detail or {})

    def forecast(self, h: int) -> np.ndarray:
        return np.asarray(self.results.forecast(h), dtype=np.float64).reshape(-1)


class _NaiveFit(_FittedModel):
    engine = "naive"

    def __init__(self, value: float, reason: str) -> None:
        self.value = float(value)
        self.detail = {"fallback": reason}

    def forecast(self, h: int) -> np.ndarray:
        return np.full(h, self.value, dtype=np.float64)


def _select_integration_order(y: np.ndarray) -> int:
    if y.size < 8:
        return 0
    try:
        from statsmodels.tsa.stattools import kpss

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pvalue = float(kpss(y, regression="c", nlags="auto")[1])
    except Exception:
        return 0
    if not np.isfinite(pvalue):
        return 0
    return 1 if pvalue < 0.05 else 0


def _fit_ets_statsmodels(y: np.ndarray, season_length: int) -> _FittedModel:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    if y.size < 2:
        return _NaiveFit(float(y[-1]), "insufficient_observations")

    trend_options = [(None, False), ("add", False), ("add", True)]
    seasonal_options: list[str | None] = [None]
    if season_length > 1 and y.size >= 2 * season_length:
        seasonal_options.append("add")

    best_aic: float | None = None
    best_results: Any = None
    best_detail: dict[str, Any] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for trend, damped in trend_options:
            for seasonal in seasonal_options:
                try:
                    model = ExponentialSmoothing(
                        y,
                        trend=trend,
                        damped_trend=damped,
                        seasonal=seasonal,
                        seasonal_periods=season_length if seasonal else None,
                        initialization_method="estimated",
                    )
                    results = model.fit(optimized=True)
                    aic = float(results.aic)
                except Exception:
                    continue
                if not np.isfinite(aic):
                    continue
                if best_aic is None or aic < best_aic:
                    best_aic = aic
                    best_results = results
                    best_detail = {
                        "trend": trend,
                        "damped_trend": damped,
                        "seasonal": seasonal,
                    }

    if best_results is None:
        return _NaiveFit(float(y[-1]), "ets_fit_failed")

    best_detail["aic"] = best_aic
    best_detail["season_length"] = season_length
    return _StatsmodelsFit(best_results, best_detail)


def _fit_arima_statsmodels(y: np.ndarray, season_length: int) -> _FittedModel:
    from statsmodels.tsa.arima.model import ARIMA

    if y.size < 2:
        return _NaiveFit(float(y[-1]), "insufficient_observations")

    order_d = _select_integration_order(y)
    trend = "c" if order_d == 0 else "t"

    best_aic: float | None = None
    best_results: Any = None
    best_order: tuple[int, int, int] | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(3):
            for q in range(3):
                try:
                    results = ARIMA(y, order=(p, order_d, q), trend=trend).fit()
                    aic = float(results.aic)
                except Exception:
                    continue
                if not np.isfinite(aic):
                    continue
                if best_aic is None or aic < best_aic:
                    best_aic = aic
                    best_results = results
                    best_order = (p, order_d, q)

    if best_results is None:
        return _NaiveFit(float(y[-1]), "arima_fit_failed")

    detail = {
        "order": best_order,
        "seasonal_order": None,
        "season_length": season_length,
        "aic": best_aic,
    }
    return _StatsmodelsFit(best_results, detail)


def _fit_ets(y: np.ndarray, season_length: int, engine: str) -> _FittedModel:
    if engine == "statsforecast":
        from statsforecast.models import AutoETS

        try:
            model = AutoETS(season_length=season_length)
            model.fit(y)
            return _StatsforecastFit(model)
        except NotImplementedError as exc:
            if not _module_available("statsmodels"):
                raise
            fitted = _fit_ets_statsmodels(y, season_length)
            fitted.detail["statsforecast_error"] = f"{type(exc).__name__}: {exc}"
            return fitted
    return _fit_ets_statsmodels(y, season_length)


def _fit_arima(y: np.ndarray, season_length: int, engine: str) -> _FittedModel:
    if engine == "statsforecast":
        from statsforecast.models import AutoARIMA

        try:
            model = AutoARIMA(season_length=season_length)
            model.fit(y)
            return _StatsforecastFit(model)
        except NotImplementedError as exc:
            if not _module_available("statsmodels"):
                raise
            fitted = _fit_arima_statsmodels(y, season_length)
            fitted.detail["statsforecast_error"] = f"{type(exc).__name__}: {exc}"
            return fitted
    return _fit_arima_statsmodels(y, season_length)


class _AutoClassical(Forecaster):
    family = "classical"
    _method = ""

    def __init__(self, season_length: int = 1, engine: str = "auto") -> None:
        try:
            season_length = operator.index(season_length)
        except TypeError as exc:
            raise TypeError("season_length must be an integer") from exc
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        if engine not in _ENGINES:
            raise ValueError(f"engine must be one of {_ENGINES!r}, got {engine!r}")
        self.season_length = int(season_length)
        self.engine = engine
        self._fit_result: _FittedModel | None = None

    def _fit_model(self, y: np.ndarray, engine: str) -> _FittedModel:
        raise NotImplementedError

    def _resolve_engine(self) -> str:
        if self.engine != "auto":
            if not _module_available(_ENGINE_PACKAGES[self.engine]):
                raise ImportError(
                    f"{self.name} requested engine={self.engine!r} but "
                    f"{_ENGINE_PACKAGES[self.engine]!r} is not installed"
                )
            return self.engine
        if _module_available("statsforecast"):
            return "statsforecast"
        if _module_available("statsmodels"):
            return "statsmodels"
        raise ImportError(f"{self.name} requires statsforecast or statsmodels")

    def _engine_for_info(self) -> str:
        if self.engine != "auto":
            return self.engine
        if _module_available("statsforecast"):
            return "statsforecast"
        if _module_available("statsmodels"):
            return "statsmodels"
        return "unavailable"

    def fit(self, y) -> _AutoClassical:
        series = as_float_1d(y)
        engine = self._resolve_engine()
        self._fit_result = self._fit_model(series, engine)
        return self

    def predict(self, h) -> np.ndarray:
        horizon = check_horizon(h)
        if self._fit_result is None:
            raise RuntimeError(f"{self.name} must be fitted before predict")
        output = np.asarray(self._fit_result.forecast(horizon), dtype=np.float64)
        if output.shape != (horizon,):
            raise RuntimeError(
                f"{self.name} expected predictions with shape {(horizon,)}, "
                f"got {output.shape}"
            )
        if not np.all(np.isfinite(output)):
            raise RuntimeError(f"{self.name} produced non-finite predictions")
        return output

    def info(self) -> ModelInfo:
        result = self._fit_result
        engine = result.engine if result is not None else self._engine_for_info()
        extra: dict[str, Any] = {
            "method": self._method,
            "engine": engine,
            "season_length": self.season_length,
        }
        if result is not None:
            extra.update(result.detail)
        return ModelInfo(
            name=self.name,
            family=self.family,
            zero_shot=False,
            params=None,
            license=_ENGINE_LICENSES.get(engine, "unknown"),
            revision=_engine_version(engine),
            extra=extra,
        )


class AutoETS(_AutoClassical):
    """Automatic exponential smoothing (ETS) selected by AIC."""

    name = "auto_ets"
    _method = "AutoETS"

    def _fit_model(self, y: np.ndarray, engine: str) -> _FittedModel:
        return _fit_ets(y, self.season_length, engine)


class AutoARIMA(_AutoClassical):
    """Automatic ARIMA order selection (seasonal when statsforecast runs)."""

    name = "auto_arima"
    _method = "AutoARIMA"

    def _fit_model(self, y: np.ndarray, engine: str) -> _FittedModel:
        return _fit_arima(y, self.season_length, engine)
