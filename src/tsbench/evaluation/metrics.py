"""Forecast accuracy metrics with explicit edge-case semantics.

The four metrics reported by every experiment are MAE, RMSE, seasonal MASE and
sMAPE. Each function documents its behaviour on the degenerate inputs that show
up in real benchmarks (empty windows, all-zero actuals, constant series, a
single observation) so a downstream NaN can always be traced to a named rule
rather than a library default.

Return values are ``float`` and may be ``nan``; ``nan`` means "undefined for
this window", which callers record rather than silently coercing to zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "mae",
    "rmse",
    "smape",
    "mase",
    "seasonal_naive_scale",
    "metric_set",
    "METRIC_NAMES",
]

METRIC_NAMES: tuple[str, ...] = ("mae", "rmse", "mase", "smape")

_EPS = 1e-12


def _as_pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=np.float64).reshape(-1)
    forecast = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if actual.size != forecast.size:
        raise ValueError(
            f"y_true and y_pred must have equal length, got {actual.size} and {forecast.size}"
        )
    return actual, forecast


def mae(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error.

    Empty input returns ``nan``. Non-finite entries are treated as missing and
    excluded; a window with no finite pairs returns ``nan``.
    """
    actual, forecast = _as_pair(y_true, y_pred)
    mask = np.isfinite(actual) & np.isfinite(forecast)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - forecast[mask])))


def rmse(y_true: Any, y_pred: Any) -> float:
    """Root mean squared error (population mean of squared errors)."""
    actual, forecast = _as_pair(y_true, y_pred)
    mask = np.isfinite(actual) & np.isfinite(forecast)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((actual[mask] - forecast[mask]) ** 2)))


def smape(y_true: Any, y_pred: Any) -> float:
    """Symmetric mean absolute percentage error, in **percent**.

    ``100 * mean( 2|a-f| / (|a|+|f|) )``. A pair whose denominator is zero
    (both values zero, or opposite infinities) contributes ``0.0`` rather than
    raising or producing ``inf`` -- Hyndman & Koehler's convention. Empty input
    and all-degenerate input return ``nan``.
    """
    actual, forecast = _as_pair(y_true, y_pred)
    mask = np.isfinite(actual) & np.isfinite(forecast)
    if not np.any(mask):
        return float("nan")
    a = actual[mask]
    f = forecast[mask]
    denominator = np.abs(a) + np.abs(f)
    numerator = 2.0 * np.abs(a - f)
    terms = np.zeros_like(denominator)
    nonzero = denominator > _EPS
    terms[nonzero] = numerator[nonzero] / denominator[nonzero]
    return float(100.0 * np.mean(terms))


def seasonal_naive_scale(y: Any, season_length: int = 1, *, in_sample: bool = True) -> float:
    """Mean absolute seasonal difference used as the MASE denominator.

    With ``in_sample=True`` the scale is ``mean(|y_t - y_{t-m}|)`` over the
    supplied series (the standard in-sample variant). With
    ``in_sample=False`` the scale is computed from the first ``m`` values of the
    *forecast-period* actuals, which is only valid when ``y`` is the actual
    future block. Returns ``nan`` when fewer than ``season_length + 1`` finite
    observations exist, and ``0.0`` for a perfectly seasonal-constant series.
    """
    array = np.asarray(y, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if season_length < 1:
        raise ValueError("season_length must be a positive integer")
    if array.size <= season_length:
        return float("nan")
    differences = np.abs(array[season_length:] - array[:-season_length])
    if differences.size == 0:
        return float("nan")
    return float(np.mean(differences))


def mase(
    y_true: Any,
    y_pred: Any,
    *,
    y_train: Any | None = None,
    season_length: int = 1,
    scale: float | None = None,
) -> float:
    """Mean absolute scaled error with a seasonal naive scale.

    The denominator is, in priority order: an explicit ``scale`` argument, the
    in-sample seasonal scale of ``y_train`` when supplied, or the in-sample
    scale of ``y_true``. An all-zero or empty denominator yields ``nan`` when
    the numerator is also zero (very common for the zero series) and ``0.0``
    when the forecast is exact; otherwise the raw scaled error is returned,
    which is well-defined and large.
    """
    actual, forecast = _as_pair(y_true, y_pred)
    if scale is None:
        reference = y_train if y_train is not None else actual
        scale = seasonal_naive_scale(reference, season_length)
    if scale is None or not np.isfinite(scale):
        return float("nan")
    numerator = mae(actual, forecast)
    if not np.isfinite(numerator):
        return float("nan")
    if abs(scale) <= _EPS:
        return 0.0 if numerator <= _EPS else float("inf")
    return float(numerator / scale)


def metric_set(
    y_true: Any,
    y_pred: Any,
    *,
    y_train: Any | None = None,
    season_length: int = 1,
) -> dict[str, float]:
    """Compute all four headline metrics for one window."""
    actual, forecast = _as_pair(y_true, y_pred)
    return {
        "mae": mae(actual, forecast),
        "rmse": rmse(actual, forecast),
        "mase": mase(actual, forecast, y_train=y_train, season_length=season_length),
        "smape": smape(actual, forecast),
    }
