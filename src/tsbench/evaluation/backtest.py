"""Rolling-origin backtest over frozen splits.

The engine walks an ordered list of windows (each a context + target carved by
:mod:`tsbench.data.splits`) and, for every window and model, fits on the context
only, predicts the horizon, and scores the prediction. It owns the timing and
memory measurement so every family is measured the same way.

The engine is deliberately model-agnostic: any object with the
``Forecaster`` shape works, and the caller supplies already-constructed models.
That keeps model lifecycle (license gates, caching, GPU placement) in the
runner, where it belongs.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.splits import (
    SplitPlan,
    assert_no_future_data,
    iter_windows,
)
from . import metrics as _metrics

__all__ = [
    "BacktestError",
    "WindowForecast",
    "backtest_model",
    "backtest_windows",
    "measure_peak_memory",
]

_EPS = 1e-12


class BacktestError(RuntimeError):
    """A backtest could not be completed consistently."""


@dataclass(frozen=True)
class WindowForecast:
    """The scored outcome of one model on one window."""

    series_id: str
    model: str
    family: str
    window_index: int
    origin: int
    context_length: int
    horizon: int
    forecast: np.ndarray
    target: np.ndarray
    metrics: dict[str, float]
    latency_ms: float
    peak_mem_mb: float | None
    train_seconds: float | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def measure_peak_memory(*, enabled: bool = True) -> tuple[float | None, Callable[[], float | None]]:
    """Start peak-memory tracking; return ``(baseline_mb, stop_mb)``.

    ``tracemalloc`` is process-global, so callers enable it once per run and
    read the peak around each fit/predict pair. When disabled, both halves
    return ``None`` and no overhead is paid.
    """
    if not enabled:
        return None, lambda: None
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[1] / (1024 * 1024)

    def stop() -> float | None:
        current, peak = tracemalloc.get_traced_memory()
        return max(0.0, peak / (1024 * 1024) - baseline)

    return baseline, stop


def backtest_windows(
    values: np.ndarray,
    plan: SplitPlan,
    model_builder: Callable[[], Any],
    *,
    series_id: str,
    period: str | Sequence[str] | None = None,
    metrics_fn: Callable[..., dict[str, float]] = _metrics.metric_set,
    season_length: int = 1,
    track_memory: bool = True,
    model_name: str | None = None,
    family: str | None = None,
) -> list[WindowForecast]:
    """Run one model across every window of ``plan``.

    A failing model on one window is recorded with ``error`` set and does not
    abort the remaining windows; a caller who wants strictness can check
    :attr:`WindowForecast.ok` on the returned rows.
    """
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    displayed_name, displayed_family = _resolve_labels(
        model_builder, model_name=model_name, family=family
    )
    windows = iter_windows(series, plan, period=period)
    _, stop_memory = measure_peak_memory(enabled=track_memory)
    results: list[WindowForecast] = []
    for index, (origin, context, target) in enumerate(windows):
        assert_no_future_data(plan, origin, plan.config.context_length, plan.config.horizon)
        model = model_builder()
        context_values = np.array(context, dtype=np.float64, copy=True)
        target_values = np.array(target, dtype=np.float64, copy=True)
        train_seconds: float | None = None
        start = time.perf_counter()
        try:
            train_start = time.perf_counter()
            model.fit(context_values)
            train_seconds = time.perf_counter() - train_start
            forecast = np.asarray(model.predict(plan.config.horizon), dtype=np.float64).reshape(-1)
            latency_ms = (time.perf_counter() - start) * 1000.0
            if forecast.size != target_values.size:
                raise BacktestError(
                    f"model {displayed_name!r} predicted {forecast.size} steps, "
                    f"expected {target_values.size}"
                )
        except Exception as exc:  # recorded, not raised: one bad window is data
            latency_ms = (time.perf_counter() - start) * 1000.0
            results.append(
                WindowForecast(
                    series_id=series_id,
                    model=displayed_name,
                    family=displayed_family,
                    window_index=index,
                    origin=origin,
                    context_length=plan.config.context_length,
                    horizon=plan.config.horizon,
                    forecast=np.full(plan.config.horizon, np.nan),
                    target=target_values,
                    metrics=_metrics.metric_set(
                        target_values, np.full(plan.config.horizon, np.nan)
                    ),
                    latency_ms=latency_ms,
                    peak_mem_mb=None,
                    train_seconds=train_seconds,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            gc.collect()
            continue
        score = metrics_fn(
            target_values,
            forecast,
            y_train=context_values,
            season_length=season_length,
        )
        results.append(
            WindowForecast(
                series_id=series_id,
                model=displayed_name,
                family=displayed_family,
                window_index=index,
                origin=origin,
                context_length=plan.config.context_length,
                horizon=plan.config.horizon,
                forecast=forecast,
                target=target_values,
                metrics=score,
                latency_ms=latency_ms,
                peak_mem_mb=stop_memory(),
                train_seconds=train_seconds,
            )
        )
        gc.collect()
    return results


def backtest_model(
    values: np.ndarray,
    plan: SplitPlan,
    model: Any,
    *,
    series_id: str,
    period: str | Sequence[str] | None = None,
    metrics_fn: Callable[..., dict[str, float]] = _metrics.metric_set,
    season_length: int = 1,
    track_memory: bool = True,
) -> list[WindowForecast]:
    """Run an already-constructed model once per window.

    The same instance is refit on each window; use this when the model is
    expensive to build but cheap to refit, and :func:`backtest_windows` when a
    fresh instance is required per window.
    """
    model_name = getattr(model, "name", type(model).__name__)
    family = getattr(model, "family", "unknown")
    return backtest_windows(
        values,
        plan,
        lambda: model,
        series_id=series_id,
        period=period,
        metrics_fn=metrics_fn,
        season_length=season_length,
        track_memory=track_memory,
        model_name=model_name,
        family=family,
    )


def _resolve_labels(
    model_builder: Callable[[], Any],
    *,
    model_name: str | None,
    family: str | None,
) -> tuple[str, str]:
    if model_name is not None and family is not None:
        return model_name, family
    probe = model_builder()
    resolved_name = model_name or getattr(probe, "name", None) or type(probe).__name__
    resolved_family = family or getattr(probe, "family", None)
    if not resolved_family:
        info = getattr(probe, "info", None)
        if callable(info):
            try:
                resolved_family = info().family
            except Exception:
                resolved_family = None
    return str(resolved_name), str(resolved_family or "unknown")
