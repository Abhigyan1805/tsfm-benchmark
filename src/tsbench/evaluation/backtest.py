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
    params: int | float | str | None = None
    zero_shot: bool | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _cuda_meter() -> Any | None:
    """Return the torch module when a CUDA device is live, else ``None``.

    Peak memory is the GPU allocator's peak on a GPU host and the host
    ``tracemalloc`` peak otherwise. Importing torch is deferred so a CPU-only
    or torch-free worker never pays for it.
    """
    try:
        import torch
    except ImportError:
        return None
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None
    return torch


def _cuda_peak_memory(torch: Any) -> tuple[Callable[[], None], Callable[[], float | None]]:
    """Per-window CUDA allocator peak, baselined on live allocations."""
    state = {"baseline": 0}

    def start_window() -> None:
        torch.cuda.synchronize()
        state["baseline"] = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    def stop_window() -> float | None:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return max(0.0, (peak - state["baseline"]) / (1024 * 1024))

    return start_window, stop_window


def measure_peak_memory(
    *, enabled: bool = True
) -> tuple[Callable[[], None], Callable[[], float | None]]:
    """Start peak-memory tracking; return ``(start_window, stop_window)``.

    On a CUDA host the measured peak is the GPU allocator's live-allocation
    peak, re-baselined for every window (so one-time weight loading is charged
    only to the window that performed it). Without CUDA the host
    ``tracemalloc`` peak is used; it is process-global and accumulates since
    tracing started, so ``start_window`` re-baselines and resets it rather than
    smearing the largest window across every later row. When disabled, both
    halves are no-ops and no overhead is paid.
    """
    if not enabled:
        return (lambda: None), (lambda: None)
    torch = _cuda_meter()
    if torch is not None:
        return _cuda_peak_memory(torch)
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    baseline_mb = tracemalloc.get_traced_memory()[0] / (1024 * 1024)

    def start_window() -> None:
        nonlocal baseline_mb
        baseline_mb = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
        tracemalloc.reset_peak()

    def stop_window() -> float | None:
        _current, peak = tracemalloc.get_traced_memory()
        return max(0.0, peak / (1024 * 1024) - baseline_mb)

    return start_window, stop_window


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
    start_memory, stop_memory = measure_peak_memory(enabled=track_memory)
    results: list[WindowForecast] = []
    for index, (origin, context, target) in enumerate(windows):
        assert_no_future_data(plan, origin, plan.config.context_length, plan.config.horizon)
        start_memory()
        model = model_builder()
        params: Any = None
        zero_shot: bool | None = None
        context_values = np.array(context, dtype=np.float64, copy=True)
        target_values = np.array(target, dtype=np.float64, copy=True)
        train_seconds: float | None = None
        start = time.perf_counter()
        try:
            train_start = time.perf_counter()
            model.fit(context_values)
            train_seconds = time.perf_counter() - train_start
            # Read metadata after fit: trained families (deep) only know their
            # parameter count once the network is built.
            params, zero_shot = _model_metadata(model)
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
                    params=params,
                    zero_shot=zero_shot,
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
                params=params,
                zero_shot=zero_shot,
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


def _model_metadata(model: Any) -> tuple[Any, bool | None]:
    """Read ``params`` and ``zero_shot`` from a model's ``ModelInfo``.

    A model that does not expose ``info`` (or whose ``info`` raises) reports
    ``(None, None)`` rather than aborting the backtest; metadata is recorded, not
    required.
    """
    info = getattr(model, "info", None)
    if not callable(info):
        return None, None
    try:
        resolved = info()
    except Exception:
        return None, None
    params = getattr(resolved, "params", None)
    zero_shot = getattr(resolved, "zero_shot", None)
    return params, bool(zero_shot) if zero_shot is not None else None


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
