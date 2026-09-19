"""Metric correctness and edge-case semantics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tsbench.evaluation import metrics


def test_mae_exact_value():
    assert metrics.mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert metrics.mae([0.0, 0.0], [1.0, -1.0]) == 1.0


def test_rmse_matches_definition_and_penalises_large_errors():
    assert metrics.rmse([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert metrics.rmse([0.0, 0.0], [0.0, 2.0]) == pytest.approx(math.sqrt(2.0))
    # RMSE >= MAE always.
    assert metrics.rmse([0.0, 0.0], [1.0, 2.0]) >= metrics.mae([0.0, 0.0], [1.0, 2.0])


def test_smape_is_percent_and_symmetric():
    assert metrics.smape([0.0], [0.0]) == 0.0
    # 2*|1-2|/(1+2) = 2/3 -> 66.67%.
    assert metrics.smape([1.0], [2.0]) == pytest.approx(200.0 / 3.0)
    assert metrics.smape([1.0], [2.0]) == pytest.approx(metrics.smape([2.0], [1.0]))
    assert 0.0 <= metrics.smape([5.0, 10.0], [7.0, 3.0]) <= 200.0


def test_smape_zero_actual_nonzero_forecast():
    # Denominator is non-zero, so this is a normal finite value: 200%.
    assert metrics.smape([0.0], [1.0]) == pytest.approx(200.0)


def test_empty_windows_return_nan_for_all_metrics():
    empty = np.array([], dtype=np.float64)
    assert math.isnan(metrics.mae(empty, empty))
    assert math.isnan(metrics.rmse(empty, empty))
    assert math.isnan(metrics.smape(empty, empty))
    assert math.isnan(metrics.mase(empty, empty))


def test_non_finite_values_are_treated_as_missing():
    y_true = [1.0, np.nan, 3.0]
    y_pred = [1.0, 2.0, np.inf]
    # Only the first pair is finite, so MAE is 0 over one usable point.
    assert metrics.mae(y_true, y_pred) == 0.0
    assert math.isnan(metrics.mae([np.nan], [1.0]))


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.mae([1.0, 2.0], [1.0])


def test_mase_perfect_forecast_is_zero():
    train = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    assert metrics.mase(train, train, y_train=train, season_length=1) == 0.0


def test_mase_uses_seasonal_naive_scale():
    # Alternating series: the step-to-step seasonal difference is exactly 10.
    history = np.tile([0.0, 10.0], 6)
    forecast = np.full(4, 5.0)
    actual = np.full(4, 6.0)
    scale = metrics.seasonal_naive_scale(history, season_length=1)
    assert scale == pytest.approx(10.0)
    expected = metrics.mae(actual, forecast) / scale
    assert metrics.mase(actual, forecast, y_train=history, season_length=1) == pytest.approx(
        expected
    )


def test_mase_constant_series_edge_cases():
    constant = np.full(12, 7.0)
    # Perfect forecast on a scale-zero series: defined as 0.0, not nan.
    assert metrics.mase(constant[:3], constant[:3], y_train=constant, season_length=1) == 0.0
    # Wrong forecast on a scale-zero series is infinite, not a crash.
    wrong = np.full(3, 1.0)
    assert metrics.mase(wrong, constant[:3], y_train=constant, season_length=1) == float("inf")


def test_mase_scale_undefined_returns_nan():
    # One observation cannot form a difference -> nan.
    assert math.isnan(metrics.seasonal_naive_scale([5.0], season_length=1))
    assert math.isnan(metrics.mase([5.0], [4.0], season_length=1))


def test_metric_set_returns_all_four_names():
    result = metrics.metric_set([1.0, 2.0, 3.0], [1.0, 2.0, 2.0], season_length=1)
    assert set(result) == set(metrics.METRIC_NAMES)
    assert result["mae"] == pytest.approx(1.0 / 3.0)


def test_metrics_accept_lists_and_arrays_identically():
    as_list = metrics.metric_set([1.0, 2.0], [2.0, 2.0])
    as_array = metrics.metric_set(np.array([1.0, 2.0]), np.array([2.0, 2.0]))
    assert as_list == as_array
