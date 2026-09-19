import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from tsbench.models.gbdt import XGBoostLags


def _seasonal_series(n=84, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    return 10.0 + 0.1 * t + 3.0 * np.sin(2.0 * np.pi * t / 7.0) + rng.normal(0.0, 0.1, n)


def _model(**kwargs):
    params = dict(lags=(1, 2, 3, 7, 14), n_estimators=50, random_state=0)
    params.update(kwargs)
    return XGBoostLags(**params)


def test_fit_predict_returns_finite_float_array():
    forecast = _model().fit(_seasonal_series()).predict(12)
    assert isinstance(forecast, np.ndarray)
    assert forecast.shape == (12,)
    assert forecast.dtype == np.float64
    assert np.all(np.isfinite(forecast))


def test_fit_returns_self():
    model = _model()
    assert model.fit(_seasonal_series()) is model


def test_predictions_are_deterministic_with_fixed_seed():
    y = _seasonal_series()
    first = _model().fit(y).predict(10)
    second = _model().fit(y).predict(10)
    np.testing.assert_array_equal(first, second)


def test_lags_only_and_calendar_ablation_change_feature_sets():
    y = pd.Series(
        _seasonal_series(),
        index=pd.date_range("2020-01-01", periods=len(_seasonal_series()), freq="D"),
    )
    lags_only = _model(use_calendar=False).fit(y)
    with_calendar = _model(use_calendar=True).fit(y)
    _, lags_only_names = lags_only._design(
        lags_only._history, np.arange(lags_only.max_lag, lags_only._history.size)
    )
    _, calendar_names = with_calendar._design(
        with_calendar._history,
        np.arange(with_calendar.max_lag, with_calendar._history.size),
    )
    assert lags_only_names == ["lag_1", "lag_2", "lag_3", "lag_7", "lag_14"]
    assert calendar_names[:5] == lags_only_names
    assert "doy_sin" in calendar_names and "dow" in calendar_names
    assert np.all(np.isfinite(lags_only.predict(5)))
    assert np.all(np.isfinite(with_calendar.predict(5)))


def test_design_matrix_rows_use_only_strictly_past_values():
    y = np.arange(1.0, 31.0)
    model = _model(lags=(1, 3, 5), use_calendar=False).fit(y)
    positions = np.arange(model.max_lag, y.size)
    features, names = model._design(model._history, positions)
    assert names == ["lag_1", "lag_3", "lag_5"]
    assert features.shape == (y.size - model.max_lag, 3)
    for row, position in enumerate(positions):
        np.testing.assert_array_equal(
            features[row], [y[position - 1], y[position - 3], y[position - 5]]
        )
    assert np.all(np.max(np.abs(features - y[positions, None]), axis=1) > 0)


def test_predict_never_reads_actual_values_after_fit():
    y = _seasonal_series()
    model = _model().fit(y)
    expected = model.predict(8)
    y[:] = -1e6
    np.testing.assert_array_equal(model.predict(8), expected)
    assert not np.any(model._history == -1e6)


def test_short_series_falls_back_to_last_value():
    model = _model(lags=(1, 2, 5)).fit([1.0, 2.0, 3.0])
    forecast = model.predict(4)
    np.testing.assert_array_equal(forecast, [3.0, 3.0, 3.0, 3.0])
    info = model.info()
    assert info.extra["fallback"] == "insufficient_history_for_lags"


def test_constant_series_stays_constant():
    forecast = _model().fit(np.full(40, 5.0)).predict(6)
    assert np.all(np.isfinite(forecast))
    np.testing.assert_allclose(forecast, np.full(6, 5.0), atol=1e-6)


def test_all_zero_series_stays_zero():
    forecast = _model().fit(np.zeros(40)).predict(6)
    assert np.all(np.isfinite(forecast))
    np.testing.assert_allclose(forecast, np.zeros(6), atol=1e-6)


def test_datetime_index_produces_calendar_features_and_future_predictions():
    y = pd.Series(
        _seasonal_series(),
        index=pd.date_range("2021-01-01", periods=len(_seasonal_series()), freq="D"),
    )
    model = _model(use_calendar=True).fit(y)
    assert model.info().extra["calendar_mode"] == "datetime"
    forecast = model.predict(10)
    assert forecast.shape == (10,)
    assert np.all(np.isfinite(forecast))


def test_hourly_series_feature_count_is_stable_at_midnight_origin():
    index = pd.date_range("2020-01-01 00:00", periods=24, freq="h")
    y = pd.Series(np.arange(24, dtype=np.float64), index=index)
    model = _model(lags=(1, 2, 3), use_calendar=True).fit(y)
    assert model.info().extra["calendar_mode"] == "datetime"
    forecast = model.predict(3)
    assert forecast.shape == (3,)
    assert np.all(np.isfinite(forecast))


def test_two_point_datetime_series_falls_back_to_last_value():
    y = pd.Series([1.0, 2.0], index=pd.date_range("2020-01-01", periods=2, freq="D"))
    model = _model(lags=(1, 2, 5), use_calendar=True).fit(y)
    assert model.info().extra["fallback"] == "insufficient_history_for_lags"
    np.testing.assert_array_equal(model.predict(3), [2.0, 2.0, 2.0])


def test_positional_features_when_index_is_not_datetime():
    y = pd.Series(_seasonal_series(), index=np.arange(len(_seasonal_series())) + 100)
    model = _model(use_calendar=True).fit(y)
    assert model.info().extra["calendar_mode"] == "position"
    assert np.all(np.isfinite(model.predict(4)))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        _model().predict(3)


@pytest.mark.parametrize("horizon", [0, -2, 2.5, None, True])
def test_invalid_horizons_raise(horizon):
    model = _model().fit(_seasonal_series())
    with pytest.raises(ValueError):
        model.predict(horizon)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        XGBoostLags(lags=())
    with pytest.raises(ValueError):
        XGBoostLags(lags=(0, 1))
    with pytest.raises(ValueError):
        XGBoostLags(n_estimators=0)
    with pytest.raises(ValueError):
        XGBoostLags(max_depth=0)
    with pytest.raises(ValueError):
        XGBoostLags(learning_rate=0.0)
    with pytest.raises(ValueError):
        XGBoostLags(subsample=0.0)
    with pytest.raises(ValueError):
        XGBoostLags(n_jobs=0)
    with pytest.raises(TypeError):
        XGBoostLags(lags="1,2")


def test_integer_lags_are_expanded():
    model = XGBoostLags(lags=3)
    assert model.lags == (1, 2, 3)
    assert model.max_lag == 3


def test_info_contract():
    model = _model(use_calendar=False).fit(_seasonal_series())
    info = model.info()
    assert info.name == "xgboost_lags"
    assert info.family == "ml"
    assert info.zero_shot is False
    assert info.extra["use_calendar"] is False
    assert info.extra["lags"] == [1, 2, 3, 7, 14]
    assert isinstance(info.license, str) and info.license
    assert isinstance(info.revision, str) and info.revision
