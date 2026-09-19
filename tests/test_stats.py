import importlib.util

import numpy as np
import pytest

from tsbench.models import stats
from tsbench.models.stats import AutoARIMA, AutoETS


def _have(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


HAS_STATSMODELS = _have("statsmodels")
HAS_STATSFORECAST = _have("statsforecast")


def _seasonal_series(n=60, season_length=12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    return (
        20.0
        + 0.2 * t
        + 4.0 * np.sin(2.0 * np.pi * t / season_length)
        + rng.normal(0.0, 0.2, n)
    )


def _engines():
    engines = []
    if HAS_STATSFORECAST:
        engines.append("statsforecast")
    if HAS_STATSMODELS:
        engines.append("statsmodels")
    return engines


@pytest.mark.parametrize("cls", [AutoETS, AutoARIMA])
@pytest.mark.parametrize("engine", _engines())
def test_predict_shape_dtype_and_finiteness(cls, engine):
    model = cls(season_length=12, engine=engine).fit(_seasonal_series())
    forecast = model.predict(8)
    assert isinstance(forecast, np.ndarray)
    assert forecast.shape == (8,)
    assert forecast.dtype == np.float64
    assert np.all(np.isfinite(forecast))


@pytest.mark.parametrize("cls", [AutoETS, AutoARIMA])
@pytest.mark.parametrize("engine", _engines())
def test_predictions_do_not_change_when_input_is_mutated_after_fit(cls, engine):
    y = _seasonal_series()
    model = cls(season_length=12, engine=engine).fit(y)
    expected = model.predict(6)
    y[:] = -1e6
    np.testing.assert_array_equal(model.predict(6), expected)


@pytest.mark.parametrize("cls", [AutoETS, AutoARIMA])
@pytest.mark.parametrize("engine", _engines())
def test_constant_series_predictions_are_constant(cls, engine):
    model = cls(season_length=1, engine=engine).fit(np.full(30, 5.0))
    forecast = model.predict(4)
    assert np.all(np.isfinite(forecast))
    np.testing.assert_allclose(forecast, np.full(4, 5.0), atol=1e-4)


@pytest.mark.parametrize("cls", [AutoETS, AutoARIMA])
@pytest.mark.parametrize("engine", _engines())
def test_all_zero_series_runs(cls, engine):
    forecast = cls(season_length=1, engine=engine).fit(np.zeros(30)).predict(4)
    assert forecast.shape == (4,)
    assert np.all(np.isfinite(forecast))
    np.testing.assert_allclose(forecast, np.zeros(4), atol=1e-4)


@pytest.mark.parametrize("cls", [AutoETS, AutoARIMA])
@pytest.mark.parametrize("engine", _engines())
def test_short_series_runs(cls, engine):
    forecast = cls(season_length=1, engine=engine).fit([1.0, 2.0, 3.0, 4.0]).predict(3)
    assert forecast.shape == (3,)
    assert np.all(np.isfinite(forecast))


def test_auto_ets_statsforecast_falls_back_for_tiny_series(monkeypatch):
    if not (HAS_STATSFORECAST and HAS_STATSMODELS):
        pytest.skip("needs statsforecast and statsmodels")
    model = AutoETS(season_length=1, engine="statsforecast").fit(np.arange(4.0))
    info = model.info()
    assert info.extra["engine"] == "statsmodels"
    assert "statsforecast_error" in info.extra
    assert np.all(np.isfinite(model.predict(3)))


def test_auto_engine_prefers_statsforecast_when_available():
    if not HAS_STATSFORECAST:
        pytest.skip("statsforecast is not installed")
    model = AutoETS(season_length=12, engine="auto").fit(_seasonal_series())
    assert model.info().extra["engine"] == "statsforecast"


def test_auto_engine_falls_back_to_statsmodels(monkeypatch):
    if not HAS_STATSMODELS:
        pytest.skip("statsmodels is not installed")
    monkeypatch.setattr(
        stats,
        "_module_available",
        lambda name: name != "statsforecast" and _have(name),
    )
    model = AutoARIMA(season_length=12, engine="auto").fit(_seasonal_series())
    assert model.info().extra["engine"] == "statsmodels"
    assert model.info().extra["seasonal_order"] is None
    assert np.all(np.isfinite(model.predict(4)))


def test_engine_is_recorded_in_info():
    engines = _engines()
    if not engines:
        pytest.skip("no classical engine installed")
    model = AutoETS(season_length=12, engine=engines[0]).fit(_seasonal_series())
    info = model.info()
    assert info.extra["engine"] in engines
    assert info.extra["method"] == "AutoETS"
    assert info.extra["season_length"] == 12
    assert info.name == "auto_ets"
    assert info.family == "classical"
    assert info.zero_shot is False
    assert isinstance(info.license, str) and info.license
    assert isinstance(info.revision, str) and info.revision


def test_statsmodels_ets_records_selected_configuration():
    if not HAS_STATSMODELS:
        pytest.skip("statsmodels is not installed")
    model = AutoETS(season_length=12, engine="statsmodels").fit(_seasonal_series())
    info = model.info()
    assert info.extra["seasonal"] == "add"
    assert info.extra["season_length"] == 12
    assert np.isfinite(info.extra["aic"])


def test_statsmodels_arima_records_selected_order():
    if not HAS_STATSMODELS:
        pytest.skip("statsmodels is not installed")
    model = AutoARIMA(season_length=12, engine="statsmodels").fit(_seasonal_series())
    order = model.info().extra["order"]
    assert len(order) == 3
    assert all(isinstance(value, int) for value in order)
    assert model.info().extra["seasonal_order"] is None


@pytest.mark.parametrize("cls", [AutoETS, AutoARIMA])
def test_predict_before_fit_raises(cls):
    with pytest.raises(RuntimeError):
        cls(season_length=12, engine="auto").predict(3)


@pytest.mark.parametrize("cls", [AutoETS, AutoARIMA])
@pytest.mark.parametrize("horizon", [0, -1, 1.5, None, True])
def test_invalid_horizons_raise(cls, horizon):
    engines = _engines()
    if not engines:
        pytest.skip("no classical engine installed")
    model = cls(season_length=12, engine=engines[0]).fit(_seasonal_series())
    with pytest.raises(ValueError):
        model.predict(horizon)


@pytest.mark.parametrize("cls", [AutoETS, AutoARIMA])
def test_invalid_engine_and_season_length_raise(cls):
    with pytest.raises(ValueError):
        cls(season_length=12, engine="not-an-engine")
    with pytest.raises(ValueError):
        cls(season_length=0, engine="auto")
    with pytest.raises(TypeError):
        cls(season_length=2.5, engine="auto")


def test_requesting_missing_engine_raises_import_error(monkeypatch):
    monkeypatch.setattr(stats, "_module_available", lambda name: False)
    with pytest.raises(ImportError):
        AutoETS(season_length=1, engine="statsforecast").fit([1.0, 2.0, 3.0])
    with pytest.raises(ImportError):
        AutoARIMA(season_length=1, engine="statsmodels").fit([1.0, 2.0, 3.0])


def test_fit_rejects_empty_and_non_finite_series():
    engines = _engines()
    if not engines:
        pytest.skip("no classical engine installed")
    with pytest.raises(ValueError):
        AutoETS(season_length=1, engine=engines[0]).fit([])
    with pytest.raises(ValueError):
        AutoETS(season_length=1, engine=engines[0]).fit([1.0, np.nan])
