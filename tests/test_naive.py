import numpy as np
import pytest

from tsbench.models import Forecaster
from tsbench.models.naive import Naive, SeasonalNaive


def test_contract_resolves_to_foundation_module_when_present():
    foundation = pytest.importorskip("tsbench.base")
    import tsbench.models as models

    assert models.Forecaster is foundation.Forecaster
    assert models.ModelInfo is foundation.ModelInfo


def test_naive_satisfies_forecaster_contract():
    assert isinstance(Naive().fit([1.0, 2.0]), Forecaster)
    assert isinstance(SeasonalNaive(season_length=2).fit([1.0, 2.0, 3.0]), Forecaster)


def test_naive_repeats_last_value_with_expected_shape_and_dtype():
    model = Naive().fit([1.0, 2.0, 3.0])
    forecast = model.predict(4)
    assert isinstance(forecast, np.ndarray)
    assert forecast.shape == (4,)
    assert forecast.dtype == np.float64
    np.testing.assert_array_equal(forecast, [3.0, 3.0, 3.0, 3.0])


def test_naive_fit_returns_self():
    model = Naive()
    assert model.fit([1.0, 2.0]) is model


def test_naive_single_observation():
    np.testing.assert_array_equal(Naive().fit([5.0]).predict(3), [5.0, 5.0, 5.0])


def test_naive_constant_series():
    forecast = Naive().fit(np.full(10, 7.5)).predict(5)
    np.testing.assert_array_equal(forecast, np.full(5, 7.5))


def test_naive_all_zero_series():
    forecast = Naive().fit(np.zeros(10)).predict(5)
    np.testing.assert_array_equal(forecast, np.zeros(5))


def test_naive_one_step_horizon():
    assert Naive().fit([1.0, 2.0, 9.0]).predict(1).tolist() == [9.0]


def test_naive_accepts_lists_and_column_vectors():
    from_list = Naive().fit([1.0, 2.0, 3.0]).predict(2)
    from_column = Naive().fit(np.array([[1.0], [2.0], [3.0]])).predict(2)
    np.testing.assert_array_equal(from_list, from_column)


def test_naive_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        Naive().predict(3)


def test_naive_rejects_empty_and_non_finite_input():
    with pytest.raises(ValueError):
        Naive().fit([])
    with pytest.raises(ValueError):
        Naive().fit([1.0, np.nan])
    with pytest.raises(ValueError):
        Naive().fit([1.0, np.inf])


@pytest.mark.parametrize("horizon", [0, -1, 1.5, "3", None, True])
def test_naive_rejects_invalid_horizons(horizon):
    model = Naive().fit([1.0, 2.0])
    with pytest.raises(ValueError):
        model.predict(horizon)


def test_naive_does_not_read_the_series_after_fit():
    y = np.array([1.0, 2.0, 3.0])
    model = Naive().fit(y)
    expected = model.predict(4)
    y[:] = 999.0
    np.testing.assert_array_equal(model.predict(4), expected)


def test_naive_info_contract():
    model = Naive().fit([1.0, 2.0, 3.0])
    info = model.info()
    assert info.name == "naive"
    assert info.family == "baseline"
    assert info.zero_shot is True
    assert info.params == 0
    assert info.extra["method"] == "last_value"
    assert info.extra["last_value"] == 3.0
    assert isinstance(info.license, str) and info.license
    assert isinstance(info.revision, str) and info.revision


def test_seasonal_naive_repeats_last_full_season():
    y = np.arange(1.0, 13.0)
    forecast = SeasonalNaive(season_length=4).fit(y).predict(6)
    np.testing.assert_array_equal(forecast, [9.0, 10.0, 11.0, 12.0, 9.0, 10.0])


def test_seasonal_naive_default_matches_naive():
    y = [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(
        SeasonalNaive().fit(y).predict(3), Naive().fit(y).predict(3)
    )


def test_seasonal_naive_short_series_uses_available_history():
    forecast = SeasonalNaive(season_length=12).fit([1.0, 2.0, 3.0]).predict(5)
    np.testing.assert_array_equal(forecast, [1.0, 2.0, 3.0, 1.0, 2.0])
    assert SeasonalNaive(season_length=12).fit([1.0, 2.0, 3.0]).info().extra[
        "effective_season_length"
    ] == 3


def test_seasonal_naive_constant_and_zero_series():
    constant = SeasonalNaive(season_length=3).fit(np.full(8, 2.5)).predict(7)
    np.testing.assert_array_equal(constant, np.full(7, 2.5))
    zeros = SeasonalNaive(season_length=3).fit(np.zeros(8)).predict(7)
    np.testing.assert_array_equal(zeros, np.zeros(7))


def test_seasonal_naive_does_not_read_the_series_after_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = SeasonalNaive(season_length=2).fit(y)
    expected = model.predict(5)
    y[:] = -1.0
    np.testing.assert_array_equal(model.predict(5), expected)


def test_seasonal_naive_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        SeasonalNaive(season_length=4).predict(3)


@pytest.mark.parametrize("season_length", [0, -1, 1.5, "4"])
def test_seasonal_naive_rejects_invalid_season_length(season_length):
    with pytest.raises((TypeError, ValueError)):
        SeasonalNaive(season_length=season_length)


def test_seasonal_naive_info_contract():
    info = SeasonalNaive(season_length=7).fit(np.arange(14.0)).info()
    assert info.name == "seasonal_naive"
    assert info.family == "baseline"
    assert info.zero_shot is True
    assert info.params == 0
    assert info.extra["season_length"] == 7
    assert info.extra["effective_season_length"] == 7
