"""Integration coverage for the local CPU backtest family set.

These tests pin the executable contract `make backtest` relies on: the five
local families are declared in the experiment config and resolvable through the
license-gated registry, and every config in the primary 24/48/96/192 horizon
sweep shares the same frozen split geometry. They do not need the real dataset;
the real-data proof lives in the committed results summary.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tsbench.evaluation import runner
from tsbench.registry import load_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"
BACKTEST_CONFIG = REPO_ROOT / "configs" / "experiments" / "backtest.yaml"
HORIZON_CONFIGS = {
    24: BACKTEST_CONFIG,
    48: REPO_ROOT / "configs" / "experiments" / "backtest_h48.yaml",
    96: REPO_ROOT / "configs" / "experiments" / "backtest_h96.yaml",
    192: REPO_ROOT / "configs" / "experiments" / "backtest_h192.yaml",
}
LOCAL_FAMILIES = [
    "naive",
    "seasonal_naive",
    "auto_ets",
    "auto_arima",
    "xgboost_lags",
]


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def test_backtest_config_declares_all_five_local_families():
    experiment = runner.load_experiment(BACKTEST_CONFIG)
    assert experiment.models == LOCAL_FAMILIES
    registry = load_registry(MODELS_CONFIG)
    for name in LOCAL_FAMILIES:
        assert name in registry, f"{name!r} missing from configs/models.yaml"


def test_experiment_config_exposes_track_memory():
    experiment = runner.load_experiment(BACKTEST_CONFIG)
    # The bounded first-results run opts out of tracemalloc peak tracking.
    assert experiment.track_memory is False
    assert runner.ExperimentConfig.from_mapping({}).track_memory is True
    with pytest.raises(runner.RunnerError, match="track_memory"):
        runner.ExperimentConfig.from_mapping({"track_memory": "yes"})


def test_primary_horizon_sweep_shares_one_frozen_geometry():
    base = runner.load_experiment(HORIZON_CONFIGS[24])
    for horizon, path in HORIZON_CONFIGS.items():
        experiment = runner.load_experiment(path)
        assert experiment.models == LOCAL_FAMILIES
        assert experiment.split.horizon == horizon
        # The split boundaries are frozen; only the forecast horizon varies.
        assert experiment.split.as_dict() == {**base.split.as_dict(), "horizon": horizon}
        assert experiment.season_length == base.season_length
        assert experiment.split_manifest == base.split_manifest


@pytest.mark.skipif(
    not (_module_available("statsforecast") and _module_available("xgboost")),
    reason="local model tier runtime deps are not installed",
)
def test_every_local_family_constructs_through_the_registry(monkeypatch):
    experiment = runner.load_experiment(BACKTEST_CONFIG)
    registry = load_registry(MODELS_CONFIG)
    monkeypatch.setattr(runner, "_try_registry", lambda: registry)
    for name in LOCAL_FAMILIES:
        model = runner._build_model(
            name,
            None,
            model_cfg=runner._model_config(experiment, name),
            seed=experiment.seed,
        )
        assert model.info().name == name
