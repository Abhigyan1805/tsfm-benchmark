"""Evaluation: metrics, rolling-origin backtest, runner, results, statistics."""

from __future__ import annotations

from .backtest import BacktestError, WindowForecast, backtest_model, backtest_windows
from .contract import CONTRACT_SOURCE, RESULT_COLUMNS, Forecaster, ModelInfo
from .metrics import (
    METRIC_NAMES,
    mae,
    mase,
    metric_set,
    rmse,
    seasonal_naive_scale,
    smape,
)
from .results import (
    RunContext,
    SchemaError,
    config_hash,
    current_git_sha,
    make_run_id,
    read_results,
    result_row,
    results_path,
    utc_now,
    validate_row,
    write_results,
)
from .runner import ExperimentConfig, RunnerError, load_experiment, run_experiment
from .stats import (
    FriedmanResult,
    PairedResult,
    StatsError,
    aggregate_by_series,
    critical_difference,
    friedman_test,
    nemenyi_critical_difference,
    pairwise_matrix,
    rank_matrix,
    wilcoxon_signed_rank,
)

__all__ = [
    "BacktestError",
    "CONTRACT_SOURCE",
    "ExperimentConfig",
    "FriedmanResult",
    "Forecaster",
    "METRIC_NAMES",
    "ModelInfo",
    "PairedResult",
    "RESULT_COLUMNS",
    "RunnerError",
    "RunContext",
    "SchemaError",
    "StatsError",
    "WindowForecast",
    "aggregate_by_series",
    "backtest_model",
    "backtest_windows",
    "config_hash",
    "critical_difference",
    "current_git_sha",
    "friedman_test",
    "load_experiment",
    "mae",
    "make_run_id",
    "mase",
    "metric_set",
    "nemenyi_critical_difference",
    "pairwise_matrix",
    "rank_matrix",
    "read_results",
    "result_row",
    "results_path",
    "rmse",
    "run_experiment",
    "seasonal_naive_scale",
    "smape",
    "utc_now",
    "validate_row",
    "wilcoxon_signed_rank",
    "write_results",
]
