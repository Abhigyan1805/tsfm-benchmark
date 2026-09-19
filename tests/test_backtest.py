"""Rolling-origin backtest and results-schema behaviour."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tsbench.data.splits import SplitConfig, compute_split_plan, enumerate_windows
from tsbench.evaluation import runner, stats
from tsbench.evaluation.backtest import backtest_model, backtest_windows
from tsbench.evaluation.contract import RESULT_COLUMNS
from tsbench.evaluation.results import (
    RunContext,
    SchemaError,
    result_row,
    validate_row,
    write_results,
)

try:
    from ._stub_models import ConstantForecaster, NaiveStub, SeasonalNaiveStub
except ImportError:  # ``tests`` is not a package until the foundation slice lands
    from _stub_models import ConstantForecaster, NaiveStub, SeasonalNaiveStub  # type: ignore


def _series(n: int = 600, season: int = 24) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(n, dtype=np.float64)
    return 100.0 + 10.0 * np.sin(2 * np.pi * t / season) + rng.normal(0, 0.5, n)


def _plan(n: int = 600, **overrides):
    options = {"context_length": 48, "horizon": 12, "stride": 12, "min_train_size": 20}
    options.update(overrides)
    config = SplitConfig(**options)
    return compute_split_plan(n, config)


def test_windows_are_disjoint_in_time_and_ordered():
    plan = _plan()
    origins = enumerate_windows(plan, period="test")
    assert origins, "expected at least one test window"
    assert origins == sorted(origins)
    for origin in origins:
        # A full lookback is behind the origin and a full horizon in front.
        assert origin - plan.config.context_length >= plan.val_end
        assert origin + plan.config.horizon <= plan.test_end


def test_backtest_produces_one_row_per_window():
    plan = _plan()
    values = _series()
    outcomes = backtest_model(values, plan, NaiveStub(), series_id="s1")
    assert len(outcomes) == len(enumerate_windows(plan, period="test"))
    for outcome in outcomes:
        assert outcome.ok
        assert outcome.metrics["mae"] >= 0.0
        assert outcome.forecast.shape == (plan.config.horizon,)
        assert outcome.target.shape == (plan.config.horizon,)
        assert outcome.train_seconds is not None


def test_context_length_is_honoured_and_context_never_sees_target():
    values = _series()
    plan = _plan(context_length=72, horizon=8, stride=8)
    seen: list[tuple[int, float]] = []

    class Spy(NaiveStub):
        def fit(self, y):
            seen.append((int(y.size), float(y[-1])))
            return super().fit(y)

    outcomes = backtest_windows(
        values, plan, Spy, series_id="s1", model_name="spy", family="baseline"
    )
    assert seen
    for (size, last), outcome in zip(seen, outcomes, strict=True):
        assert size == 72
        # The last context value is exactly values[origin-1].
        assert last == values[outcome.origin - 1]


def test_models_receive_different_metrics_on_seasonal_series():
    values = _series()
    plan = _plan()
    naive = backtest_model(values, plan, NaiveStub(), series_id="s")
    seasonal = backtest_model(
        values, plan, SeasonalNaiveStub(season_length=24), series_id="s"
    )
    assert len(naive) == len(seasonal)
    mae_naive = np.mean([o.metrics["mae"] for o in naive])
    mae_seasonal = np.mean([o.metrics["mae"] for o in seasonal])
    assert mae_seasonal < mae_naive


def test_backtest_records_failure_without_aborting_other_windows():
    values = _series()
    plan = _plan()
    state = {"first": True}

    class Flaky(NaiveStub):
        def fit(self, y):
            if state["first"]:
                state["first"] = False
                raise RuntimeError("boom")
            return super().fit(y)

    outcomes = backtest_windows(
        values, plan, Flaky, series_id="s", model_name="flaky", family="baseline"
    )
    assert len(outcomes) == len(enumerate_windows(plan, period="test"))
    assert outcomes[0].error is not None and "boom" in outcomes[0].error
    assert all(outcome.ok for outcome in outcomes[1:])


def test_result_row_has_exactly_the_frozen_schema():
    row = result_row(
        run_id="r",
        dataset="d",
        series_id="s",
        model="m",
        family="baseline",
        context_length=48,
        horizon=12,
        window_index=0,
        mae=1.0,
        rmse=2.0,
        mase=0.5,
        smape=10.0,
        git_sha="deadbeef",
        config_hash="c0ffee",
    )
    assert tuple(row) == RESULT_COLUMNS
    assert "train_seconds" in row
    validate_row(row)


@pytest.mark.parametrize(
    "model_name,family,zero_shot",
    [
        ("naive", "baseline", True),
        ("seasonal_naive", "baseline", True),
        ("auto_ets", "classical", False),
        ("xgboost_lags", "ml", False),
    ],
)
def test_zero_shot_column_preserves_the_models_own_flag(
    model_name, family, zero_shot
):
    stub = type(
        model_name,
        (NaiveStub,),
        {"name": model_name, "family": family, "zero_shot": zero_shot},
    )
    outcomes = backtest_windows(
        _series(), _plan(), stub, series_id="s", model_name=model_name, family=family
    )
    assert outcomes
    assert all(runner._row_zero_shot(outcome) is zero_shot for outcome in outcomes)


def test_untrained_family_has_empty_train_seconds():
    row = result_row(
        run_id="r",
        dataset="d",
        series_id="s",
        model="naive",
        family="baseline",
        context_length=48,
        horizon=12,
        window_index=0,
        mae=1.0,
        rmse=2.0,
        mase=0.5,
        smape=10.0,
        git_sha="sha",
        config_hash="cfg",
    )
    assert row["train_seconds"] == ""


@pytest.mark.parametrize("family", ["classical", "ml", "deep"])
def test_trained_family_records_train_seconds(family):
    row = result_row(
        run_id="r",
        dataset="d",
        series_id="s",
        model="auto_ets",
        family=family,
        context_length=48,
        horizon=12,
        window_index=0,
        mae=1.0,
        rmse=2.0,
        mase=0.5,
        smape=10.0,
        git_sha="sha",
        config_hash="cfg",
        train_seconds=1.25,
    )
    assert row["train_seconds"] == 1.25


@pytest.mark.parametrize("family", ["classical", "ml"])
def test_backtest_records_fit_wall_clock_for_trained_families(family):
    plan = _plan()
    values = _series()
    stub = type("TimedStub", (NaiveStub,), {"name": "timed", "family": family})

    outcomes = backtest_windows(
        values, plan, stub, series_id="s", model_name="timed", family=family
    )
    assert outcomes
    assert all(outcome.train_seconds is not None for outcome in outcomes)
    first = outcomes[0]
    row = result_row(
        run_id="r",
        dataset="d",
        series_id="s",
        model="timed",
        family=family,
        context_length=plan.config.context_length,
        horizon=plan.config.horizon,
        window_index=0,
        mae=first.metrics["mae"],
        rmse=first.metrics["rmse"],
        mase=first.metrics["mase"],
        smape=first.metrics["smape"],
        git_sha="sha",
        config_hash="cfg",
        train_seconds=first.train_seconds,
    )
    assert row["train_seconds"] != ""


def test_schema_drift_is_rejected():
    row = result_row(
        run_id="r",
        dataset="d",
        series_id="s",
        model="m",
        family="baseline",
        context_length=1,
        horizon=1,
        window_index=0,
        mae=0.0,
        rmse=0.0,
        mase=0.0,
        smape=0.0,
        git_sha="x",
        config_hash="y",
    )
    row.pop("rmse")
    with pytest.raises(SchemaError):
        validate_row(row)


def test_write_results_layout_and_metadata(tmp_path: Path):
    context = RunContext.create("unit", {"name": "unit", "horizon": 12})
    rows = [
        result_row(
            run_id=context.run_id,
            dataset="d",
            series_id=f"s{i}",
            model="naive",
            family="baseline",
            context_length=48,
            horizon=12,
            window_index=i,
            mae=float(i),
            rmse=float(i),
            mase=float(i),
            smape=float(i),
            git_sha=context.git_sha,
            config_hash=context.config_sha,
        )
        for i in range(3)
    ]
    csv_path = write_results(rows, root=tmp_path, context=context)
    assert csv_path == tmp_path / context.run_id / "results.csv"
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == RESULT_COLUMNS
        assert len(list(reader)) == 3
    metadata = json.loads((csv_path.parent / "run.json").read_text())
    assert metadata["git_sha"]
    assert metadata["config_hash"] == context.config_sha
    assert metadata["n_rows"] == 3


def test_runner_end_to_end_on_local_csv_fixture(tmp_path: Path):
    """The smoke path: config -> loader -> frozen split -> rows on disk."""
    fixture = Path(__file__).parent / "fixtures" / "smoke_series.csv"
    if not fixture.is_file():
        pytest.skip("foundation-owned smoke fixture not present on this branch")

    values = np.loadtxt(fixture, delimiter=",", skiprows=1, usecols=1)
    assert values.size > 0
    repo_root = Path(__file__).resolve().parents[1]
    config = {
        "name": "unit-smoke",
        "dataset": {"name": "smoke", "loader": "local_csv"},
        "models": ["naive"],
        "model_configs": {
            "naive": {"entrypoint": "tests._stub_models:NaiveStub"},
        },
        "split": {
            "train_frac": 0.6,
            "val_frac": 0.2,
            "test_frac": 0.2,
            "context_length": 20,
            "horizon": 4,
            "stride": 4,
            "min_train_size": 5,
        },
        "datasets_config": str(repo_root / "configs" / "datasets.yaml"),
        "output_dir": str(tmp_path / "results"),
    }
    import yaml

    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    # Build a frozen manifest for the fixture, as the real pipeline does.
    from tsbench.data import SplitConfig, build_manifest, load_local_csv, save_manifest

    series = load_local_csv(fixture, series_id="smoke_series")
    manifest = build_manifest(
        "smoke", [series], SplitConfig.from_mapping(config["split"]), loader="local_csv"
    )
    manifest_path = tmp_path / "split_manifest.json"
    save_manifest(manifest, manifest_path)
    config["dataset"]["split_manifest"] = str(manifest_path)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = runner.run_experiment(config_path, root=repo_root)
    assert summary["n_rows"] > 0
    csv_path = Path(summary["results"])
    assert csv_path.is_file()
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == RESULT_COLUMNS
        rows = list(reader)
    assert rows and all(row["model"] == "naive" for row in rows)
    assert all(row["train_seconds"] == "" for row in rows)


def test_runner_refuses_a_preliminary_manifest(tmp_path: Path):
    from tsbench.data import SplitConfig, build_manifest, save_manifest

    values = np.arange(200, dtype=np.float64)
    import yaml

    class _S:
        series_id = "s"

        def __init__(self, values):
            self.values = values

    # Self-contained catalogue + CSV so this test does not depend on the
    # foundation-owned smoke fixture.
    import pandas as pd

    stamps = pd.date_range("2024-01-01", periods=values.size, freq="D")
    csv_path = tmp_path / "prelim_series.csv"
    pd.DataFrame({"timestamp": stamps, "value": values}).to_csv(csv_path, index=False)
    catalogue = {
        "version": 1,
        "default_dataset": "prelim",
        "datasets": {
            "prelim": {
                "loader": "local_csv",
                "path": str(csv_path),
                "license": "Apache-2.0",
                "series_id": "s",
            }
        },
    }
    catalogue_path = tmp_path / "datasets.yaml"
    catalogue_path.write_text(yaml.safe_dump(catalogue), encoding="utf-8")

    manifest = build_manifest(
        "prelim", [_S(values)], SplitConfig(context_length=10, horizon=4), preliminary=True
    )
    manifest_path = tmp_path / "split_manifest.json"
    save_manifest(manifest, manifest_path)
    repo_root = Path(__file__).resolve().parents[1]
    config = {
        "name": "prelim",
        "dataset": {"name": "prelim", "loader": "local_csv"},
        "split_manifest": str(manifest_path),
        "models": ["naive"],
        "split": {"context_length": 10, "horizon": 4},
        "datasets_config": str(catalogue_path),
    }
    config_path = tmp_path / "prelim.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="PRELIMINARY"):
        runner.run_experiment(config_path, root=repo_root)


def test_constant_model_forecast_is_exact_on_constant_series():
    values = np.full(300, 42.0)
    plan = _plan(300)
    outcomes = backtest_model(values, plan, ConstantForecaster(value=42.0), series_id="s")
    assert all(outcome.metrics["mae"] == 0.0 for outcome in outcomes)


# ---------------------------------------------------------------------------
# Paired statistics over result rows
# ---------------------------------------------------------------------------


def _rows_for(model: str, scores: dict[str, float], horizon: int = 12) -> list[dict]:
    return [
        {
            "series_id": sid,
            "model": model,
            "horizon": horizon,
            "mase": value,
            "mae": value,
        }
        for sid, value in scores.items()
    ]


def test_aggregate_by_series_collapses_windows_and_drops_non_finite():
    rows = [
        {"series_id": "s1", "model": "a", "horizon": 12, "mase": 1.0},
        {"series_id": "s1", "model": "a", "horizon": 12, "mase": 3.0},
        {"series_id": "s2", "model": "a", "horizon": 12, "mase": ""},
        {"series_id": "s2", "model": "a", "horizon": 12, "mase": float("nan")},
    ]
    scores = stats.aggregate_by_series(rows, metric="mase", model="a")
    assert scores == {"s1": 2.0}


def test_wilcoxon_detects_a_consistent_difference():
    a = {f"s{i}": 2.0 for i in range(10)}
    b = {f"s{i}": 1.0 for i in range(10)}
    result = stats.wilcoxon_signed_rank(a, b, model_a="a", model_b="b")
    assert result.n_pairs == 10
    assert result.p_value < 0.01
    assert result.significant
    assert result.median_difference == pytest.approx(1.0)


def test_wilcoxon_with_identical_models_is_not_significant():
    a = {f"s{i}": float(i) for i in range(10)}
    result = stats.wilcoxon_signed_rank(a, dict(a))
    assert result.p_value == 1.0
    assert not result.significant
    assert result.n_pairs == 0


def test_wilcoxon_matches_known_rank_sum():
    # All positive differences -> the smallest possible statistic (0).
    a = {f"s{i}": 10.0 + i for i in range(6)}
    b = {f"s{i}": float(i) for i in range(6)}
    result = stats.wilcoxon_signed_rank(a, b)
    assert result.statistic == 0.0
    # exact two-sided p for n=6, W=0 is 2/64 = 0.03125.
    assert result.p_value == pytest.approx(2 / 64)
    assert result.method == "exact"


def test_wilcoxon_requires_shared_series():
    with pytest.raises(stats.StatsError):
        stats.wilcoxon_signed_rank({"a": 1.0}, {"b": 1.0})


def test_wilcoxon_one_sided_less_evaluates_the_exact_tail():
    # Ties in the absolute differences exercise the averaged-rank null.
    a = {0: 10, 1: 12, 2: 15, 3: 20, 4: 26, 5: 33, 6: 41, 7: 50}
    b = {0: 11, 1: 10, 2: 16, 3: 18, 4: 29, 5: 30, 6: 44, 7: 46}
    less = stats.wilcoxon_signed_rank(a, b, alternative="less")
    assert less.method == "exact"
    assert less.p_value == pytest.approx(0.66796875)
    # The two-sided default is unchanged.
    assert stats.wilcoxon_signed_rank(a, b).p_value == pytest.approx(0.6953125)


def test_wilcoxon_one_sided_greater_evaluates_the_exact_tail():
    a = {i: float(v) for i, v in enumerate([-1, -2, 3, -4, -5, -6, -7, -8])}
    b = {i: 0.0 for i in range(8)}
    greater = stats.wilcoxon_signed_rank(a, b, alternative="greater")
    assert greater.method == "exact"
    assert greater.p_value == pytest.approx(0.98828125)
    less = stats.wilcoxon_signed_rank(a, b, alternative="less")
    assert less.p_value == pytest.approx(0.01953125)


def test_wilcoxon_rejects_an_unknown_alternative():
    with pytest.raises(stats.StatsError, match="unknown alternative"):
        stats.wilcoxon_signed_rank(
            {f"s{i}": float(i) for i in range(4)},
            {f"s{i}": float(i) + 1 for i in range(4)},
            alternative="both",
        )


def test_friedman_test_and_critical_difference():
    scores = {
        "m1": {f"s{i}": float(i) for i in range(10)},
        "m2": {f"s{i}": float(i) + 0.5 for i in range(10)},
        "m3": {f"s{i}": float(i) + 1.0 for i in range(10)},
    }
    friedman = stats.friedman_test(scores)
    assert friedman.k_models == 3
    assert friedman.n_series == 10
    assert friedman.significant
    # m1 is best on every series, so its average rank is 1.
    assert friedman.average_ranks["m1"] == pytest.approx(1.0)
    assert friedman.average_ranks["m3"] == pytest.approx(3.0)
    cd = stats.critical_difference(scores)
    assert cd["critical_difference"] > 0
    assert cd["ranked_models"][0]["model"] == "m1"


def test_friedman_needs_at_least_three_models():
    scores = {
        "m1": {f"s{i}": float(i) for i in range(5)},
        "m2": {f"s{i}": float(i) + 1 for i in range(5)},
    }
    with pytest.raises(stats.StatsError):
        stats.friedman_test(scores)


def test_stats_run_on_a_real_backtest():
    values = _series(900)
    plan = _plan(900, context_length=72, horizon=12, stride=12)
    naive_rows = backtest_model(values, plan, NaiveStub(), series_id="s1")
    seasonal_rows = backtest_model(
        values, plan, SeasonalNaiveStub(season_length=24), series_id="s1"
    )
    rows = [
        {
            "series_id": o.series_id,
            "model": o.model,
            "horizon": o.horizon,
            "mase": o.metrics["mase"],
            "mae": o.metrics["mae"],
        }
        for o in [*naive_rows, *seasonal_rows]
    ]
    naive = stats.aggregate_by_series(rows, metric="mase", model="naive")
    seasonal = stats.aggregate_by_series(rows, metric="mase", model="seasonal_naive")
    result = stats.wilcoxon_signed_rank(
        naive, seasonal, model_a="naive", model_b="seasonal_naive"
    )
    assert result.median_difference > 0  # seasonal naive is better (lower MASE)


def test_friedman_test_all_ties_returns_zero_statistic():
    scores = {
        "m1": {"s1": 1.0, "s2": 2.0},
        "m2": {"s1": 1.0, "s2": 2.0},
        "m3": {"s1": 1.0, "s2": 2.0},
    }
    result = stats.friedman_test(scores)
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert all(rank == pytest.approx(2.0) for rank in result.average_ranks.values())


def test_measure_peak_memory_isolates_each_measurement():
    from tsbench.evaluation.backtest import measure_peak_memory

    start_window, stop_window = measure_peak_memory()
    large = bytearray(8_000_000)
    first = stop_window()
    assert first is not None and first > 1.0
    assert large  # keep the allocation alive across the first measurement

    del large
    import gc

    gc.collect()
    start_window()
    small = bytearray(2_000)
    second = stop_window()
    assert second is not None
    assert second < first
    assert small


def test_runner_rejects_split_fractions_that_disagree_with_manifest(tmp_path: Path):
    import pandas as pd
    import yaml

    from tsbench.data import SplitConfig, build_manifest, save_manifest

    values = np.arange(200, dtype=np.float64)

    class _S:
        series_id = "s"

        def __init__(self, values):
            self.values = values

    stamps = pd.date_range("2024-01-01", periods=values.size, freq="D")
    csv_path = tmp_path / "split_series.csv"
    pd.DataFrame({"timestamp": stamps, "value": values}).to_csv(csv_path, index=False)
    catalogue = {
        "version": 1,
        "default_dataset": "splitdata",
        "datasets": {
            "splitdata": {
                "loader": "local_csv",
                "path": str(csv_path),
                "license": "Apache-2.0",
                "series_id": "s",
            }
        },
    }
    catalogue_path = tmp_path / "datasets.yaml"
    catalogue_path.write_text(yaml.safe_dump(catalogue), encoding="utf-8")

    manifest_split = {
        "train_frac": 0.6,
        "val_frac": 0.2,
        "test_frac": 0.2,
        "context_length": 10,
        "horizon": 4,
        "stride": 4,
        "min_train_size": 5,
    }
    manifest = build_manifest(
        "splitdata", [_S(values)], SplitConfig.from_mapping(manifest_split)
    )
    manifest_path = tmp_path / "split_manifest.json"
    save_manifest(manifest, manifest_path)

    config = {
        "name": "splitdata",
        "dataset": {"name": "splitdata", "loader": "local_csv"},
        "split_manifest": str(manifest_path),
        "models": ["naive"],
        "split": {**manifest_split, "train_frac": 0.5, "val_frac": 0.25, "test_frac": 0.25},
        "datasets_config": str(catalogue_path),
    }
    config_path = tmp_path / "splitdata.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    with pytest.raises(runner.RunnerError, match="frozen manifest"):
        runner.run_experiment(config_path, root=repo_root)


def test_entrypoint_override_cannot_bypass_the_license_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import yaml

    from tsbench.registry import LicenseError, load_registry

    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "gated": {
                        "entrypoint": "tests._stub_models:NaiveStub",
                        "family": "baseline",
                        "zero_shot": True,
                        "license": "CC-BY-NC-4.0",
                        "revision": "test-pin",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    registry = load_registry(models_path)
    monkeypatch.setattr(runner, "_try_registry", lambda: registry)
    monkeypatch.delenv("TSBENCH_ALLOW_NONCOMMERCIAL", raising=False)
    with pytest.raises(LicenseError):
        runner._build_model(
            "gated",
            None,
            model_cfg={"entrypoint": "tests._stub_models:NaiveStub"},
            seed=None,
        )


def test_registry_entrypoint_wins_over_a_config_override_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import yaml

    from tsbench.registry import load_registry

    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "naive": {
                        "entrypoint": "tests._stub_models:NaiveStub",
                        "family": "baseline",
                        "zero_shot": True,
                        "license": "Apache-2.0",
                        "revision": "test-pin",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    registry = load_registry(models_path)
    monkeypatch.setattr(runner, "_try_registry", lambda: registry)

    model = runner._build_model(
        "naive",
        None,
        model_cfg={
            "entrypoint": "tests._stub_models:ConstantForecaster",
            "kwargs": {"value": 5.0},
        },
        seed=None,
    )
    assert isinstance(model, NaiveStub)
    model.fit(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(model.predict(2), [3.0, 3.0])


def test_config_entrypoint_is_used_when_the_registry_entrypoint_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import yaml

    from tsbench.registry import load_registry

    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "naive": {
                        "entrypoint": "tsbench.models.not_landed:Naive",
                        "family": "baseline",
                        "zero_shot": True,
                        "license": "Apache-2.0",
                        "revision": "test-pin",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    registry = load_registry(models_path)
    monkeypatch.setattr(runner, "_try_registry", lambda: registry)

    model = runner._build_model(
        "naive",
        None,
        model_cfg={"entrypoint": "tests._stub_models:NaiveStub"},
        seed=None,
    )
    assert isinstance(model, NaiveStub)


def test_config_entrypoint_fallback_still_honors_the_registry_license_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import yaml

    from tsbench.registry import LicenseError, load_registry

    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "gated": {
                        "entrypoint": "tsbench.models.not_landed:Gated",
                        "family": "baseline",
                        "zero_shot": True,
                        "license": "CC-BY-NC-4.0",
                        "revision": "test-pin",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    registry = load_registry(models_path)
    monkeypatch.setattr(runner, "_try_registry", lambda: registry)
    monkeypatch.delenv("TSBENCH_ALLOW_NONCOMMERCIAL", raising=False)
    with pytest.raises(LicenseError):
        runner._build_model(
            "gated",
            None,
            model_cfg={"entrypoint": "tests._stub_models:NaiveStub"},
            seed=None,
        )


def test_registry_baselines_construct_through_runner_with_a_seed(
    monkeypatch: pytest.MonkeyPatch,
):
    """A top-level seed must not crash baselines that reject the kwarg."""
    from tsbench.models.naive import Naive, SeasonalNaive
    from tsbench.registry import load_registry

    repo_root = Path(__file__).resolve().parents[1]
    registry = load_registry(repo_root / "configs" / "models.yaml")
    monkeypatch.setattr(runner, "_try_registry", lambda: registry)

    naive = runner._build_model("naive", None, model_cfg={}, seed=20240919)
    assert isinstance(naive, Naive)
    naive.fit(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(naive.predict(2), [3.0, 3.0])

    seasonal = runner._build_model(
        "seasonal_naive",
        None,
        model_cfg={"kwargs": {"season_length": 24}},
        seed=20240919,
    )
    assert isinstance(seasonal, SeasonalNaive)
    assert seasonal.season_length == 24


def test_seed_is_forwarded_only_when_the_constructor_accepts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import yaml

    from tsbench.registry import load_registry

    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "stub": {
                        "entrypoint": "tests._stub_models:NaiveStub",
                        "family": "baseline",
                        "zero_shot": True,
                        "license": "Apache-2.0",
                        "revision": "test-pin",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    registry = load_registry(models_path)
    monkeypatch.setattr(runner, "_try_registry", lambda: registry)

    seeded = runner._build_model("stub", None, model_cfg={}, seed=7)
    assert seeded.extra["seed"] == 7

    suppressed = runner._build_model("stub", None, model_cfg={"seed": False}, seed=7)
    assert "seed" not in suppressed.extra


def test_runner_translates_a_missing_manifest_to_runner_error(tmp_path: Path):
    import yaml

    repo_root = Path(__file__).resolve().parents[1]
    config = {
        "name": "missing-manifest",
        "dataset": {"name": "smoke", "loader": "local_csv"},
        "split_manifest": str(tmp_path / "does_not_exist.json"),
        "models": ["naive"],
        "split": {
            "train_frac": 0.6,
            "val_frac": 0.2,
            "test_frac": 0.2,
            "context_length": 20,
            "horizon": 4,
            "stride": 4,
            "min_train_size": 5,
        },
        "datasets_config": str(repo_root / "configs" / "datasets.yaml"),
    }
    config_path = tmp_path / "missing.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="split manifest not found"):
        runner.run_experiment(config_path, root=repo_root)


def test_runner_translates_a_missing_catalogue_to_runner_error(tmp_path: Path):
    import yaml

    config = {
        "name": "missing-catalogue",
        "dataset": {"name": "electricity_hourly"},
        "models": ["naive"],
        "split": {"context_length": 20, "horizon": 4},
        "datasets_config": str(tmp_path / "no_such_catalogue.yaml"),
    }
    config_path = tmp_path / "missing_catalogue.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="dataset catalogue not found"):
        runner.run_experiment(config_path, root=tmp_path)

