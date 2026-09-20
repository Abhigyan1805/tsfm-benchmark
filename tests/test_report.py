"""Integration coverage for the telemetry-store report script.

``scripts/make_plots.py`` is the executable contract behind ``make report``: it
reads the run directories under ``results/``, keeps the newest run per
``config_hash`` (so a re-run replaces its predecessor), collapses rows to one
summary row per ``(model, family, horizon)``, and regenerates the figures.
These tests drive the script over a synthesised store, so they never need the
real dataset or network.
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from tsbench.evaluation.results import RunContext, result_row, write_results

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "make_plots.py"
DATASET = "electricity_hourly"


def _write_run(
    root: Path,
    run_id: str,
    config: dict,
    specs: list[tuple[str, str, int, float, float]],
) -> None:
    """Write one run directory with ``specs = (model, family, horizon, mae, latency)``."""
    context = RunContext.create(config["name"], config, run_id=run_id, cwd=root)
    rows = [
        result_row(
            run_id=run_id,
            dataset=DATASET,
            series_id=f"S{index}",
            model=model,
            family=family,
            context_length=168,
            horizon=horizon,
            window_index=index,
            mae=mae,
            rmse=mae * 2.0,
            mase=mae / 10.0,
            smape=mae,
            latency_ms=latency,
            train_seconds=0.5 if family not in {"baseline", "tsfm"} else "",
            git_sha=context.git_sha,
            config_hash=context.config_sha,
        )
        for index, (model, family, horizon, mae, latency) in enumerate(specs)
    ]
    write_results(rows, root=root, context=context)


def _run_script(root: Path, summary: Path, figures: Path, *extra: str):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--results-root",
            str(root),
            "--dataset",
            DATASET,
            "--summary",
            str(summary),
            "--figures-dir",
            str(figures),
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_make_plots_dedupes_config_hashes_and_summarises(tmp_path: Path):
    root = tmp_path / "results"
    config = {"name": "backtest", "horizon": 24}
    # Same config hash: the older run must be replaced by the newer one.
    _write_run(root, "20240101T000000Z-old", config, [("naive", "baseline", 24, 100.0, 0.1)])
    _write_run(root, "20240102T000000Z-new", config, [("naive", "baseline", 24, 2.0, 0.2)])
    # A second config (different horizon) is kept alongside.
    _write_run(
        root,
        "20240103T000000Z-h48",
        {"name": "backtest_h48", "horizon": 48},
        [("auto_ets", "classical", 48, 4.0, 30.0)],
    )

    summary_path = tmp_path / "summary.csv"
    proc = _run_script(root, summary_path, tmp_path / "figures", "--no-figures")
    assert proc.returncode == 0, proc.stderr
    assert "naive" in proc.stdout

    with open(summary_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    by_model = {row["model"]: row for row in rows}
    assert float(by_model["naive"]["mae"]) == pytest.approx(2.0)
    assert float(by_model["naive"]["mase"]) == pytest.approx(0.2)
    assert int(by_model["auto_ets"]["horizon"]) == 48
    assert float(by_model["auto_ets"]["train_seconds"]) == pytest.approx(0.5)
    # Baseline families have no measured training wall-clock.
    assert by_model["naive"]["train_seconds"].strip() == ""


def test_make_plots_refuses_to_average_distinct_configs_at_the_same_horizon(
    tmp_path: Path,
):
    root = tmp_path / "results"
    summary_path = tmp_path / "summary.csv"
    _write_run(
        root,
        "20240101T000000Z-backtest",
        {"name": "backtest", "horizon": 24},
        [("naive", "baseline", 24, 2.0, 0.2)],
    )
    proc = _run_script(root, summary_path, tmp_path / "figures", "--no-figures")
    assert proc.returncode == 0, proc.stderr

    # A second, differently-configured run covers the same model and horizon.
    # It must not be averaged into the first run's clean summary row.
    _write_run(
        root,
        "20240102T000000Z-full",
        {"name": "backtest_full", "horizon": 24, "stride": 24},
        [("naive", "baseline", 24, 100.0, 0.2)],
    )
    proc = _run_script(root, summary_path, tmp_path / "figures", "--no-figures")
    assert proc.returncode == 2, proc.stdout
    assert "ambiguous" in proc.stderr
    assert "Traceback" not in proc.stderr

    with open(summary_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["model"] for row in rows] == ["naive"]
    assert float(rows[0]["mae"]) == pytest.approx(2.0)
    assert int(rows[0]["n_windows"]) == 1


def test_make_plots_reports_an_empty_store_without_a_traceback(tmp_path: Path):
    root = tmp_path / "results"
    root.mkdir()
    proc = _run_script(root, tmp_path / "summary.csv", tmp_path / "figures", "--no-figures")
    assert proc.returncode == 2
    assert "error:" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_make_plots_carries_peak_memory_params_and_zero_shot(tmp_path: Path):
    root = tmp_path / "results"
    root.mkdir()
    context = RunContext.create(
        "gpu", {"name": "gpu", "horizon": 24}, run_id="20240102T000000Z-gpu", cwd=root
    )
    rows = [
        result_row(
            run_id=context.run_id,
            dataset=DATASET,
            series_id="S0",
            model="timesfm25",
            family="tsfm",
            context_length=168,
            horizon=24,
            window_index=0,
            mae=90.0,
            rmse=120.0,
            mase=0.9,
            smape=9.0,
            latency_ms=30.0,
            peak_mem_mb=812.5,
            params=231289280,
            zero_shot=True,
            git_sha=context.git_sha,
            config_hash=context.config_sha,
        ),
        result_row(
            run_id=context.run_id,
            dataset=DATASET,
            series_id="S0",
            model="lstm",
            family="deep",
            context_length=168,
            horizon=24,
            window_index=0,
            mae=95.0,
            rmse=125.0,
            mase=0.95,
            smape=9.5,
            latency_ms=12.0,
            peak_mem_mb=3.5,
            params=1234,
            zero_shot=False,
            git_sha=context.git_sha,
            config_hash=context.config_sha,
        ),
    ]
    write_results(rows, root=root, context=context)

    # A committed telemetry root is merged with the live store.
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    old = RunContext.create(
        "backtest",
        {"name": "backtest", "horizon": 24},
        run_id="20240101T000000Z-backtest",
        cwd=root,
    )
    write_results(
        [
            result_row(
                run_id=old.run_id,
                dataset=DATASET,
                series_id="S1",
                model="naive",
                family="baseline",
                context_length=168,
                horizon=24,
                window_index=0,
                mae=1.0,
                rmse=2.0,
                mase=1.0,
                smape=1.0,
                latency_ms=0.1,
                git_sha=old.git_sha,
                config_hash=old.config_sha,
            )
        ],
        root=telemetry,
        context=old,
    )

    summary_path = tmp_path / "summary.csv"
    proc = _run_script(
        root,
        summary_path,
        tmp_path / "figures",
        "--no-figures",
        "--telemetry-root",
        str(telemetry),
    )
    assert proc.returncode == 0, proc.stderr
    with open(summary_path, newline="", encoding="utf-8") as handle:
        by_model = {row["model"]: row for row in csv.DictReader(handle)}
    assert set(by_model) == {"timesfm25", "lstm", "naive"}
    assert float(by_model["timesfm25"]["peak_mem_mb"]) == pytest.approx(812.5)
    assert float(by_model["timesfm25"]["params"]) == pytest.approx(231289280)
    assert by_model["timesfm25"]["zero_shot"] == "True"
    assert by_model["lstm"]["zero_shot"] == "False"
    # The baseline from the committed telemetry root is present, and its
    # untrained train_seconds is empty rather than zero.
    assert by_model["naive"]["train_seconds"].strip() == ""


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="matplotlib is not installed",
)
def test_make_plots_writes_the_figures(tmp_path: Path):
    root = tmp_path / "results"
    _write_run(
        root,
        "20240102T000000Z-h24",
        {"name": "backtest", "horizon": 24},
        [
            ("naive", "baseline", 24, 2.0, 0.2),
            ("auto_ets", "classical", 24, 1.0, 30.0),
        ],
    )
    figures = tmp_path / "figures"
    proc = _run_script(root, tmp_path / "summary.csv", figures)
    assert proc.returncode == 0, proc.stderr
    assert (figures / "mase_by_family_horizon.png").is_file()
    assert (figures / "accuracy_vs_cost.png").is_file()
