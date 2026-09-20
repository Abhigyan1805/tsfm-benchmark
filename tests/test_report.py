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


def test_make_plots_reports_an_empty_store_without_a_traceback(tmp_path: Path):
    root = tmp_path / "results"
    root.mkdir()
    proc = _run_script(root, tmp_path / "summary.csv", tmp_path / "figures", "--no-figures")
    assert proc.returncode == 2
    assert "error:" in proc.stderr
    assert "Traceback" not in proc.stderr


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
