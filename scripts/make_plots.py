#!/usr/bin/env python3
"""Aggregate the telemetry stores and regenerate the results figures.

The evaluation runner writes one directory per run::

    <root>/<run_id>/results.csv    one row per (series, model, window)
    <root>/<run_id>/run.json       provenance (git sha, config hash, split)

This script reads the live ``results/`` store plus the committed
``docs/telemetry/`` store (both runs, CPU and GPU), collapses the union to one
summary row per ``(model, family, horizon)``, and writes:

* ``docs/results_summary.csv`` -- the committed machine-readable summary;
* ``docs/figures/mase_by_family_horizon.png`` -- accuracy per family per horizon;
* ``docs/figures/accuracy_vs_cost.png`` -- the accuracy/cost scatter.

Several runs may cover the same dataset (for example a horizon sweep, or a
re-run after a fix). Each run carries a ``config_hash``; only the newest run per
hash is used so a re-run replaces rather than double-counts its predecessor.
Roots are ordered by precedence: each experiment is owned by the first root
that provides it, so a fresh ``results/`` run supersedes the committed telemetry
copy of the same experiment instead of blending two config hashes for it.

Examples::

    python scripts/make_plots.py
    python scripts/make_plots.py --results-root results --telemetry-root docs/telemetry
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

SUMMARY_COLUMNS = [
    "dataset",
    "horizon",
    "family",
    "model",
    "context_length",
    "n_windows",
    "n_series",
    "mae",
    "rmse",
    "mase",
    "smape",
    "latency_p50_ms",
    "latency_p95_ms",
    "peak_mem_mb",
    "train_seconds",
    "params",
    "zero_shot",
]

# Colour-blind-friendly qualitative palette, stable per family.
_FAMILY_COLOURS = {
    "baseline": "#4c72b0",
    "classical": "#dd8452",
    "ml": "#55a868",
    "deep": "#c44e52",
    "tsfm": "#8172b3",
}


class ReportError(RuntimeError):
    """The telemetry store could not be summarised."""


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series([float("nan")] * len(frame), index=frame.index)
    return pd.to_numeric(frame[column].replace("", None), errors="coerce")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value == value:
        return value != 0
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_runs(
    root: Path, *, dataset: str | None
) -> list[tuple[Path, dict[str, Any], pd.DataFrame]]:
    """Read every run directory under one root (no dedupe yet).

    Run dirs are discovered recursively so a store organised by tier
    (``docs/telemetry/cpu/<run_id>`` and ``docs/telemetry/gpu/<run_id>``) merges
    with the flat live ``results/<run_id>`` layout.
    """
    found: list[tuple[Path, dict[str, Any], pd.DataFrame]] = []
    for csv_path in sorted(root.rglob("results.csv")):
        directory = csv_path.parent
        metadata: dict[str, Any] = {}
        run_json = directory / "run.json"
        if run_json.is_file():
            try:
                metadata = json.loads(run_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"warning: ignoring {run_json}: {exc}", file=sys.stderr)
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        if dataset is not None and "dataset" in frame:
            frame = frame[frame["dataset"] == dataset]
        if frame.empty:
            continue
        metadata["_dir"] = directory
        found.append((directory, metadata, frame))
    return found


def _experiment(metadata: dict[str, Any], run_dir: Path) -> str:
    """The experiment identity that owns a run's rows across roots."""
    return str(metadata.get("experiment") or run_dir.name)


def load_runs(
    roots: str | Path | Sequence[str | Path], *, dataset: str | None = None
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load runs from every root, keeping the newest per ``config_hash``.

    ``roots`` is ordered by precedence; each experiment is owned by the first
    root that provides it, so a fresh ``results/`` run supersedes the committed
    telemetry copy of the same-experiment run instead of being merged with it
    (two config hashes for one experiment never blend). If different
    experiments still disagree on a summary row, :func:`summarise` fails loudly.
    ``roots`` may be a single path or a sequence; missing roots are skipped so a
    checkout without a live run directory can still regenerate the summary from
    the committed telemetry alone.
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]
    frames: list[pd.DataFrame] = []
    runs: list[dict[str, Any]] = []
    claimed: set[str] = set()
    for root in roots:
        directory = Path(root)
        if not directory.is_dir():
            continue
        newest: dict[str, tuple[Path, dict[str, Any], pd.DataFrame]] = {}
        for run_dir, metadata, frame in _read_runs(directory, dataset=dataset):
            config_hash = str(metadata.get("config_hash") or run_dir.name)
            previous = newest.get(config_hash)
            if previous is None or run_dir.name > previous[0].name:
                newest[config_hash] = (run_dir, metadata, frame)
        root_experiments: set[str] = set()
        for run_dir, metadata, frame in sorted(
            newest.values(), key=lambda item: item[0].name
        ):
            experiment = _experiment(metadata, run_dir)
            if experiment in claimed:
                continue
            frame = frame.copy()
            frame["_run_id"] = run_dir.name
            frame["_git_sha"] = str(metadata.get("git_sha", "") or "")
            frame["_config_hash"] = str(metadata.get("config_hash", "") or "")
            frame["_root"] = str(directory)
            frames.append(frame)
            runs.append(
                {
                    "run_id": run_dir.name,
                    "experiment": metadata.get("experiment"),
                    "config_hash": metadata.get("config_hash"),
                    "git_sha": metadata.get("git_sha"),
                    "root": str(directory),
                    "n_rows": int(len(frame)),
                }
            )
            root_experiments.add(experiment)
        claimed.update(root_experiments)
    if not frames:
        listed = ", ".join(str(Path(r)) for r in roots)
        raise ReportError(
            f"no usable results under {listed}"
            + (f" for dataset {dataset!r}" if dataset else "")
            + "; run `make backtest` first, or add a telemetry root"
        )
    return pd.concat(frames, ignore_index=True), runs


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse window rows to one summary row per ``(model, family, horizon)``.

    A summary row must come from a single run. Two runs with different configs
    can cover the same model and horizon (a stale run left behind by an
    in-place config rewrite, or a full-stride follow-up); averaging them
    silently would double-count windows, so that overlap is a hard error.
    """
    work = frame.copy()
    for column in (
        "mae",
        "rmse",
        "mase",
        "smape",
        "latency_ms",
        "train_seconds",
        "peak_mem_mb",
    ):
        work[column] = _numeric(work, column)
    work["horizon"] = pd.to_numeric(work["horizon"], errors="coerce").astype("Int64")
    work["context_length"] = pd.to_numeric(
        work["context_length"], errors="coerce"
    ).astype("Int64")
    if "_run_id" in work:
        identity = work["_run_id"].astype(str)
    else:
        identity = work.get("config_hash", pd.Series("", index=work.index)).astype(str)

    def _percentile(series: pd.Series, q: float) -> float:
        clean = series.dropna()
        return float(clean.quantile(q)) if not clean.empty else float("nan")

    def _first_present(series: pd.Series) -> Any:
        for value in series:
            if value is None:
                continue
            if isinstance(value, float) and value != value:
                continue
            if str(value).strip() == "":
                continue
            return value
        return None

    rows: list[dict[str, Any]] = []
    keys = ["dataset", "horizon", "family", "model", "context_length"]
    for key, group in work.groupby(keys, dropna=False):
        dataset, horizon, family, model, context_length = key
        run_ids = sorted(set(identity.loc[group.index]))
        if len(run_ids) > 1:
            detail = (
                f"{model!r} ({family}) at horizon {horizon} appears in multiple "
                f"runs ({', '.join(run_ids)})"
            )
            if "_config_hash" in work:
                hashes = sorted(
                    {str(value) for value in work["_config_hash"].loc[group.index]}
                )
                detail += f" with config hashes {hashes}"
            if "_root" in work:
                roots = sorted(
                    {str(value) for value in work["_root"].loc[group.index]}
                )
                detail += f" under roots {roots}"
            raise ReportError(
                f"ambiguous results: {detail}; remove the stale run or scope "
                "--results-root to one config set so distinct configs are not "
                "averaged"
            )
        scored = int(group["mae"].notna().sum())
        rows.append(
            {
                "dataset": dataset,
                "horizon": int(horizon) if pd.notna(horizon) else None,
                "family": family,
                "model": model,
                "context_length": int(context_length) if pd.notna(context_length) else None,
                "n_windows": scored,
                "n_series": int(group["series_id"].nunique()),
                "mae": float(group["mae"].mean()),
                "rmse": float(group["rmse"].mean()),
                "mase": float(group["mase"].mean()),
                "smape": float(group["smape"].mean()),
                "latency_p50_ms": _percentile(group["latency_ms"], 0.5),
                "latency_p95_ms": _percentile(group["latency_ms"], 0.95),
                "peak_mem_mb": float(group["peak_mem_mb"].max())
                if group["peak_mem_mb"].notna().any()
                else float("nan"),
                "train_seconds": _percentile(group["train_seconds"], 0.5),
                "params": _first_present(group["params"])
                if "params" in group
                else None,
                "zero_shot": any(_truthy(value) for value in group["zero_shot"])
                if "zero_shot" in group
                else False,
            }
        )
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary = summary.sort_values(
        by=["horizon", "mase", "model"], kind="stable"
    ).reset_index(drop=True)
    return summary


def format_markdown(summary: pd.DataFrame) -> str:
    """Render the summary as a GitHub-flavoured markdown table."""
    headers = [
        "horizon",
        "family",
        "model",
        "MAE",
        "RMSE",
        "MASE",
        "sMAPE",
        "latency p50 (ms)",
        "latency p95 (ms)",
        "peak mem (MB)",
        "train (s)",
        "params",
        "0-shot",
    ]

    def _fmt(value: Any, digits: int = 4) -> str:
        if value is None or (isinstance(value, float) and value != value):
            return "n/a"
        return f"{float(value):.{digits}f}"

    def _fmt_optional(value: Any, digits: int = 4) -> str:
        if value is None or (isinstance(value, float) and value != value):
            return "-"
        if isinstance(value, bool):
            return "yes" if value else "no"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["horizon"])) if pd.notna(row["horizon"]) else "n/a",
                    str(row["family"]),
                    str(row["model"]),
                    _fmt(row["mae"]),
                    _fmt(row["rmse"]),
                    _fmt(row["mase"]),
                    _fmt(row["smape"], 2),
                    f"{row['latency_p50_ms']:.1f}"
                    if pd.notna(row["latency_p50_ms"])
                    else "n/a",
                    f"{row['latency_p95_ms']:.1f}"
                    if pd.notna(row["latency_p95_ms"])
                    else "n/a",
                    _fmt_optional(row.get("peak_mem_mb")),
                    _fmt_optional(row.get("train_seconds"), 3),
                    _fmt_optional(row.get("params"), 0),
                    _fmt_optional(row.get("zero_shot")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _import_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ReportError(
            "matplotlib is required for figures; install with "
            "`pip install -e \".[report]\"`"
        ) from exc
    return plt


def plot_mase_by_family_horizon(summary: pd.DataFrame, path: Path) -> None:
    plt = _import_pyplot()
    models = list(dict.fromkeys(summary["model"]))
    horizons = list(dict.fromkeys(summary["horizon"].dropna().astype(int)))
    if not models or not horizons:
        return
    width = 0.8 / len(models)
    x = range(len(horizons))
    fig, ax = plt.subplots(figsize=(1.9 * len(horizons) + 3.5, 5.0))
    for index, model in enumerate(models):
        heights: list[float] = []
        for horizon in horizons:
            match = summary[(summary["model"] == model) & (summary["horizon"] == horizon)]
            heights.append(float(match["mase"].iloc[0]) if not match.empty else 0.0)
        family = str(summary[summary["model"] == model]["family"].iloc[0])
        offset = (index - (len(models) - 1) / 2) * width
        ax.bar(
            [position + offset for position in x],
            heights,
            width=width,
            label=model,
            color=_FAMILY_COLOURS.get(family, "#666666"),
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("forecast horizon (steps)")
    ax.set_ylabel("seasonal MASE (mean, log scale)")
    ax.set_title("Seasonal MASE by family and horizon")
    ax.grid(axis="y", which="both", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, ncols=min(len(models), 5), fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_accuracy_vs_cost(summary: pd.DataFrame, path: Path) -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for _, row in summary.iterrows():
        x = row["mase"]
        y = row["latency_p50_ms"]
        if pd.isna(x) or pd.isna(y) or y <= 0:
            continue
        colour = _FAMILY_COLOURS.get(str(row["family"]), "#666666")
        marker = {24: "o", 48: "s", 96: "^", 192: "D"}.get(int(row["horizon"]), "o")
        ax.scatter(x, y, color=colour, marker=marker, s=70, edgecolor="white", zorder=3)
        ax.annotate(
            f"{row['model']} (h{int(row['horizon'])})",
            (x, y),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
            alpha=0.85,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("seasonal MASE (mean, lower is better, log scale)")
    ax.set_ylabel("latency p50 (ms, lower is cheaper, log scale)")
    ax.set_title("Accuracy vs inference cost")
    ax.grid(which="both", alpha=0.25, linewidth=0.6)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colour, label=family)
        for family, colour in _FAMILY_COLOURS.items()
        if family in set(summary["family"])
    ]
    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=8, title="family")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the results summary and figures")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--telemetry-root",
        type=Path,
        default=Path("docs/telemetry"),
        help="committed telemetry store, merged with --results-root",
    )
    parser.add_argument("--dataset", default="electricity_hourly")
    parser.add_argument("--summary", type=Path, default=Path("docs/results_summary.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--no-figures", action="store_true", help="write the summary only"
    )
    args = parser.parse_args(argv)
    roots = [args.results_root, args.telemetry_root]
    try:
        frame, runs = load_runs(roots, dataset=args.dataset or None)
        summary = summarise(frame)
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    print(format_markdown(summary))
    print()
    print(f"summary: {args.summary} ({len(summary)} rows from {len(runs)} run(s))")
    if not args.no_figures:
        try:
            plot_mase_by_family_horizon(
                summary, args.figures_dir / "mase_by_family_horizon.png"
            )
            plot_accuracy_vs_cost(
                summary, args.figures_dir / "accuracy_vs_cost.png"
            )
        except ReportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"figures: {args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
