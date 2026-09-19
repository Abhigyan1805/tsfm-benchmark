"""Result rows and run metadata.

Every experiment writes one row per (series, model, window) carrying the exact
``RESULT_COLUMNS`` schema from ``tsbench.base``. Rows are validated on the way
out so a schema drift fails the run instead of producing an unreadable results
directory. Runs are laid out as::

    results/<run_id>/results.csv
    results/<run_id>/run.json

``run.json`` records the git sha, config hash, split-manifest digest, and the
measured-vs-estimated provenance flag. ``train_seconds`` is present in every row
and is empty (``""``) for untrained families, as the schema requires.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contract import RESULT_COLUMNS

__all__ = [
    "SchemaError",
    "RunContext",
    "config_hash",
    "current_git_sha",
    "git_dirty",
    "make_run_id",
    "result_row",
    "results_path",
    "utc_now",
    "validate_row",
    "write_results",
]

# Families that do not train a model in a given window; their train_seconds is
# empty rather than 0.0 so "not trained" is distinguishable from "trained fast".
UNTRAINED_FAMILIES = frozenset({"baseline", "tsfm"})


class SchemaError(ValueError):
    """A result row does not conform to ``RESULT_COLUMNS``."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable sha256 over a config mapping (order-insensitive)."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def current_git_sha(cwd: str | Path | None = None) -> str:
    """Return the current commit sha, or ``"unknown"`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def git_dirty(cwd: str | Path | None = None) -> bool:
    """True when the working tree has uncommitted changes (or no repo)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(out.stdout.strip())


def make_run_id(experiment: str, *, stamp: str | None = None) -> str:
    """Build a filesystem-safe, sortable run id."""
    moment = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in experiment)
    return f"{moment}-{safe}"


@dataclass
class RunContext:
    """Immutable provenance for one experiment run."""

    run_id: str
    experiment: str
    config: Mapping[str, Any]
    git_sha: str
    config_sha: str
    split_manifest: str | None = None
    seed: int | None = None
    measured: bool = True
    started_at: str = field(default_factory=utc_now)
    environment: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        experiment: str,
        config: Mapping[str, Any],
        *,
        run_id: str | None = None,
        split_manifest: str | None = None,
        seed: int | None = None,
        measured: bool = True,
        cwd: str | Path | None = None,
    ) -> RunContext:
        return cls(
            run_id=run_id or make_run_id(experiment),
            experiment=experiment,
            config=dict(config),
            git_sha=current_git_sha(cwd),
            config_sha=config_hash(config),
            split_manifest=split_manifest,
            seed=seed,
            measured=measured,
            environment={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "dirty": git_dirty(cwd),
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "git_sha": self.git_sha,
            "config_hash": self.config_sha,
            "config": self.config,
            "split_manifest": self.split_manifest,
            "seed": self.seed,
            "measured": self.measured,
            "started_at": self.started_at,
            "environment": self.environment,
        }


def result_row(
    *,
    run_id: str,
    dataset: str,
    series_id: str,
    model: str,
    family: str,
    context_length: int,
    horizon: int,
    window_index: int,
    mae: float,
    rmse: float,
    mase: float,
    smape: float,
    git_sha: str,
    config_hash: str,
    latency_ms: float | None = None,
    peak_mem_mb: float | None = None,
    params: int | float | str | None = None,
    zero_shot: bool | None = None,
    train_seconds: float | None = None,
    measured: bool = True,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a row with exactly the ``RESULT_COLUMNS`` schema.

    ``train_seconds`` is empty for untrained families (baseline, zero-shot
    TSFM) regardless of what the backtest measured, so "not trained" is never
    confused with "trained in N seconds". For trained families it is the
    measured fit time, or empty if the caller did not supply one.
    """
    train_value: float | str
    if family in UNTRAINED_FAMILIES:
        train_value = ""
    elif train_seconds is None:
        train_value = ""
    else:
        train_value = float(train_seconds)
    row: dict[str, Any] = {
        "run_id": run_id,
        "dataset": dataset,
        "series_id": series_id,
        "model": model,
        "family": family,
        "context_length": int(context_length),
        "horizon": int(horizon),
        "window_index": int(window_index),
        "mae": mae,
        "rmse": rmse,
        "mase": mase,
        "smape": smape,
        "latency_ms": latency_ms,
        "peak_mem_mb": peak_mem_mb,
        "params": params,
        "zero_shot": zero_shot,
        "train_seconds": train_value,
        "git_sha": git_sha,
        "config_hash": config_hash,
        "measured": measured,
        "timestamp": timestamp or utc_now(),
    }
    validate_row(row)
    return row


def validate_row(row: Mapping[str, Any]) -> None:
    """Raise :class:`SchemaError` unless ``row`` matches ``RESULT_COLUMNS``."""
    keys = tuple(row.keys())
    if keys != RESULT_COLUMNS:
        missing = [c for c in RESULT_COLUMNS if c not in row]
        extra = [c for c in row if c not in RESULT_COLUMNS]
        raise SchemaError(
            "result row does not match RESULT_COLUMNS "
            f"(missing={missing}, extra={extra}, order_ok={keys == RESULT_COLUMNS})"
        )


def results_path(root: str | Path, run_id: str, name: str = "results.csv") -> Path:
    return Path(root) / run_id / name


def write_results(
    rows: Iterable[Mapping[str, Any]],
    *,
    root: str | Path,
    context: RunContext,
) -> Path:
    """Write rows and run metadata under ``results/<run_id>/``.

    Rows are validated and written in schema order. The run record is written
    even for an empty result set so a failed run leaves a trace.
    """
    import csv

    directory = Path(root) / context.run_id
    directory.mkdir(parents=True, exist_ok=True)
    row_list = [dict(row) for row in rows]
    for row in row_list:
        validate_row(row)
    csv_file = directory / "results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_COLUMNS))
        writer.writeheader()
        for row in row_list:
            writer.writerow({col: _csv_value(row[col]) for col in RESULT_COLUMNS})
    metadata = context.as_dict()
    metadata["n_rows"] = len(row_list)
    metadata["finished_at"] = utc_now()
    (directory / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return csv_file


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def read_results(path: str | Path) -> list[dict[str, Any]]:
    """Read a results CSV back into rows (used by stats and tests)."""
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
