#!/usr/bin/env python3
"""Colab-side GPU runner for tsfm-benchmark (see docs/colab-handoff.md).

Subcommands:

* ``env-check`` -- report python/torch/CUDA/model-package availability and the
  ``TSBENCH_ALLOW_MODEL_DOWNLOAD`` gate state. Stdlib-only.
* ``run`` -- execute a JSON job file against the frozen model registry on this
  machine's GPU, writing one result JSON per job. Results are keyed by a
  content cache key (job spec + series + runner version), so re-running after a
  disconnect skips completed jobs and only fills the gap.
* ``bundle`` -- archive a run directory (manifest + results) for download back
  to the D-drive clone, printing the archive's sha256.

Examples:
    python scripts/colab_run.py env-check --require torch --require-gpu
    python scripts/colab_run.py run --jobs jobs.json --out results/colab-01
    python scripts/colab_run.py bundle --out results/colab-01
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNNER_VERSION = "1"
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DOWNLOAD_ENV = "TSBENCH_ALLOW_MODEL_DOWNLOAD"

DEFAULT_ENTRIES = {
    "lstm": "tsbench.models.deep.lstm:LSTMForecaster",
    "transformer": "tsbench.models.deep.transformer:TransformerForecaster",
    "timesfm25": "tsbench.models.tsfm.timesfm:TimesFM25",
    "chronos_bolt": "tsbench.models.tsfm.chronos:ChronosBolt",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _add_src_to_path() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _floats(values: Any) -> list[float]:
    out: list[float] = []
    for item in values:
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"non-finite series value: {item!r}")
        out.append(number)
    return out


def load_series(job: dict[str, Any]) -> list[float]:
    """Read a series from ``series`` (inline) or ``series_path`` (json/csv)."""
    if "series" in job:
        series = _floats(job["series"])
    elif "series_path" in job:
        path = Path(str(job["series_path"]))
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("series", payload.get("values"))
            if not isinstance(payload, list):
                raise ValueError(f"{path} does not contain a JSON series list")
            series = _floats(payload)
        elif path.suffix.lower() == ".csv":
            series = _load_csv_series(path, job.get("column"))
        else:
            raise ValueError(f"unsupported series file type: {path.suffix!r}")
    else:
        raise ValueError("job needs 'series' or 'series_path'")
    if not series:
        raise ValueError("series is empty")
    return series


def _load_csv_series(path: Path, column: str | None) -> list[float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    values: list[float] = []
    index = 0
    if column:
        header = rows[0]
        if column not in header:
            raise ValueError(f"column {column!r} not found in {path}")
        index = header.index(column)
        rows = rows[1:]
    for row in rows:
        if index >= len(row):
            continue
        try:
            values.append(float(row[index]))
        except ValueError:
            continue
    return values


def cache_key(job: dict[str, Any], series: list[float] | None = None) -> str:
    """Content-addressed key: job spec + series content + runner version."""
    payload = {
        "runner": RUNNER_VERSION,
        "job": job,
        "series_hash": (
            hashlib.sha256(canonical_json(series).encode("utf-8")).hexdigest()
            if series is not None
            else None
        ),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def load_entry(entry: str) -> Any:
    module_name, _, attribute = entry.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"entry must be 'module:Class', got {entry!r}")
    _add_src_to_path()
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def _metrics(predictions: list[float], holdout: list[float]) -> dict[str, float]:
    errors = [pred - actual for pred, actual in zip(predictions, holdout, strict=True)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return {"mae": mae, "rmse": rmse, "n": len(errors)}


def execute_job(job: dict[str, Any], series: list[float], key: str) -> dict[str, Any]:
    horizon = int(job["horizon"])
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if len(series) < horizon + 2:
        raise ValueError(f"need at least horizon + 2 points, got {len(series)}")
    model_key = job.get("model")
    entry = job.get("entry") or (
        DEFAULT_ENTRIES.get(str(model_key)) if model_key is not None else None
    )
    if not entry:
        raise ValueError(
            f"unknown model {model_key!r}; known: {sorted(DEFAULT_ENTRIES)} "
            "or pass an explicit 'entry'"
        )
    model_class = load_entry(str(entry))
    model = model_class(**dict(job.get("params") or {}))
    train, holdout = series[:-horizon], series[-horizon:]
    start = time.monotonic()
    model.fit(train)
    fit_seconds = time.monotonic() - start
    start = time.monotonic()
    predictions = [float(value) for value in model.predict(horizon)]
    predict_seconds = time.monotonic() - start
    if len(predictions) != horizon:
        raise ValueError(
            f"model returned {len(predictions)} points for horizon {horizon}"
        )
    return {
        "cache_key": key,
        "model": model_key or entry,
        "entry": entry,
        "params": dict(job.get("params") or {}),
        "horizon": horizon,
        "n_train": len(train),
        "metrics": _metrics(predictions, holdout),
        "predictions": predictions,
        "holdout": holdout,
        "timings": {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        },
        "info": model.info() if hasattr(model, "info") else {},
        "created_at": _utc_now(),
    }


def _environment_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if importlib.util.find_spec("torch") is not None:
        import torch

        summary["torch"] = torch.__version__
        summary["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            summary["gpu"] = torch.cuda.get_device_name(0)
    if importlib.util.find_spec("numpy") is not None:
        import numpy

        summary["numpy"] = numpy.__version__
    summary["download_allowed"] = os.environ.get(DOWNLOAD_ENV) == "1"
    return summary


def _manifest(jobs_path: str, total: int) -> dict[str, Any]:
    now = _utc_now()
    return {
        "runner_version": RUNNER_VERSION,
        "jobs_file": str(jobs_path),
        "environment": _environment_summary(),
        "started_at": now,
        "updated_at": now,
        "totals": {"jobs": total, "ran": 0, "cached": 0, "failed": 0},
        "results": {},
    }


def _load_jobs(jobs_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("jobs")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{jobs_path} must contain a non-empty 'jobs' list")
    return payload


def cmd_run(
    jobs_path: str,
    out_dir: str,
    force: bool = False,
    limit: int | None = None,
    fail_fast: bool = False,
) -> int:
    jobs_file = Path(jobs_path)
    jobs = _load_jobs(jobs_file)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "jobs.json", {"jobs": jobs})
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["jobs_file"] = str(jobs_file)
        manifest["totals"]["jobs"] = len(jobs)
    else:
        manifest = _manifest(str(jobs_file), len(jobs))
    manifest["updated_at"] = _utc_now()

    executed = 0
    exit_code = 0
    for index, job in enumerate(jobs):
        if limit is not None and executed >= limit:
            break
        try:
            series = load_series(job)
            key = cache_key(job, series)
            result_path = out / f"{key}.json"
            if result_path.exists() and not force:
                manifest["totals"]["cached"] += 1
                manifest["results"][key] = {
                    "job_index": index,
                    "model": job.get("model") or job.get("entry"),
                    "status": "cached",
                    "path": result_path.name,
                }
                print(f"[cached] job {index} -> {key}", flush=True)
                continue
            result = execute_job(job, series, key)
            write_json_atomic(result_path, result)
            executed += 1
            manifest["totals"]["ran"] += 1
            manifest["results"][key] = {
                "job_index": index,
                "model": result["model"],
                "status": "ok",
                "path": result_path.name,
                "mae": result["metrics"]["mae"],
                "rmse": result["metrics"]["rmse"],
                "fit_seconds": result["timings"]["fit_seconds"],
                "predict_seconds": result["timings"]["predict_seconds"],
            }
            print(
                f"[ran] job {index} model={result['model']} key={key} "
                f"mae={result['metrics']['mae']:.6g}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            executed += 1
            exit_code = 1
            manifest["totals"]["failed"] += 1
            manifest["results"][f"failed-job-{index}"] = {
                "job_index": index,
                "model": job.get("model") or job.get("entry"),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[failed] job {index}: {type(exc).__name__}: {exc}", flush=True)
            if fail_fast:
                break
        finally:
            manifest["updated_at"] = _utc_now()
            write_json_atomic(manifest_path, manifest)

    totals = manifest["totals"]
    print(
        f"done: ran={totals['ran']} cached={totals['cached']} "
        f"failed={totals['failed']} out={out}",
        flush=True,
    )
    return exit_code


def cmd_env_check(require: str = "", require_gpu: bool = False) -> int:
    _add_src_to_path()
    environment = _environment_summary()
    print(f"python: {environment['python']} ({environment['platform']})")
    available: dict[str, bool] = {}
    for package in ("torch", "numpy", "timesfm", "chronos"):
        available[package] = importlib.util.find_spec(package) is not None
        print(f"{package}: {'available' if available[package] else 'missing'}")
    if environment.get("torch"):
        print(f"torch: {environment['torch']}")
        if environment.get("cuda_available"):
            print(f"cuda: yes ({environment.get('gpu', 'unknown device')})")
        else:
            print("cuda: no (CPU only)")
    env_gate = f"{DOWNLOAD_ENV}={'1' if environment['download_allowed'] else '0'}"
    print(f"download gate: {env_gate}")

    missing = [name.strip() for name in require.split(",") if name.strip()]
    missing = [name for name in missing if not available.get(name, False)]
    failure = False
    if missing:
        print(f"FAIL: required packages missing: {', '.join(missing)}")
        failure = True
    if require_gpu and not environment.get("cuda_available"):
        print("FAIL: --require-gpu set but CUDA is not available")
        failure = True
    print("env-check: " + ("FAILED" if failure else "OK"))
    return 1 if failure else 0


def cmd_bundle(out_dir: str, dest: str | None = None) -> int:
    out = Path(out_dir)
    if not out.is_dir():
        print(f"bundle: {out} is not a directory", file=sys.stderr)
        return 2
    destination = Path(dest) if dest else out.with_suffix(".tar.gz")
    members = [path for path in sorted(out.rglob("*")) if path.is_file()]
    if not members:
        print(f"bundle: {out} has no files", file=sys.stderr)
        return 2
    with tarfile.open(destination, "w:gz") as archive:
        for path in members:
            if path.name.endswith(".tmp"):
                continue
            archive.add(path, arcname=str(Path(out.name) / path.relative_to(out)))
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"bundle: {destination} sha256={digest}")
    print(f"download this file back to the D-drive clone, then extract into {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Colab GPU runner helper")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    env_parser = subparsers.add_parser("env-check", help="report env/GPU readiness")
    env_parser.add_argument("--require", default="", help="comma-separated packages")
    env_parser.add_argument("--require-gpu", action="store_true")

    run_parser = subparsers.add_parser("run", help="run a JSON job file")
    run_parser.add_argument("--jobs", required=True)
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--force", action="store_true", help="re-run cached jobs")
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--fail-fast", action="store_true")

    bundle_parser = subparsers.add_parser("bundle", help="archive a run directory")
    bundle_parser.add_argument("--out", required=True)
    bundle_parser.add_argument("--dest", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "env-check":
        return cmd_env_check(args.require, args.require_gpu)
    if args.cmd == "run":
        return cmd_run(args.jobs, args.out, args.force, args.limit, args.fail_fast)
    if args.cmd == "bundle":
        return cmd_bundle(args.out, args.dest)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
