"""Command-line entry point for tsbench.

Experiment runs are config-driven: ``python -m tsbench run --config <path>``.
The runner itself lives in the evaluation slice
(``tsbench.evaluation.runner.run_experiment``); the foundation branch wires
the command and reports a clear error until that slice lands.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .registry import ALLOW_NONCOMMERCIAL_ENV, RegistryError, load_registry

__all__ = ["PipelineUnavailableError", "build_parser", "main"]

DEFAULT_MODELS_CONFIG = Path("configs") / "models.yaml"
RESULTS_DIR_ENV = "TSBENCH_RESULTS_DIR"
EVALUATION_RUNNER = ("tsbench.evaluation.runner", "run_experiment")

_MANIFEST_COLUMNS = ("model", "family", "zero_shot", "params", "license", "revision", "permissive")


class PipelineUnavailableError(RuntimeError):
    """The sibling slice that executes experiments is not importable."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsbench",
        description="Benchmark time-series forecasters on accuracy and inference cost.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run an experiment config")
    run_parser.add_argument("--config", required=True, type=Path, help="experiment YAML")
    run_parser.add_argument(
        "--output", type=Path, default=None, help=f"override ${RESULTS_DIR_ENV}"
    )
    run_parser.add_argument(
        "--allow-noncommercial",
        action="store_true",
        help=f"same as {ALLOW_NONCOMMERCIAL_ENV}=1 for this run",
    )

    licenses_parser = subparsers.add_parser(
        "licenses", help="print the model license manifest"
    )
    licenses_parser.add_argument(
        "--models-config", type=Path, default=DEFAULT_MODELS_CONFIG
    )
    return parser


def _load_runner() -> Any:
    module_name, attribute = EVALUATION_RUNNER
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise PipelineUnavailableError(
            f"cannot execute experiments: {module_name}.{attribute} is unavailable. "
            "The evaluation slice owns that module; this foundation branch ships "
            "packaging, configs, the registry with its license gate, CI, and the "
            "smoke wiring only."
        ) from exc
    runner = getattr(module, attribute, None)
    if not callable(runner):
        raise PipelineUnavailableError(
            f"{module_name} does not expose a callable {attribute}"
        )
    return runner


def _runner_error_type() -> type[BaseException] | None:
    """The evaluation slice's ``RunnerError``, or ``None`` if unavailable."""
    try:
        module = importlib.import_module(EVALUATION_RUNNER[0])
    except ImportError:
        return None
    error = getattr(module, "RunnerError", None)
    if isinstance(error, type) and issubclass(error, BaseException):
        return error
    return None


def _run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"error: experiment config not found: {config_path}", file=sys.stderr)
        return 2
    if args.allow_noncommercial:
        os.environ[ALLOW_NONCOMMERCIAL_ENV] = "1"
    if args.output is not None:
        os.environ[RESULTS_DIR_ENV] = str(args.output)
    runner = _load_runner()
    print(f"running experiment config: {config_path}", file=sys.stderr)
    try:
        result = runner(config_path)
    except Exception as exc:
        runner_error = _runner_error_type()
        if runner_error is None or not isinstance(exc, runner_error):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if result is not None:
        print(result)
    return 0


def _render_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _format_manifest(rows: list[dict[str, Any]]) -> str:
    rendered = [
        {column: _render_cell(row.get(column)) for column in _MANIFEST_COLUMNS}
        for row in rows
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in rendered))
        for column in _MANIFEST_COLUMNS
    }
    lines = [
        "  ".join(column.ljust(widths[column]) for column in _MANIFEST_COLUMNS),
        "  ".join("-" * widths[column] for column in _MANIFEST_COLUMNS),
    ]
    lines.extend(
        "  ".join(row[column].ljust(widths[column]) for column in _MANIFEST_COLUMNS)
        for row in rendered
    )
    return "\n".join(lines)


def _licenses(args: argparse.Namespace) -> int:
    registry = load_registry(args.models_config)
    print(_format_manifest(registry.manifest()))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "licenses":
            return _licenses(args)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except PipelineUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
