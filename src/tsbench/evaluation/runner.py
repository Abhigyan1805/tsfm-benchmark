"""Config-driven experiment runner.

``run_experiment(config_path)`` is the entry point the CLI imports. It:

1. loads the experiment YAML and resolves the dataset from
   ``configs/datasets.yaml``;
2. loads the frozen split manifest and verifies it against the on-disk series
   (a mismatch is a hard failure, never a warning);
3. enumerates leakage-checked rolling-origin windows;
4. assembles and runs the requested models; and
5. writes ``results/<run_id>/`` rows in the exact ``RESULT_COLUMNS`` schema,
   including ``train_seconds`` (empty for untrained families).

Model construction goes through the registry when it is importable, and falls
back to a config-declared local stub otherwise, so the smoke path can run on a
branch where the foundation and model slices have not landed yet.
"""

from __future__ import annotations

import importlib
import inspect
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..data.catalog import (
    DEFAULT_DATASETS_CONFIG,
    DatasetSpec,
    load_catalog,
    resolve_member_path,
)
from ..data.loaders import DataError, load_local_csv, load_monash_tsf
from ..data.splits import (
    SplitConfig,
    SplitError,
    compute_split_plan,
    load_manifest,
    manifest_digest,
    verify_manifest,
)
from .backtest import WindowForecast, backtest_windows
from .contract import RESULT_COLUMNS
from .results import RunContext, result_row, write_results

__all__ = [
    "ExperimentConfig",
    "RunnerError",
    "load_experiment",
    "run_experiment",
]

DEFAULT_RESULTS_DIR = Path("results")
RESULTS_DIR_ENV = "TSBENCH_RESULTS_DIR"


class RunnerError(RuntimeError):
    """An experiment could not be executed."""


@dataclass
class ExperimentConfig:
    """Normalised experiment description."""

    name: str
    raw: dict[str, Any]
    dataset: dict[str, Any]
    models: list[str]
    split: SplitConfig
    output_dir: Path
    seed: int | None = None
    season_length: int = 1
    track_memory: bool = True
    warmup: bool = False
    datasets_config: Path = DEFAULT_DATASETS_CONFIG
    split_manifest: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str = "<config>") -> ExperimentConfig:
        if not isinstance(raw, Mapping):
            raise RunnerError(f"{source}: expected a mapping at the document root")
        name = str(raw.get("name") or Path(source).stem)
        dataset = raw.get("dataset") or {}
        if not isinstance(dataset, Mapping):
            raise RunnerError(f"{source}: 'dataset' must be a mapping")
        models = raw.get("models") or []
        if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
            raise RunnerError(f"{source}: 'models' must be a list")
        split = SplitConfig.from_mapping(raw.get("split"))
        seed = raw.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise RunnerError(f"{source}: 'seed' must be an integer, got {seed!r}")
        track_memory = raw.get("track_memory", True)
        if not isinstance(track_memory, bool):
            raise RunnerError(
                f"{source}: 'track_memory' must be a boolean, got {track_memory!r}"
            )
        warmup = raw.get("warmup", False)
        if not isinstance(warmup, bool):
            raise RunnerError(
                f"{source}: 'warmup' must be a boolean, got {warmup!r}"
            )
        override = os.environ.get(RESULTS_DIR_ENV)
        output_dir = Path(override or raw.get("output_dir") or DEFAULT_RESULTS_DIR)
        manifest = raw.get("split_manifest") or dataset.get("split_manifest")
        return cls(
            name=name,
            raw=dict(raw),
            dataset=dict(dataset),
            models=[str(m) for m in models],
            split=split,
            output_dir=output_dir,
            seed=seed,
            season_length=int(raw.get("season_length", 1)),
            track_memory=track_memory,
            warmup=warmup,
            datasets_config=Path(raw.get("datasets_config", DEFAULT_DATASETS_CONFIG)),
            split_manifest=Path(manifest) if manifest else None,
            metadata={k: v for k, v in raw.items() if k not in {"dataset", "models"}},
        )


def load_experiment(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise RunnerError(f"experiment config not found: {config_path}")
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RunnerError(f"{config_path}: invalid YAML: {exc}") from exc
    try:
        return ExperimentConfig.from_mapping(document or {}, source=str(config_path))
    except RunnerError:
        raise
    except (SplitError, ValueError, TypeError) as exc:
        raise RunnerError(f"{config_path}: {exc}") from exc


def _load_series(
    spec: DatasetSpec,
    dataset_cfg: Mapping[str, Any],
    *,
    root: Path,
) -> list[Any]:
    """Load the dataset's series using the catalogue spec as the default."""
    loader = dataset_cfg.get("loader") or spec.loader
    if loader == "local_csv":
        path = dataset_cfg.get("path") or spec.path
        if not path:
            raise RunnerError(f"dataset {spec.name!r}: local_csv needs a 'path'")
        series_id = dataset_cfg.get("series_id") or spec.extra.get("series_id")
        one = load_local_csv(
            root / path,
            series_id=series_id,
            timestamp_column=dataset_cfg.get("timestamp_column", "timestamp"),
            value_column=dataset_cfg.get("value_column", "value"),
        )
        return [one]
    if loader == "monash_tsf":
        path = resolve_member_path(spec, root=root)
        if path is None or not path.is_file():
            raise RunnerError(
                f"dataset {spec.name!r}: {path} is absent; run "
                f"`python -m tsbench.data.build_catalog --materialize` first"
            )
        if dataset_cfg.get("series_ids"):
            return load_monash_tsf(path, series_ids=list(dataset_cfg["series_ids"]))
        max_series = dataset_cfg.get("max_series")
        return load_monash_tsf(path, max_series=int(max_series) if max_series else None)
    raise RunnerError(f"dataset {spec.name!r}: unknown loader {loader!r}")


def _build_model(name: str, spec: Any, *, model_cfg: Mapping[str, Any], seed: int | None) -> Any:
    """Instantiate a model, routing registry-known names through the license gate.

    When the registry resolves ``name`` and its entrypoint imports, construction
    goes through ``ModelRegistry.instantiate`` so the gate runs against the
    entrypoint actually built. A config-declared ``entrypoint`` is a stub
    fallback for branches where the registered module has not landed yet, so it
    is used only when the registry's own entrypoint cannot be imported; that
    fallback still calls ``check_license`` first. Models the registry does not
    know may declare their own entrypoint; that path imports the callable
    directly and is therefore outside the registry's license gate (the TSFM
    wrappers' own checkpoint-id tables still apply).
    """
    kwargs = dict(model_cfg.get("kwargs") or {})
    entrypoint = model_cfg.get("entrypoint")
    registry = _try_registry()
    if registry is not None and name in registry:
        registry_spec = registry.spec(name)
        try:
            target = _entrypoint_target(name, registry_spec.entrypoint)
        except RunnerError as exc:
            if not entrypoint:
                raise RunnerError(
                    f"model {name!r}: registry entrypoint is unavailable: {exc}"
                ) from exc
            from ..registry import check_license

            check_license(registry_spec)
        else:
            return registry.instantiate(
                name, **_seed_kwargs(kwargs, target, model_cfg=model_cfg, seed=seed)
            )
    if entrypoint:
        target = _entrypoint_target(name, entrypoint)
        seeded = _seed_kwargs(kwargs, target, model_cfg=model_cfg, seed=seed)
        return target(**seeded) if seeded else target()
    raise RunnerError(
        f"model {name!r}: no entrypoint in the experiment config and no registry "
        "entry available"
    )


def _entrypoint_target(name: str, entrypoint: Any) -> Any:
    """Resolve ``module:attribute`` to a callable, or raise ``RunnerError``."""
    module_name, _, attribute = str(entrypoint).partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RunnerError(f"model {name!r}: cannot import {module_name!r}: {exc}") from exc
    target = getattr(module, attribute, None)
    if target is None:
        raise RunnerError(f"model {name!r}: {entrypoint!r} has no {attribute!r}")
    return target


def _seed_kwargs(
    kwargs: Mapping[str, Any],
    target: Any,
    *,
    model_cfg: Mapping[str, Any],
    seed: int | None,
) -> dict[str, Any]:
    """Forward the run seed only when the target accepts it.

    Deterministic baselines must not receive an unsupported ``seed`` kwarg. The
    run seed is added only when the constructor signature accepts it or when
    ``model_configs.<model>.seed`` explicitly opts in; an explicit ``false``
    always suppresses it.
    """
    resolved = dict(kwargs)
    if seed is None or resolved.get("seed") is not None:
        return resolved
    requested = model_cfg.get("seed")
    if requested is False:
        return resolved
    if requested is True or _accepts_kwarg(target, "seed"):
        resolved.setdefault("seed", seed)
    return resolved


def _accepts_kwarg(target: Any, name: str) -> bool:
    """True when ``target`` can be called with the keyword ``name``."""
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False
    if name in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _try_registry() -> Any | None:
    try:
        module = importlib.import_module("tsbench.registry")
    except ImportError:
        return None
    loader = getattr(module, "load_registry", None)
    if not callable(loader):
        return None
    try:
        return loader()
    except Exception:
        return None


def _rows_for_model(
    forecast_rows: Sequence[WindowForecast],
    *,
    context: RunContext,
    dataset: str,
    window_offset: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outcome in forecast_rows:
        rows.append(
            result_row(
                run_id=context.run_id,
                dataset=dataset,
                series_id=outcome.series_id,
                model=outcome.model,
                family=outcome.family,
                context_length=outcome.context_length,
                horizon=outcome.horizon,
                window_index=window_offset + outcome.window_index,
                mae=outcome.metrics.get("mae", float("nan")),
                rmse=outcome.metrics.get("rmse", float("nan")),
                mase=outcome.metrics.get("mase", float("nan")),
                smape=outcome.metrics.get("smape", float("nan")),
                latency_ms=outcome.latency_ms,
                peak_mem_mb=outcome.peak_mem_mb,
                params=outcome.params,
                zero_shot=_row_zero_shot(outcome),
                train_seconds=outcome.train_seconds,
                git_sha=context.git_sha,
                config_hash=context.config_sha,
                measured=context.measured,
            )
        )
    return rows


def _row_zero_shot(outcome: WindowForecast) -> bool | None:
    """Emit the model's own zero-shot flag, or empty when it is unreported.

    ``zero_shot`` marks forecasters with no learned parameters, so reference
    floors and untuned TSFMs report ``True`` while classical, ML, and deep
    families report ``False``.
    """
    if outcome.zero_shot is None:
        return None
    return bool(outcome.zero_shot)


def run_experiment(config_path: str | Path, *, root: str | Path = ".") -> dict[str, Any]:
    """Execute an experiment config and return a run summary."""
    experiment = load_experiment(config_path)
    base = Path(root)
    try:
        specs, _catalog_meta = load_catalog(base / experiment.datasets_config)
        dataset_cfg = dict(experiment.dataset)
        dataset_name = str(dataset_cfg.get("name") or "")
        if not dataset_name:
            if len(specs) == 1:
                dataset_name = next(iter(specs))
            else:
                raise RunnerError(
                    f"{config_path}: dataset.name is required when the catalogue has "
                    f"multiple datasets ({', '.join(specs)})"
                )
        if dataset_name not in specs:
            raise RunnerError(f"{config_path}: unknown dataset {dataset_name!r}")
        spec = specs[dataset_name]

        series = _load_series(spec, dataset_cfg, root=base)

        manifest_path = experiment.split_manifest or (
            Path(spec.split_manifest) if spec.split_manifest else None
        )
        if manifest_path is None:
            raise RunnerError(
                f"{config_path}: no split manifest; set dataset.split_manifest or "
                "split_manifest"
            )
        manifest = load_manifest(manifest_path)
        if manifest.preliminary:
            raise RunnerError(
                f"{manifest_path}: split manifest is marked PRELIMINARY; refusing to "
                "produce results from an unfrozen split"
            )
        problems = verify_manifest(manifest, {s.series_id: s for s in series})
        if problems:
            detail = "; ".join(problems[:5])
            raise RunnerError(f"{manifest_path}: split verification failed: {detail}")

        requested_fracs = (
            experiment.split.train_frac,
            experiment.split.val_frac,
            experiment.split.test_frac,
        )
        manifest_fracs = (
            manifest.split.train_frac,
            manifest.split.val_frac,
            manifest.split.test_frac,
        )
        if requested_fracs != manifest_fracs:
            raise RunnerError(
                f"{config_path}: split fractions {requested_fracs} disagree with the "
                f"frozen manifest {manifest_fracs}; the committed split is binding"
            )
    except (DataError, SplitError, FileNotFoundError) as exc:
        raise RunnerError(str(exc)) from exc

    frozen = {entry["series_id"]: entry for entry in manifest.series}
    ordered = [s for s in series if s.series_id in frozen]
    if not ordered:
        raise RunnerError(
            f"{config_path}: no loaded series match the split manifest "
            f"({manifest.dataset})"
        )

    context = RunContext.create(
        experiment.name,
        experiment.raw,
        split_manifest=manifest_digest(manifest),
        seed=experiment.seed,
        cwd=base,
    )

    all_rows: list[dict[str, Any]] = []
    window_offset = 0
    for s in ordered:
        values = np.asarray(s.values, dtype=np.float64)
        plan = compute_split_plan(values.size, experiment.split)
        for model_name in experiment.models:
            model_cfg = _model_config(experiment, model_name)
            builder = lambda name=model_name, cfg=model_cfg: _build_model(  # noqa: E731
                name, spec, model_cfg=cfg, seed=experiment.seed
            )
            outcomes = backtest_windows(
                values,
                plan,
                builder,
                series_id=s.series_id,
                season_length=experiment.season_length,
                track_memory=experiment.track_memory,
                warmup=experiment.warmup,
            )
            all_rows.extend(
                _rows_for_model(
                    outcomes,
                    context=context,
                    dataset=dataset_name,
                    window_offset=window_offset,
                )
            )
        window_offset += 10_000  # keep window indices unique across series

    csv_path = write_results(all_rows, root=base / experiment.output_dir, context=context)
    return {
        "run_id": context.run_id,
        "results": str(csv_path),
        "n_rows": len(all_rows),
        "dataset": dataset_name,
        "git_sha": context.git_sha,
        "config_hash": context.config_sha,
        "split_manifest": context.split_manifest,
    }


def _model_config(experiment: ExperimentConfig, model_name: str) -> Mapping[str, Any]:
    """Look up per-model overrides in the experiment config."""
    models_cfg = experiment.raw.get("model_configs") or {}
    if isinstance(models_cfg, Mapping) and model_name in models_cfg:
        value = models_cfg[model_name]
        return value if isinstance(value, Mapping) else {}
    return {}


# Re-exported for tests and callers.
__all__ += ["RESULT_COLUMNS", "SplitError"]
