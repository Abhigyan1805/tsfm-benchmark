"""Frozen split manifests and leakage-audited window enumeration.

A split is a pure function of a series' length and a small frozen config:

* ``train_end`` / ``val_end`` / ``test_end`` are **exclusive** indices into the
  series, so ``values[:train_end]`` is the training slice and
  ``values[train_end:val_end]`` is validation.
* A **backtest window** is identified by the index of its first forecast step
  (the origin). The lookback (``context_length`` values) ends at the origin and
  the target (``horizon`` values) begins at the origin, so no window ever reads
  a value at or after its target block.

The manifest stores a digest for every boundary value, which makes a frozen
split tamper-evident: if the underlying series changes, the recorded hashes stop
matching. Windows are enumerated on demand and never stored, so the manifest
stays small while remaining fully auditable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "LeakageError",
    "SplitConfig",
    "SplitError",
    "SplitManifest",
    "SplitPlan",
    "build_manifest",
    "compute_split_plan",
    "enumerate_windows",
    "iter_windows",
    "load_manifest",
    "manifest_digest",
    "save_manifest",
    "value_sha256",
    "verify_manifest",
    "window_digest",
]

MANIFEST_VERSION = 1


class SplitError(ValueError):
    """A split configuration is inconsistent or a series is too short."""


class LeakageError(AssertionError):
    """An enumerated window violates the no-future-data contract."""


def value_sha256(values: np.ndarray) -> str:
    """Hex sha256 of a float64 value block (stable across platforms)."""
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SplitConfig:
    """Frozen proportions and backtest geometry for one dataset."""

    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2
    context_length: int = 168
    horizon: int = 24
    stride: int = 24
    min_train_size: int = 1

    def __post_init__(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-9:
            raise SplitError(f"split fractions must sum to 1.0, got {total}")
        if min(self.train_frac, self.val_frac, self.test_frac) < 0:
            raise SplitError("split fractions must be non-negative")
        for name in ("context_length", "horizon", "stride", "min_train_size"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise SplitError(f"{name} must be a positive integer, got {value!r}")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> SplitConfig:
        if not raw:
            return cls()
        allowed = {
            "train_frac",
            "val_frac",
            "test_frac",
            "context_length",
            "horizon",
            "stride",
            "min_train_size",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise SplitError(f"unknown split field(s): {', '.join(unknown)}")
        return cls(**dict(raw))

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_frac": self.train_frac,
            "val_frac": self.val_frac,
            "test_frac": self.test_frac,
            "context_length": self.context_length,
            "horizon": self.horizon,
            "stride": self.stride,
            "min_train_size": self.min_train_size,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class SplitPlan:
    """Resolved, index-exact split boundaries for one series."""

    n_observations: int
    train_end: int
    val_end: int
    test_end: int
    config: SplitConfig

    @property
    def train_range(self) -> tuple[int, int]:
        return (0, self.train_end)

    @property
    def val_range(self) -> tuple[int, int]:
        return (self.train_end, self.val_end)

    @property
    def test_range(self) -> tuple[int, int]:
        return (self.val_end, self.test_end)

    def period_of(self, index: int) -> str:
        if index < 0 or index >= self.test_end:
            raise SplitError(f"index {index} outside [0, {self.test_end})")
        if index < self.train_end:
            return "train"
        if index < self.val_end:
            return "val"
        return "test"

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "train_end": self.train_end,
            "val_end": self.val_end,
            "test_end": self.test_end,
            "config": self.config.as_dict(),
        }


def compute_split_plan(n_observations: int, config: SplitConfig) -> SplitPlan:
    """Resolve fractional boundaries to exact indices via ``floor``.

    ``floor`` (rather than ``round``) is deterministic and never over-allocates
    a period, so the fractional shares recorded in the manifest correspond
    exactly to the committed boundaries.
    """
    if n_observations < 1:
        raise SplitError("a series must have at least one observation")
    train_span = int(np.floor(n_observations * config.train_frac))
    val_span = int(np.floor(n_observations * config.val_frac))
    train_end = train_span
    val_end = train_end + val_span
    test_end = n_observations
    if train_end < config.min_train_size:
        raise SplitError(
            f"series of length {n_observations} yields a {train_end}-point training "
            f"slice, below min_train_size={config.min_train_size}"
        )
    if train_end >= val_end or val_end >= test_end:
        raise SplitError(
            f"series of length {n_observations} is too short for split "
            f"{config.as_dict()}"
        )
    return SplitPlan(
        n_observations=n_observations,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        config=config,
    )


def enumerate_windows(
    plan: SplitPlan,
    *,
    period: str = "test",
    start_index: int | None = None,
) -> list[int]:
    """Return window origins (first forecast-step indices) in ``period``.

    A window origin must leave a full lookback behind it and a full horizon in
    front of it *inside the requested period*. When ``start_index`` is given,
    the first origin is ``max(start_index, period_start + context_length)``.
    """
    if period not in {"train", "val", "test"}:
        raise SplitError(f"unknown period {period!r}")
    lo, hi = {"train": plan.train_range, "val": plan.val_range, "test": plan.test_range}[
        period
    ]
    first = lo + plan.config.context_length
    if start_index is not None:
        first = max(first, int(start_index))
    last = hi - plan.config.horizon
    origins: list[int] = []
    origin = first
    while origin <= last:
        origins.append(origin)
        origin += plan.config.stride
    return origins


def _resolve_periods(
    plan: SplitPlan,
    *,
    period: str | Sequence[str] | None,
) -> list[str]:
    if period is None:
        return ["test"]
    if isinstance(period, str):
        periods = [period]
    else:
        periods = list(period)
    bad = [p for p in periods if p not in {"train", "val", "test"}]
    if bad:
        raise SplitError(f"unknown period(s): {', '.join(map(str, bad))}")
    return periods


def iter_windows(
    values: np.ndarray,
    plan: SplitPlan,
    *,
    period: str | Sequence[str] | None = None,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return ``(origin, context, target)`` for each window, leakage-checked.

    The contexts are *views* into ``values``: ``values[origin-context : origin]``
    and ``values[origin : origin+horizon]``. Copies are the caller's concern.
    """
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if series.size != plan.n_observations:
        raise SplitError(
            f"plan covers {plan.n_observations} observations but series has {series.size}"
        )
    context_length = plan.config.context_length
    horizon = plan.config.horizon
    windows: list[tuple[int, np.ndarray, np.ndarray]] = []
    for named in _resolve_periods(plan, period=period):
        for origin in enumerate_windows(plan, period=named):
            context = series[origin - context_length : origin]
            target = series[origin : origin + horizon]
            assert_no_future_data(plan, origin, context_length, horizon)
            windows.append((origin, context, target))
    windows.sort(key=lambda item: item[0])
    return windows


def assert_no_future_data(
    plan: SplitPlan, origin: int, context_length: int, horizon: int
) -> None:
    """Structural guard: context and target must not overlap.

    Also asserts the lookback stays inside the series. The harder question --
    whether *scaling* or *features* peeked -- is answered by the dedicated
    leakage tests that drive the leakers.
    """
    if context_length < 1 or horizon < 1:
        raise LeakageError("context_length and horizon must be positive")
    context_start = origin - context_length
    target_end = origin + horizon
    if context_start < 0:
        raise LeakageError(
            f"origin {origin} has only {origin} values behind it; need {context_length}"
        )
    if target_end > plan.n_observations:
        raise LeakageError(
            f"origin {origin} target ends at {target_end} beyond series length "
            f"{plan.n_observations}"
        )
    if context_start >= origin or target_end <= origin:
        raise LeakageError(f"origin {origin} produces an empty context or target")
    # The context slice is [context_start, origin) and the target slice is
    # [origin, target_end): they share only the boundary, never an index.
    if target_end - origin != horizon or origin - context_start != context_length:
        raise LeakageError("window slices overlap")


def window_digest(context: np.ndarray, target: np.ndarray) -> str:
    """Digest binding one window's context and target values."""
    digest = hashlib.sha256()
    digest.update(b"tsbench-window-v1")
    digest.update(value_sha256(context).encode("ascii"))
    digest.update(value_sha256(target).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class SplitManifest:
    """Tamper-evident record of a frozen split."""

    dataset: str
    loader: str
    source: str | None
    license: str | None
    split: SplitConfig
    series: list[dict[str, Any]]
    preliminary: bool = False
    version: int = MANIFEST_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset": self.dataset,
            "loader": self.loader,
            "source": self.source,
            "license": self.license,
            "preliminary": self.preliminary,
            "split": self.split.as_dict(),
            "series": self.series,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def binding_digest(self) -> str:
        """Digest over the frozen parts, suitable for a results row."""
        return manifest_digest(self)


def manifest_digest(manifest: SplitManifest) -> str:
    payload = manifest.to_dict()
    payload.pop("metadata", None)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def save_manifest(manifest: SplitManifest, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return destination


_REQUIRED_SERIES_FIELDS = (
    "series_id",
    "n_observations",
    "train_end",
    "val_end",
    "test_end",
    "value_sha256",
)


def load_manifest(path: str | Path) -> SplitManifest:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"split manifest not found: {source}")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SplitError(f"{source}: invalid JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise SplitError(f"{source}: expected an object at the root")
    if int(document.get("version", 0)) != MANIFEST_VERSION:
        raise SplitError(
            f"{source}: unsupported manifest version {document.get('version')!r}"
        )
    series = document.get("series")
    if not isinstance(series, list) or not series:
        raise SplitError(f"{source}: manifest has no series")
    for entry in series:
        if not isinstance(entry, Mapping):
            raise SplitError(f"{source}: series entry must be an object")
        missing = [f for f in _REQUIRED_SERIES_FIELDS if f not in entry]
        if missing:
            raise SplitError(f"{source}: series entry missing {', '.join(missing)}")
    return SplitManifest(
        dataset=str(document["dataset"]),
        loader=str(document.get("loader", "")),
        source=document.get("source"),
        license=document.get("license"),
        split=SplitConfig.from_mapping(document.get("split")),
        series=[dict(entry) for entry in series],
        preliminary=bool(document.get("preliminary", False)),
        version=int(document.get("version", MANIFEST_VERSION)),
        metadata=dict(document.get("metadata", {})),
    )


def build_manifest(
    dataset: str,
    series: Iterable[Any],
    split: SplitConfig,
    *,
    loader: str = "monash_tsf",
    source: str | None = None,
    license: str | None = None,
    preliminary: bool = False,
    metadata: dict[str, Any] | None = None,
) -> SplitManifest:
    """Build a manifest from series-like objects (``.series_id`` / ``.values``)."""
    entries: list[dict[str, Any]] = []
    for s in series:
        values = np.asarray(s.values, dtype=np.float64).reshape(-1)
        plan = compute_split_plan(values.size, split)
        entry: dict[str, Any] = {
            "series_id": str(s.series_id),
            "n_observations": int(values.size),
            "train_end": plan.train_end,
            "val_end": plan.val_end,
            "test_end": plan.test_end,
            "value_sha256": value_sha256(values),
            "boundary": {
                "train_end": _boundary_record(values, plan.train_end),
                "val_end": _boundary_record(values, plan.val_end),
                "test_end": _boundary_record(values, plan.test_end),
            },
        }
        entries.append(entry)
    if not entries:
        raise SplitError("cannot build a manifest with no series")
    return SplitManifest(
        dataset=dataset,
        loader=loader,
        source=source,
        license=license,
        split=split,
        series=entries,
        preliminary=preliminary,
        metadata=dict(metadata or {}),
    )


def _boundary_record(values: np.ndarray, end: int) -> dict[str, Any]:
    """Hash the last few values before an exclusive boundary."""
    lookback = max(0, end - 3)
    block = values[lookback:end]
    return {
        "end": int(end),
        "n_values": int(block.size),
        "sha256": value_sha256(block) if block.size else None,
    }


def verify_manifest(
    manifest: SplitManifest,
    series_by_id: Mapping[str, Any],
) -> list[str]:
    """Return a list of human-readable mismatches (empty means the split froze)."""
    problems: list[str] = []
    for entry in manifest.series:
        sid = entry["series_id"]
        if sid not in series_by_id:
            problems.append(f"{sid}: missing from loaded series")
            continue
        values = np.asarray(series_by_id[sid].values, dtype=np.float64).reshape(-1)
        if values.size != entry["n_observations"]:
            problems.append(
                f"{sid}: length {values.size} != manifest {entry['n_observations']}"
            )
            continue
        actual = value_sha256(values)
        if actual != entry["value_sha256"]:
            problems.append(f"{sid}: value digest changed")
        plan = compute_split_plan(values.size, manifest.split)
        for name in ("train_end", "val_end", "test_end"):
            if getattr(plan, name) != entry[name]:
                problems.append(
                    f"{sid}: {name} {getattr(plan, name)} != manifest {entry[name]}"
                )
    return problems
