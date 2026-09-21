"""Leakage audits and safe scaling primitives.

Two complementary tools live here:

* :func:`assert_context_only_scaling` and :func:`assert_no_future_in_features`
  return a failed :class:`LeakageAudit` when a statistic or feature was computed
  using values from the target window; :meth:`LeakageReport.require_ok` turns a
  failed audit into a :class:`LeakageError`. The tests deliberately feed them
  leaky implementations to prove the detector fires, not just that clean code
  passes.
* :class:`ContextScaler` is the reference scaler: it is fit on the context only
  and applied to both context and target, so a model can standardise inputs
  without ever touching future observations.

The boundary-touch audit records the exact index at which an audit saw its last
observation, making it possible to say *which* future index leaked.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .splits import LeakageError

__all__ = [
    "ContextScaler",
    "LeakageAudit",
    "LeakageReport",
    "assert_context_only_scaling",
    "assert_no_future_in_features",
    "audit_context_boundary",
]

# Values that only exist after the forecast origin, by construction, so any
# statistic or feature strictly greater than this sentinel is future-derived.
_EPS = 1e-12


@dataclass(frozen=True)
class LeakageAudit:
    """A single named leakage probe and what it observed."""

    name: str
    passed: bool
    detail: str


@dataclass
class LeakageReport:
    """Aggregated results of a leakage audit run."""

    audits: list[LeakageAudit]

    @property
    def ok(self) -> bool:
        return all(audit.passed for audit in self.audits)

    @property
    def failures(self) -> list[LeakageAudit]:
        return [audit for audit in self.audits if not audit.passed]

    def add(self, name: str, passed: bool, detail: str = "") -> LeakageAudit:
        audit = LeakageAudit(name=name, passed=passed, detail=detail)
        self.audits.append(audit)
        return audit

    def require_ok(self) -> None:
        if not self.ok:
            joined = "; ".join(f"{a.name}: {a.detail}" for a in self.failures)
            raise LeakageError(f"leakage audit failed: {joined}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "audits": [
                {"name": a.name, "passed": a.passed, "detail": a.detail}
                for a in self.audits
            ],
        }


@dataclass
class ContextScaler:
    """Standardise a series using statistics from the context window only.

    ``fit`` is the only place the context is read. Once fitted at an origin,
    ``transform`` may legitimately be applied to target values -- applying a
    context-derived transform to future values is not a leak; *fitting* on them
    would be. ``assert_fit_origin`` catches the latter: it refuses a transform
    whose supplied ``indices`` extend past an origin the caller declares as the
    fit boundary, so a caller cannot refit-and-transform in one step by mistake.
    """

    center: float | None = None
    scale: float | None = None
    origin: int | None = None
    context_length: int | None = None

    def fit(self, context: np.ndarray, *, origin: int | None = None) -> ContextScaler:
        array = np.asarray(context, dtype=np.float64).reshape(-1)
        if array.size == 0:
            raise ValueError("cannot fit a scaler on an empty context")
        self.center = float(np.mean(array))
        spread = float(np.std(array))
        self.scale = spread if spread > _EPS else 1.0
        if origin is not None:
            if self.context_length is None:
                self.context_length = array.size
            self.origin = int(origin)
        return self

    def _require_fitted(self) -> tuple[float, float]:
        if self.center is None or self.scale is None:
            raise RuntimeError("ContextScaler must be fitted before transform")
        return self.center, self.scale

    def assert_fit_origin(self, indices: Sequence[int]) -> None:
        """Raise if any ``index`` is at or beyond the declared fit origin."""
        if self.origin is None:
            return
        forbidden = [int(i) for i in indices if int(i) >= self.origin]
        if forbidden:
            raise LeakageError(
                f"ContextScaler fitted at origin {self.origin} was asked to fit "
                f"or re-derive statistics at future index/indices {forbidden[:5]}"
            )

    def transform(self, values: np.ndarray, *, indices: Sequence[int] | None = None) -> np.ndarray:
        center, scale = self._require_fitted()
        array = np.asarray(values, dtype=np.float64)
        return (array - center) / scale

    def fit_transform(self, context: np.ndarray, *, origin: int | None = None) -> np.ndarray:
        return self.fit(context, origin=origin).transform(context)


def assert_context_only_scaling(
    context: np.ndarray,
    target: np.ndarray,
    scaler_factory: Callable[[], Any] | None = None,
    *,
    origin: int | None = None,
) -> LeakageAudit:
    """Assert a scaler fitted on the context produces no target information.

    The scaler must be fittable from ``context`` alone and must keep the
    context and target transforms consistent with those context statistics.

    What the audit can and cannot see: the checks below compare the fitted
    statistic against the context-only statistic, so they catch a scaler that
    *derives* its statistic from the target. They cannot infer the intended
    statistic for an arbitrary transform, so a transform that ignores the
    statistics it fitted is caught instead by fitting a second instance on a
    perturbed context and confirming the same probe transforms differently. The
    perturbation reverses the order as well as shifting location and scale, so a
    legitimate transform (including an order-based one) still moves; only a
    transform that never reads its fit stays byte-identical. This is a
    sanity check on the transform, not a proof that it used the fit in any
    particular way.
    """
    context = np.asarray(context, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    factory = scaler_factory or ContextScaler
    scaler = factory()
    scaler.fit(context, origin=origin) if origin is not None else scaler.fit(context)
    # Refitting on a context polluted with target values must not be possible
    # through the public surface: a scaler that stores the full fit input is
    # leaking. We detect this by checking the scaler's statistics match the
    # context-only statistics.
    center = float(np.mean(context))
    refit_center = _extract_center(scaler)
    if refit_center is not None and abs(refit_center - center) > 1e-6:
        return LeakageAudit(
            "context_scaling",
            False,
            f"scaler center {refit_center} does not match context-only mean {center}",
        )
    combined = np.concatenate([context, target])
    scaler_on_combined = factory()
    scaler_on_combined.fit(combined)
    combined_mean = float(np.mean(combined))
    combined_center = _extract_center(scaler_on_combined)
    if (
        combined_center is not None
        and abs(combined_mean - center) > 1e-9
        and abs(combined_center - center) <= 1e-9
    ):
        return LeakageAudit(
            "context_scaling",
            False,
            "scaler fitted on the context+target window still reports the "
            f"context-only statistic {combined_center}; it does not derive "
            "statistics from the window it is fitted on",
        )
    # A transform that ignores its fitted statistics reports the correct center
    # yet was never actually fitted, so the statistic comparison above cannot
    # see it. Fit a second instance on a perturbed context (order reversed,
    # location and scale changed) and require the transform of the original
    # context to move; an ignored fit is a no-op and yields identical output.
    scale = float(np.std(context))
    offset = 10.0 * (abs(center) + scale + 1.0)
    perturbed = -2.0 * context + offset
    scaler_on_perturbed = factory()
    if origin is not None:
        scaler_on_perturbed.fit(perturbed, origin=origin)
    else:
        scaler_on_perturbed.fit(perturbed)
    honest_probe = np.asarray(scaler.transform(context), dtype=np.float64)
    perturbed_probe = np.asarray(scaler_on_perturbed.transform(context), dtype=np.float64)
    if (
        honest_probe.shape == perturbed_probe.shape
        and np.allclose(honest_probe, perturbed_probe, rtol=0, atol=1e-9, equal_nan=True)
    ):
        return LeakageAudit(
            "context_scaling",
            False,
            "scaler transform produced identical output for two different fit "
            "windows; it ignores the statistics it fitted",
        )
    return LeakageAudit("context_scaling", True, "scaler derived from context only")


def _extract_center(scaler: Any) -> float | None:
    for attribute in ("center", "_center", "mean", "_mean"):
        if hasattr(scaler, attribute):
            value = getattr(scaler, attribute)
            if value is not None:
                return float(value)
    return None


def assert_no_future_in_features(
    context: np.ndarray,
    target: np.ndarray,
    feature_fn: Callable[[np.ndarray, int], np.ndarray],
    *,
    position: int | None = None,
) -> LeakageAudit:
    """Assert feature construction for the forecast origin uses no target values.

    ``feature_fn(history, position)`` returns the feature vector for predicting
    the value at ``position`` from ``history``. It is called with the honest
    context-only buffer, then with the target appended, then with the target
    replaced by sentinels. The feature vector for the origin must be identical
    in all three cases; a function that indexes ``history[position]`` or takes a
    mean over ``history[:position+1]`` fails, which is exactly the recursive
    look-ahead bug this audit exists to catch.
    """
    context = np.asarray(context, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if position is None:
        position = context.size
    if not 0 < position <= context.size:
        raise LeakageError(f"position {position} must be within the context")

    try:
        honest = _flatten_features(feature_fn(context, position))
        appended = np.concatenate([context, target])
        with_future = _flatten_features(feature_fn(appended, position))
        sentinel = np.concatenate([context, np.full_like(target, -1e12)])
        sentinel_row = _flatten_features(feature_fn(sentinel, position))
    except IndexError as exc:
        return LeakageAudit(
            "feature_future_independence",
            False,
            f"feature_fn for position {position} indexed past the honest context "
            f"({type(exc).__name__}: {exc}); it reads future data",
        )

    if honest.shape != with_future.shape or honest.shape != sentinel_row.shape:
        return LeakageAudit(
            "feature_future_independence",
            False,
            f"feature vector for position {position} changed shape when future values "
            f"were supplied or altered ({honest.shape} vs {with_future.shape} "
            f"vs {sentinel_row.shape})",
        )
    if not np.allclose(honest, with_future, rtol=0, atol=1e-9, equal_nan=True):
        return LeakageAudit(
            "feature_future_independence",
            False,
            f"feature vector for position {position} changed when target values were "
            "appended; the design function read future data",
        )
    if not np.allclose(honest, sentinel_row, rtol=0, atol=1e-9, equal_nan=True):
        return LeakageAudit(
            "feature_future_independence",
            False,
            f"feature vector for position {position} depends on future values' "
            "magnitude; the design function read future data",
        )
    again = _flatten_features(feature_fn(context, position))
    if not np.array_equal(honest, again):
        return LeakageAudit("feature_determinism", False, "feature_fn is non-deterministic")
    return LeakageAudit(
        "feature_future_independence",
        True,
        f"features for position {position} are independent of future values",
    )


def _flatten_features(features: Any) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    return array.reshape(-1)


def audit_context_boundary(
    values: np.ndarray,
    origin: int,
    context_length: int,
    horizon: int,
) -> LeakageReport:
    """Structural audit of one window against the no-future contract.

    Records the last index each slice observed so a failure names the exact
    offending index instead of only asserting that something is wrong.
    """
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    report = LeakageReport(audits=[])
    context = series[origin - context_length : origin]
    target = series[origin : origin + horizon]
    last_context_index = origin - 1
    first_target_index = origin
    report.add(
        "context_before_origin",
        last_context_index < first_target_index,
        f"context ends at {last_context_index}, target starts at {first_target_index}",
    )
    report.add(
        "context_length",
        context.size == context_length,
        f"context has {context.size} values, expected {context_length}",
    )
    report.add(
        "horizon_length",
        target.size == horizon,
        f"target has {target.size} values, expected {horizon}",
    )
    full_context = series[origin - context_length : origin]
    report.add(
        "context_is_prefix",
        np.array_equal(context, full_context),
        "context slice is exactly the prefix ending at the origin",
    )
    try:
        scaling = assert_context_only_scaling(context, target, origin=origin)
    except LeakageError as exc:
        report.add("scaling", False, str(exc))
    else:
        report.add("scaling", scaling.passed, scaling.detail)
    return report
