"""Leakage audits: prove the detectors fire on real leaks.

These are adversarial tests. Each one feeds a *leaky* implementation to an audit
and asserts the audit rejects it, then feeds the context-only reference and
asserts it passes. A test suite that only checks clean code would pass even if
the detector did nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsbench.data import (
    ContextScaler,
    LeakageError,
    SplitConfig,
    assert_context_only_scaling,
    assert_no_future_in_features,
    audit_context_boundary,
    compute_split_plan,
    iter_windows,
)
from tsbench.data.splits import assert_no_future_data


def _series(n: int = 500) -> np.ndarray:
    rng = np.random.default_rng(7)
    t = np.arange(n, dtype=np.float64)
    return 50.0 + 5.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, n)


def _window(origin: int = 400, context_length: int = 48, horizon: int = 12):
    values = _series(500)
    context = values[origin - context_length : origin]
    target = values[origin : origin + horizon]
    return values, context, target, origin


# ---------------------------------------------------------------------------
# Structural guards
# ---------------------------------------------------------------------------


def test_windows_never_overlap_context_and_target():
    values = _series()
    plan = compute_split_plan(values.size, SplitConfig(context_length=48, horizon=12, stride=12))
    for origin, context, target in iter_windows(values, plan, period="test"):
        assert context.size == 48 and target.size == 12
        # Every context index is strictly less than every target index.
        assert origin - context.size >= 0
        assert origin + target.size <= values.size
        assert not np.shares_memory(context, target)


def test_assert_no_future_data_rejects_truncated_lookback():
    plan = compute_split_plan(500, SplitConfig(context_length=48, horizon=12))
    with pytest.raises(LeakageError):
        assert_no_future_data(plan, origin=10, context_length=48, horizon=12)


def test_assert_no_future_data_rejects_overrun_target():
    plan = compute_split_plan(500, SplitConfig(context_length=48, horizon=12))
    with pytest.raises(LeakageError):
        assert_no_future_data(plan, origin=495, context_length=48, horizon=12)


# ---------------------------------------------------------------------------
# Scaling leakage
# ---------------------------------------------------------------------------


def test_reference_scaler_is_context_only_and_passes():
    _, context, target, origin = _window()
    audit = assert_context_only_scaling(context, target, origin=origin)
    assert audit.passed, audit.detail
    scaler = ContextScaler().fit(context, origin=origin)
    assert scaler.center == pytest.approx(float(np.mean(context)))
    scaled_target = scaler.transform(target)
    assert np.all(np.isfinite(scaled_target))


def test_scaler_that_fits_on_context_plus_target_is_rejected():
    _, context, target, origin = _window()

    class LeakyScaler:
        """Fits the mean over context *and* target -- the classic leak."""

        def fit(self, values, origin=None):
            combined = np.concatenate([context, target])
            self.center = float(np.mean(combined))
            self.scale = 1.0
            return self

    audit = assert_context_only_scaling(
        context, target, scaler_factory=LeakyScaler, origin=origin
    )
    assert audit.passed is False
    assert "center" in audit.detail


def test_scaler_that_ignores_its_fitted_window_is_rejected():
    _, context, target, origin = _window()

    class InputBlindScaler:
        """Reports the context mean no matter which window it is fit on."""

        def fit(self, values, origin=None):
            self.center = float(np.mean(context))
            self.scale = 1.0
            return self

    audit = assert_context_only_scaling(
        context, target, scaler_factory=InputBlindScaler, origin=origin
    )
    assert audit.passed is False
    assert audit.name == "context_scaling"
    assert "window" in audit.detail


def test_scaler_transform_that_ignores_its_fit_is_rejected():
    """An honestly-fitted but no-op transform is caught by the perturbation probe."""
    _, context, target, origin = _window()

    class IgnoringTransform:
        """Fits real statistics, then transforms with a function of the input alone."""

        def fit(self, values, origin=None):
            self.center = float(np.mean(values))
            spread = float(np.std(values))
            self.scale = spread if spread > 0 else 1.0
            return self

        def transform(self, values, indices=None):
            return np.asarray(values, dtype=np.float64)

    audit = assert_context_only_scaling(
        context, target, scaler_factory=IgnoringTransform, origin=origin
    )
    assert audit.passed is False
    assert audit.name == "context_scaling"
    assert "ignores the statistics" in audit.detail


def test_scaler_refuses_to_re_derive_at_future_indices():
    _, context, target, origin = _window()
    scaler = ContextScaler().fit(context, origin=origin)
    # Refitting (or re-deriving statistics) at the first future index is a leak.
    with pytest.raises(LeakageError):
        scaler.assert_fit_origin([origin])
    # Transforming future values with context statistics is fine.
    scaler.assert_fit_origin([origin - 1])
    assert np.all(np.isfinite(scaler.transform(target)))


def test_context_scaler_on_constant_window_does_not_divide_by_zero():
    constant = np.full(24, 3.0)
    scaler = ContextScaler().fit(constant)
    assert scaler.scale == 1.0
    scaled = scaler.transform(np.array([3.0, 4.0]))
    assert np.all(np.isfinite(scaled))


# ---------------------------------------------------------------------------
# Feature leakage
# ---------------------------------------------------------------------------


def test_context_only_feature_passes():
    _, context, target, origin = _window()
    position = context.size

    def features(history, position):
        """Lag features for ``position`` read only values before it."""
        history = np.asarray(history, dtype=np.float64)
        return np.array([history[position - 1], history[position - 2]])

    audit = assert_no_future_in_features(context, target, features, position=position)
    assert audit.passed, audit.detail
    assert audit.name == "feature_future_independence"


@pytest.mark.parametrize("how", ["future_lag", "rolling_mean", "concatenate"])
def test_feature_functions_that_read_target_are_rejected(how: str):
    _, context, target, origin = _window()
    position = context.size

    def leaky(history, position):
        """A design that reads the value *at* the forecast target."""
        history = np.asarray(history, dtype=np.float64)
        if how == "concatenate":
            return np.array([history.size], dtype=np.float64)
        if how == "future_lag":
            # BUG: history[position] is the target value itself.
            return np.array([history[position]], dtype=np.float64)
        # rolling_mean over a window that includes the target point
        lo = max(0, position - 1)
        return np.array([float(np.mean(history[lo : position + 1]))], dtype=np.float64)

    audit = assert_no_future_in_features(context, target, leaky, position=position)
    assert not audit.passed
    assert "future" in audit.detail


def test_non_deterministic_feature_is_rejected():
    _, context, target, origin = _window()
    state = {"n": 0}

    def jittery(history, position):
        state["n"] += 1
        return np.array([float(state["n"])], dtype=np.float64)

    audit = assert_no_future_in_features(context, target, jittery, position=context.size)
    assert not audit.passed


# ---------------------------------------------------------------------------
# Boundary audit and report aggregation
# ---------------------------------------------------------------------------


def test_audit_context_boundary_reports_last_seen_index():
    values, _, _, origin = _window()
    report = audit_context_boundary(values, origin, context_length=48, horizon=12)
    assert report.ok
    by_name = {audit.name: audit for audit in report.audits}
    assert by_name["context_before_origin"].passed
    assert by_name["context_length"].passed
    report.require_ok()


def test_leakage_report_aggregates_failures():
    from tsbench.data.leakage import LeakageReport

    report = LeakageReport(audits=[])
    report.add("clean", True, "ok")
    report.add("dirty", False, "found future index 400")
    assert not report.ok
    assert [a.name for a in report.failures] == ["dirty"]
    with pytest.raises(LeakageError, match="dirty"):
        report.require_ok()


# ---------------------------------------------------------------------------
# Declared splits are self-consistent
# ---------------------------------------------------------------------------


def test_train_val_test_boundaries_are_monotonic_and_cover_series():
    values = _series(1000)
    plan = compute_split_plan(values.size, SplitConfig())
    assert 0 < plan.train_end < plan.val_end < plan.test_end == values.size
    assert plan.period_of(0) == "train"
    assert plan.period_of(plan.train_end) == "val"
    assert plan.period_of(plan.val_end) == "test"
    assert plan.period_of(plan.test_end - 1) == "test"
    with pytest.raises(ValueError):
        plan.period_of(plan.test_end)


def test_test_windows_are_strictly_inside_the_test_period():
    values = _series(2000)
    plan = compute_split_plan(
        values.size, SplitConfig(context_length=168, horizon=24, stride=24)
    )
    origins = [
        origin for origin, _, _ in iter_windows(values, plan, period="test")
    ]
    for origin in origins:
        assert origin >= plan.val_end + plan.config.context_length
        assert origin + plan.config.horizon <= plan.test_end


def test_manifest_digest_changes_when_series_changes():
    from tsbench.data import build_manifest, manifest_digest

    values = _series(500)
    split = SplitConfig(context_length=48, horizon=12)

    class _S:
        series_id = "s"

    first = _S()
    first.values = values
    second = _S()
    second.values = values.copy()
    second.values[0] += 1.0

    digest_a = manifest_digest(build_manifest("d", [first], split))
    digest_b = manifest_digest(build_manifest("d", [second], split))
    assert digest_a != digest_b


def test_manifest_verification_catches_tampering():
    from tsbench.data import build_manifest, verify_manifest

    values = _series(500)
    split = SplitConfig(context_length=48, horizon=12)

    class _S:
        def __init__(self, values):
            self.series_id = "s"
            self.values = values

    original = _S(values)
    manifest = build_manifest("d", [original], split)
    assert verify_manifest(manifest, {"s": original}) == []

    tampered = _S(values.copy())
    tampered.values[-1] += 100.0
    problems = verify_manifest(manifest, {"s": tampered})
    assert problems and "digest" in problems[0]
