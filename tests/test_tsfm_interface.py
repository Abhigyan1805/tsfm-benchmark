"""Interface tests for the zero-shot TSFM wrappers (fake backend, no network).

The license gate and every shape/decoding path run with plain objects; no
checkpoint is imported, downloaded or touched. Real weight loads happen only in
a GPU session with ``TSBENCH_ALLOW_MODEL_DOWNLOAD=1`` (docs/colab-handoff.md).
"""

from __future__ import annotations

import importlib.util
import math
import os
import unittest
from unittest import mock

from tsbench.models.tsfm import (
    CHRONOS_VERIFIED,
    DOWNLOAD_ENV,
    TIMESFM_APPROVED,
    DownloadNotAllowed,
    UnverifiedWeights,
    download_allowed,
    flatten_floats,
    median_point,
    require_weights_allowed,
)
from tsbench.models.tsfm.chronos import ChronosBolt
from tsbench.models.tsfm.timesfm import TimesFM25

HAVE_TIMESFM = importlib.util.find_spec("timesfm") is not None
HAVE_CHRONOS = importlib.util.find_spec("chronos") is not None


class FakeTimesFMBackend:
    """Mimics ``TimesFM_2p5_200M_torch.forecast(horizon, inputs)``."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_inputs = None

    def forecast(self, horizon, inputs):
        self.calls += 1
        self.seen_inputs = inputs
        return [[float(index + 1) for index in range(horizon)]], None


class FakeChronosBackend:
    """Mimics ``ChronosBoltPipeline.predict(inputs, prediction_length)``."""

    def __init__(self) -> None:
        self.calls = 0
        self.prediction_length = None

    def predict(self, inputs, prediction_length=None):
        self.calls += 1
        self.prediction_length = prediction_length
        return [[[float(quantile)] * prediction_length for quantile in range(3)]]


def _sine(n: int = 24) -> list[float]:
    return [math.sin(index / 2.0) for index in range(n)]


class LicenseGateTests(unittest.TestCase):
    def test_default_models_are_the_approved_lines(self) -> None:
        self.assertEqual(TimesFM25().model_id, "google/timesfm-2.5-200m-pytorch")
        self.assertEqual(ChronosBolt().model_id, "amazon/chronos-bolt-small")
        self.assertIn(TimesFM25().model_id, TIMESFM_APPROVED)
        self.assertIn(ChronosBolt().model_id, CHRONOS_VERIFIED)

    def test_info_records_revision_and_license(self) -> None:
        info = TimesFM25().info()
        self.assertTrue(info.zero_shot)
        self.assertEqual(info.license, "apache-2.0")
        self.assertEqual(
            info.revision, TIMESFM_APPROVED[info.extra["model_id"]].revision
        )
        self.assertEqual(info.params, TIMESFM_APPROVED[info.extra["model_id"]].params)
        self.assertFalse(info.extra["weights_loaded"])
        self.assertTrue(info.extra["license_verified"])
        chronos_info = ChronosBolt().info()
        self.assertEqual(chronos_info.name, "chronos_bolt")
        self.assertTrue(chronos_info.zero_shot)
        self.assertEqual(chronos_info.family, "tsfm")

    def test_timesfm_3_0_is_refused(self) -> None:
        with self.assertRaisesRegex(UnverifiedWeights, "non-commercial"):
            TimesFM25(model_id="google/timesfm-3.0-pytorch")
        with self.assertRaises(UnverifiedWeights):
            TimesFM25(model_id="google/timesfm-2.0-500m-pytorch")
        with self.assertRaises(UnverifiedWeights):
            TimesFM25(model_id="someone/random-checkpoint")

    def test_chronos_unverified_ids_are_refused(self) -> None:
        with self.assertRaises(UnverifiedWeights):
            ChronosBolt(model_id="amazon/chronos-2")
        with self.assertRaises(UnverifiedWeights):
            ChronosBolt(model_id="amazon/chronos-t5-small")

    def test_chronos_all_verified_ids_load_metadata(self) -> None:
        for model_id, weights in CHRONOS_VERIFIED.items():
            model = ChronosBolt(model_id=model_id)
            self.assertEqual(model.license, weights.license)
            self.assertEqual(model.revision, weights.revision)

    def test_download_gate_blocks_before_import(self) -> None:
        with mock.patch.dict(os.environ, {DOWNLOAD_ENV: "0"}):
            self.assertFalse(download_allowed())
            model = TimesFM25().fit(_sine())
            with self.assertRaises(DownloadNotAllowed):
                model.predict(3)
            chronos = ChronosBolt().fit(_sine())
            with self.assertRaises(DownloadNotAllowed):
                chronos.predict(3)

    def test_non_permissive_license_is_refused(self) -> None:
        weights = TIMESFM_APPROVED["google/timesfm-2.5-200m-pytorch"]
        bad = type(weights)(
            model_id=weights.model_id,
            license="cc-by-nc-4.0",
            revision=weights.revision,
            source=weights.source,
            verified_on=weights.verified_on,
        )
        with (
            mock.patch.dict(os.environ, {DOWNLOAD_ENV: "1"}),
            self.assertRaises(UnverifiedWeights),
        ):
            require_weights_allowed(weights.model_id, bad)

    def test_gate_allows_verified_weights_when_opted_in(self) -> None:
        with mock.patch.dict(os.environ, {DOWNLOAD_ENV: "1"}):
            self.assertTrue(download_allowed())
            weights = TIMESFM_APPROVED["google/timesfm-2.5-200m-pytorch"]
            require_weights_allowed(weights.model_id, weights)


class FakeBackendForecastTests(unittest.TestCase):
    def test_timesfm_shapes(self) -> None:
        backend = FakeTimesFMBackend()
        model = TimesFM25(backend=backend)
        model.fit(_sine())
        predictions = _values(model.predict(6))
        self.assertEqual(len(predictions), 6)
        self.assertEqual(predictions, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(backend.calls, 1)
        self.assertTrue(info_weights(model))

    def test_chronos_uses_median_quantile(self) -> None:
        backend = FakeChronosBackend()
        model = ChronosBolt(backend=backend)
        model.fit(_sine())
        predictions = _values(model.predict(4))
        self.assertEqual(predictions, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(backend.prediction_length, 4)

    def test_context_is_capped_and_finite(self) -> None:
        model = TimesFM25(backend=FakeTimesFMBackend(), max_context=64)
        model.fit([float(index) for index in range(200)])
        self.assertEqual(model.info().extra["context_len"], 64)

    def test_predict_requires_fit(self) -> None:
        with self.assertRaises(RuntimeError):
            TimesFM25(backend=FakeTimesFMBackend()).predict(2)
        with self.assertRaises(RuntimeError):
            ChronosBolt(backend=FakeChronosBackend()).predict(2)

    def test_fit_rejects_empty_or_non_finite(self) -> None:
        with self.assertRaises(ValueError):
            TimesFM25(backend=FakeTimesFMBackend()).fit([])
        with self.assertRaises(ValueError):
            ChronosBolt(backend=FakeChronosBackend()).fit([1.0, float("nan")])
        with self.assertRaises(ValueError):
            ChronosBolt(backend=FakeChronosBackend()).fit(_sine()).predict(0)

    def test_real_import_error_is_actionable(self) -> None:
        if HAVE_TIMESFM or HAVE_CHRONOS:
            self.skipTest("a real TSFM package is installed; import check skipped")
        with mock.patch.dict(os.environ, {DOWNLOAD_ENV: "1"}):
            with self.assertRaisesRegex(ImportError, "timesfm"):
                TimesFM25().fit(_sine()).predict(2)
            with self.assertRaisesRegex(ImportError, "chronos"):
                ChronosBolt().fit(_sine()).predict(2)


def info_weights(model) -> bool:
    return bool(model.info().extra["weights_loaded"])


def _values(result) -> list[float]:
    """Normalize a predict result (numpy array or list) to a float list."""
    if hasattr(result, "tolist"):
        result = result.tolist()
    return [float(value) for value in result]


class OutputConversionTests(unittest.TestCase):
    def test_flatten_floats(self) -> None:
        self.assertEqual(flatten_floats([[1, 2], [3]]), [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            flatten_floats([1.0, float("-inf")])

    def test_median_point_variants(self) -> None:
        self.assertEqual(median_point([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])
        self.assertEqual(median_point([[1.0, 2.0, 3.0]]), [1.0, 2.0, 3.0])
        nested = [[[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]]]
        self.assertEqual(median_point(nested), [5.0, 5.0])
        with self.assertRaises(ValueError):
            median_point([])


if __name__ == "__main__":
    unittest.main()
