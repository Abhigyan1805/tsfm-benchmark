"""CPU-only shape/seed tests for the deep forecasters and the Colab runner.

The trained paths run when torch is importable (CPU wheel is enough:
``pip install torch --index-url https://download.pytorch.org/whl/cpu``). On a
stdlib-only host the tests still cover the windowing/seeding/shape contract and
assert the lazy-dependency error instead of skipping silently.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tsbench.base import ModelInfo as FrozenModelInfo
from tsbench.models.deep import (
    make_windows,
    normalize,
    set_seed,
    to_float_list,
)
from tsbench.models.deep.lstm import LSTMForecaster
from tsbench.models.deep.transformer import TransformerForecaster

ROOT = Path(__file__).resolve().parents[1]
HAVE_TORCH = importlib.util.find_spec("torch") is not None


def _values(result) -> list[float]:
    """Normalize a predict result (numpy array or list) to a float list."""
    if hasattr(result, "tolist"):
        result = result.tolist()
    return [float(value) for value in result]


def _load_colab_run():
    spec = importlib.util.spec_from_file_location(
        "colab_run", ROOT / "scripts" / "colab_run.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SharedHelperTests(unittest.TestCase):
    def test_make_windows_shapes(self) -> None:
        inputs, targets = make_windows(list(range(10)), context=3, horizon=2)
        self.assertEqual(len(inputs), 6)
        self.assertEqual(inputs[0], [0.0, 1.0, 2.0])
        self.assertEqual(targets[0], [3.0, 4.0])
        self.assertEqual(inputs[-1], [5.0, 6.0, 7.0])
        self.assertEqual(targets[-1], [8.0, 9.0])

    def test_make_windows_rejects_short_series(self) -> None:
        with self.assertRaises(ValueError):
            make_windows([1.0, 2.0], context=2, horizon=2)

    def test_normalize_is_unit_scale(self) -> None:
        normalized, mean, std = normalize([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(mean, 2.5)
        self.assertGreater(std, 0.0)
        self.assertAlmostEqual(sum(normalized) / len(normalized), 0.0, places=12)

    def test_to_float_list_flattens_tensor_like(self) -> None:
        class FakeTensor:
            def tolist(self):
                return [[1.0, 2.0], [3.0, 4.0]]

        self.assertEqual(to_float_list(FakeTensor()), [1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            to_float_list([1.0, math.inf])

    def test_set_seed_is_reproducible(self) -> None:
        set_seed(7)
        first = [random.random() for _ in range(5)]
        set_seed(7)
        second = [random.random() for _ in range(5)]
        self.assertEqual(first, second)


class DeepConfigTests(unittest.TestCase):
    def test_registry_identity(self) -> None:
        self.assertEqual(LSTMForecaster.name, "lstm")
        self.assertEqual(TransformerForecaster.name, "transformer")
        self.assertIs(LSTMForecaster.zero_shot, False)
        self.assertIs(TransformerForecaster.zero_shot, False)

    def test_info_before_fit(self) -> None:
        info = LSTMForecaster().info()
        self.assertEqual(info.name, "lstm")
        self.assertEqual(info.family, "deep")
        self.assertIs(info.zero_shot, False)
        self.assertIsNone(info.params)
        self.assertEqual(info.license, "Apache-2.0")
        self.assertEqual(info.extra["config"]["seed"], 0)
        self.assertFalse(info.extra["trained"])

    def test_info_matches_frozen_contract(self) -> None:
        self.assertIsInstance(LSTMForecaster().info(), FrozenModelInfo)
        self.assertIsInstance(TransformerForecaster().info(), FrozenModelInfo)

    def test_invalid_configs(self) -> None:
        with self.assertRaises(ValueError):
            LSTMForecaster(context=0)
        with self.assertRaises(ValueError):
            LSTMForecaster(lr=0.0)
        with self.assertRaises(ValueError):
            TransformerForecaster(heads=3, hidden=8)
        with self.assertRaises(ValueError):
            TransformerForecaster(feedforward=0)

    def test_predict_before_fit_raises_without_torch(self) -> None:
        if HAVE_TORCH:
            self.skipTest("torch is installed; fit path is exercised separately")
        with self.assertRaisesRegex(ImportError, "torch"):
            LSTMForecaster().fit([float(i) for i in range(10)])

    def test_predict_before_fit_raises(self) -> None:
        if not HAVE_TORCH:
            self.skipTest("torch is not installed")
        with self.assertRaises(RuntimeError):
            LSTMForecaster().predict(3)
        with self.assertRaises(RuntimeError):
            TransformerForecaster().predict(3)

    def test_fit_rejects_short_series(self) -> None:
        if not HAVE_TORCH:
            self.skipTest("torch is not installed")
        model = LSTMForecaster(context=8)
        with self.assertRaises(ValueError):
            model.fit([1.0, 2.0, 3.0])


def _sine(n: int = 48) -> list[float]:
    return [math.sin(i / 3.0) for i in range(n)]


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class DeepTrainingTests(unittest.TestCase):
    def test_lstm_shapes_and_seed_determinism(self) -> None:
        series = _sine()
        first = LSTMForecaster(context=4, hidden=4, epochs=2, batch_size=8, seed=11)
        first.fit(series)
        predictions = _values(first.predict(5))
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(math.isfinite(value) for value in predictions))
        self.assertGreater(first.info().params, 0)

        second = LSTMForecaster(context=4, hidden=4, epochs=2, batch_size=8, seed=11)
        second.fit(series)
        self.assertEqual(predictions, _values(second.predict(5)))

    def test_transformer_shapes_and_chunked_horizon(self) -> None:
        series = _sine()
        model = TransformerForecaster(
            context=4,
            hidden=8,
            layers=1,
            heads=2,
            feedforward=16,
            max_horizon=3,
            epochs=2,
            batch_size=8,
            seed=3,
        )
        model.fit(series)
        predictions = _values(model.predict(7))
        self.assertEqual(len(predictions), 7)
        self.assertTrue(all(math.isfinite(value) for value in predictions))

    def test_different_seeds_can_differ(self) -> None:
        series = _sine()
        left = LSTMForecaster(context=4, hidden=4, epochs=2, seed=1).fit(series)
        right = LSTMForecaster(context=4, hidden=4, epochs=2, seed=2).fit(series)
        self.assertNotEqual(_values(left.predict(3)), _values(right.predict(3)))


class ColabRunnerTests(unittest.TestCase):
    def test_cache_key_covers_job_and_series(self) -> None:
        colab_run = _load_colab_run()
        job = {"model": "lstm", "horizon": 4, "params": {"seed": 0}}
        series = _sine(12)
        key = colab_run.cache_key(job, series)
        self.assertEqual(len(key), 16)
        self.assertEqual(key, colab_run.cache_key(dict(job), list(series)))
        self.assertNotEqual(key, colab_run.cache_key({**job, "horizon": 5}, series))
        self.assertNotEqual(
            key, colab_run.cache_key(job, [value + 0.1 for value in series])
        )

    def test_load_series_inline(self) -> None:
        colab_run = _load_colab_run()
        self.assertEqual(colab_run.load_series({"series": [1, 2, 3]}), [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            colab_run.load_series({"series": []})
        with self.assertRaises(ValueError):
            colab_run.load_series({})

    def test_result_files_are_idempotent(self) -> None:
        colab_run = _load_colab_run()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            out.mkdir()
            key = colab_run.cache_key({"model": "fake", "horizon": 2}, [1.0, 2.0, 3.0])
            payload = {"cache_key": key, "ok": True}
            path = out / f"{key}.json"
            colab_run.write_json_atomic(path, payload)
            self.assertTrue(path.exists())
            self.assertFalse((out / f"{key}.json.tmp").exists())
            self.assertEqual(
                key,
                colab_run.cache_key({"horizon": 2, "model": "fake"}, [1.0, 2.0, 3.0]),
            )
            self.assertIn(os.path.basename(path), os.listdir(out))


class EndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoint = _load_script("serve_endpoint", "serve_endpoint.py")

    def test_extract_jobs_accepts_the_documented_shapes(self) -> None:
        single = {"model": "naive", "series": [1, 2, 3], "horizon": 1}
        self.assertEqual(self.endpoint.extract_jobs(single), [single])
        self.assertEqual(
            self.endpoint.extract_jobs({"jobs": [single]}), [single]
        )
        self.assertEqual(self.endpoint.extract_jobs([{"entry": "a:B"}]), [{"entry": "a:B"}])
        for bad in ({}, {"jobs": []}, [1], "nope"):
            with self.assertRaises(ValueError):
                self.endpoint.extract_jobs(bad)

    def test_auth_ok_requires_matching_bearer_token(self) -> None:
        self.assertTrue(self.endpoint.auth_ok("Bearer secret", "secret"))
        self.assertTrue(self.endpoint.auth_ok("bearer secret", "secret"))
        self.assertFalse(self.endpoint.auth_ok("Bearer wrong", "secret"))
        self.assertFalse(self.endpoint.auth_ok(None, "secret"))
        self.assertFalse(self.endpoint.auth_ok("Bearer secret", ""))

    def test_run_payload_serves_from_cache(self) -> None:
        calls: list[str] = []

        def fake_execute(job, series, key):
            calls.append(key)
            return {"cache_key": key, "model": job["model"]}

        job = {"model": "naive", "series": [1.0, 2.0, 3.0], "horizon": 1}
        with (
            mock.patch.object(
                self.endpoint.colab_run, "execute_job", side_effect=fake_execute
            ),
            tempfile.TemporaryDirectory() as tmp,
        ):
            out = Path(tmp)
            first = self.endpoint.run_payload({"jobs": [job]}, out)
            second = self.endpoint.run_payload({"jobs": [job]}, out)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first[0]["status"], "ok")
        self.assertEqual(second[0]["status"], "cached")

    def test_run_payload_enforces_max_jobs(self) -> None:
        job = {"model": "naive", "series": [1.0, 2.0, 3.0], "horizon": 1}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.endpoint.run_payload(
                    {"jobs": [job, job]}, Path(tmp), max_jobs=1
                )

    def test_build_server_refuses_without_token(self) -> None:
        with self.assertRaises(ValueError):
            self.endpoint.build_server("127.0.0.1", 0, "", Path("."))

    def test_health_payload_is_secret_free(self) -> None:
        health = self.endpoint.health_payload(Path("/tmp"))
        self.assertEqual(health["status"], "ok")
        self.assertNotIn("token", health)


class KaggleRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kaggle = _load_script("kaggle_run", "kaggle_run.py")
        self.jobs = [{"model": "naive", "series": [1.0, 2.0, 3.0], "horizon": 1}]

    def test_parse_username(self) -> None:
        text = "Configuration values from /home/x/.kaggle\n- username: alice\n"
        self.assertEqual(self.kaggle.parse_username(text), "alice")
        with self.assertRaises(ValueError):
            self.kaggle.parse_username("no username here")

    def test_kernel_slug_is_deterministic_and_ref_sensitive(self) -> None:
        first = self.kaggle.kernel_slug(self.jobs, "abc")
        self.assertEqual(first, self.kaggle.kernel_slug(self.jobs, "abc"))
        self.assertNotEqual(first, self.kaggle.kernel_slug(self.jobs, "def"))
        self.assertRegex(first, r"^tsbench-gpu-[0-9a-f]{12}$")

    def test_kernel_source_embeds_jobs_and_gate(self) -> None:
        source = self.kaggle.kernel_source(
            self.jobs,
            repo="https://example.invalid/repo.git",
            ref="deadbeef",
            run_name="run-1",
            pip_packages=["timesfm[torch]"],
        )
        self.assertIn("TSBENCH_ALLOW_MODEL_DOWNLOAD", source)
        self.assertIn("timesfm[torch]", source)
        self.assertIn("/kaggle/working", source)
        self.assertIn("git\", \"clone", source)

    def test_build_kernel_dir_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            metadata = self.kaggle.build_kernel_dir(
                workdir,
                self.jobs,
                owner="alice",
                slug="tsbench-gpu-test",
                title="t",
                repo="https://example.invalid/repo.git",
                ref="deadbeef",
                run_name="run-1",
                pip_packages=[],
            )
            self.assertEqual(metadata["id"], "alice/tsbench-gpu-test")
            self.assertEqual(metadata["enable_gpu"], "true")
            self.assertEqual(metadata["enable_internet"], "true")
            self.assertTrue((workdir / "kernel.py").exists())
            self.assertTrue((workdir / "kernel-metadata.json").exists())

    def test_partition_and_ingest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            pending, cached = self.kaggle.partition_jobs(self.jobs, out)
            self.assertEqual(len(pending), 1)
            self.assertEqual(cached, [])
            key = pending[0][2]

            fetched = root / "fetched"
            run = fetched / "run-1"
            run.mkdir(parents=True)
            (run / f"{key}.json").write_text(
                json.dumps({"cache_key": key, "model": "naive"}), encoding="utf-8"
            )
            (run / "manifest.json").write_text("{}", encoding="utf-8")
            with tarfile.open(fetched / "run-1.tar.gz", "w:gz") as archive:
                archive.add(run, arcname="run-1")

            summary = self.kaggle.ingest(fetched, out, [key])
            self.assertEqual(summary["ingested"], [key])
            self.assertEqual(summary["missing"], [])

            pending_after, cached_after = self.kaggle.partition_jobs(self.jobs, out)
            self.assertEqual(pending_after, [])
            self.assertEqual(len(cached_after), 1)

    def test_ingest_reports_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fetched = root / "fetched"
            fetched.mkdir()
            summary = self.kaggle.ingest(fetched, root / "out", ["0" * 16])
            self.assertEqual(summary["ingested"], [])
            self.assertEqual(summary["missing"], ["0" * 16])


if __name__ == "__main__":
    unittest.main()
