"""GPU-tier experiment configs and the Kaggle experiment compute route.

The GPU configs must evaluate the deep and zero-shot TSFM families on the exact
frozen windows the local CPU families used, and the Kaggle route must pack them
into a resumable, self-contained kernel. Nothing here touches the network or a
GPU; the kernel is exercised with ``subprocess`` mocked.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from tsbench.evaluation.results import config_hash
from tsbench.evaluation.runner import load_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "kaggle_run.py"
GPU_CONFIGS = ["gpu", "gpu_h48", "gpu_h96", "gpu_h192"]
CPU_CONFIGS = ["backtest", "backtest_h48", "backtest_h96", "backtest_h192"]
GPU_MODELS = ["lstm", "transformer", "timesfm25", "chronos_bolt"]


def _load_kaggle():
    spec = importlib.util.spec_from_file_location("kaggle_run", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_configs_reuse_the_cpu_frozen_windows():
    for gpu_name, cpu_name, horizon in zip(
        GPU_CONFIGS, CPU_CONFIGS, [24, 48, 96, 192], strict=True
    ):
        gpu = load_experiment(REPO_ROOT / "configs" / "experiments" / f"{gpu_name}.yaml")
        cpu = load_experiment(REPO_ROOT / "configs" / "experiments" / f"{cpu_name}.yaml")
        assert gpu.split.train_frac == cpu.split.train_frac
        assert gpu.split.val_frac == cpu.split.val_frac
        assert gpu.split.test_frac == cpu.split.test_frac
        assert gpu.split.context_length == cpu.split.context_length == 168
        assert gpu.split.stride == cpu.split.stride == 240
        assert gpu.split.min_train_size == cpu.split.min_train_size == 168
        assert gpu.split.horizon == cpu.split.horizon == horizon
        assert gpu.season_length == cpu.season_length == 24
        assert gpu.seed == cpu.seed == 20240919
        assert gpu.dataset["name"] == cpu.dataset["name"] == "electricity_hourly"
        assert gpu.dataset["split_manifest"] == cpu.dataset["split_manifest"]
        assert gpu.models == GPU_MODELS
        assert gpu.track_memory is True
        for model in GPU_MODELS:
            assert gpu.raw["model_configs"][model]["kwargs"]


def test_gpu_probe_covers_every_gpu_registry_key_on_the_fixture():
    probe = load_experiment(REPO_ROOT / "configs" / "experiments" / "gpu_probe.yaml")
    assert probe.models == GPU_MODELS
    assert probe.dataset["name"] == "smoke"
    assert probe.dataset["loader"] == "local_csv"
    assert probe.dataset["split_manifest"] == "tests/fixtures/smoke_split_manifest.json"


def test_gpu_models_are_permissive_registry_keys():
    from tsbench.registry import load_registry

    registry = load_registry(REPO_ROOT / "configs" / "models.yaml")
    for model in GPU_MODELS:
        assert model in registry, model
        assert registry.spec(model).license == "Apache-2.0"


def test_experiment_slug_is_deterministic_and_ref_sensitive():
    kaggle = _load_kaggle()
    configs = ["configs/experiments/gpu.yaml"]
    first = kaggle.experiment_slug(configs, "abc")
    assert first == kaggle.experiment_slug(list(configs), "abc")
    assert first != kaggle.experiment_slug(configs, "def")
    assert first.startswith("tsbench-gpu-exp-")


def test_experiment_config_hash_matches_the_runner():
    kaggle = _load_kaggle()
    for name in GPU_CONFIGS:
        path = REPO_ROOT / "configs" / "experiments" / f"{name}.yaml"
        assert kaggle.experiment_config_hash(str(path)) == config_hash(
            dict(load_experiment(path).raw)
        )


def test_partition_experiments_resumes_ingested_runs(tmp_path: Path):
    kaggle = _load_kaggle()
    configs = [
        str(REPO_ROOT / "configs" / "experiments" / "gpu.yaml"),
        str(REPO_ROOT / "configs" / "experiments" / "gpu_h48.yaml"),
    ]
    out = tmp_path / "results"
    done = out / "20260101T000000Z-gpu"
    done.mkdir(parents=True)
    (done / "run.json").write_text(
        json.dumps({"config_hash": kaggle.experiment_config_hash(configs[0])}),
        encoding="utf-8",
    )
    pending, cached = kaggle.partition_experiments(configs, out)
    assert [path for path, _ in cached] == [configs[0]]
    assert [path for path, _ in pending] == [configs[1]]


def test_experiment_kernel_source_runs_each_config(tmp_path: Path):
    from unittest import mock

    kaggle = _load_kaggle()
    configs = ["configs/experiments/gpu.yaml", "configs/experiments/gpu_h48.yaml"]
    source = kaggle.experiment_kernel_source(
        configs,
        repo="https://example.invalid/repo.git",
        ref="deadbeef",
        run_name="run-1",
        pip_packages=["timesfm[torch]"],
    )
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append([str(part) for part in cmd])
        return subprocess.CompletedProcess(cmd, 0)

    namespace: dict = {}
    with mock.patch.dict(os.environ, {"TSBENCH_KAGGLE_WORKING": str(tmp_path)}):
        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            exec(compile(source, "kernel.py", "exec"), namespace)

    assert namespace["CONFIGS"] == configs
    assert namespace["PIP_PACKAGES"] == ["timesfm[torch]"]
    assert calls[0][:4] == [sys.executable, "-m", "pip", "install"]
    assert calls[1] == [
        "git",
        "clone",
        "--quiet",
        "https://example.invalid/repo.git",
        str(namespace["REPO_DIR"]),
    ]
    assert calls[2][:3] == ["git", "-C", str(namespace["REPO_DIR"])]
    assert calls[3][1:3] == ["-m", "tsbench.data.build_catalog"]
    run_calls = [call for call in calls if call[1:3] == ["-m", "tsbench"]]
    assert len(run_calls) == len(configs)
    for call, config in zip(run_calls, configs, strict=True):
        assert call[5] == config
    assert (tmp_path / "run-1" / "results.tar.gz").is_file()


def test_extract_experiment_results_restores_run_dirs(tmp_path: Path):
    kaggle = _load_kaggle()
    run_id = "20260101T000000Z-gpu"
    staging = tmp_path / "staging" / run_id
    staging.mkdir(parents=True)
    (staging / "results.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (staging / "run.json").write_text("{}", encoding="utf-8")
    fetched = tmp_path / "fetched"
    fetched.mkdir()
    with tarfile.open(fetched / "results.tar.gz", "w:gz") as archive:
        for path in sorted(staging.rglob("*")):
            archive.add(path, arcname=str(path.relative_to(staging.parent)))
    out = tmp_path / "out"
    extracted = kaggle.extract_experiment_results(fetched, out)
    assert extracted
    assert (out / run_id / "results.csv").is_file()
    assert (out / run_id / "run.json").is_file()
