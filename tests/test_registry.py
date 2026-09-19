"""Registry resolution, manifest validation, and the license gate."""

from __future__ import annotations

import csv
import datetime as dt
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

from tsbench.base import RESULT_COLUMNS, Forecaster, ModelInfo
from tsbench.cli import main
from tsbench.registry import (
    ALLOW_NONCOMMERCIAL_ENV,
    LicenseError,
    ModelConfigError,
    ModelImportError,
    RegistryError,
    is_permissive_license,
    load_registry,
    noncommercial_allowed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"
SMOKE_CONFIG = REPO_ROOT / "configs" / "experiments" / "smoke.yaml"

FROZEN_MODELS: dict[str, tuple[str, str, bool]] = {
    "naive": ("tsbench.models.naive:Naive", "baseline", True),
    "seasonal_naive": ("tsbench.models.naive:SeasonalNaive", "baseline", True),
    "auto_ets": ("tsbench.models.stats:AutoETS", "classical", False),
    "auto_arima": ("tsbench.models.stats:AutoARIMA", "classical", False),
    "xgboost_lags": ("tsbench.models.gbdt:XGBoostLags", "ml", False),
    "lstm": ("tsbench.models.deep.lstm:LSTMForecaster", "deep", False),
    "transformer": ("tsbench.models.deep.transformer:TransformerForecaster", "deep", False),
    "timesfm25": ("tsbench.models.tsfm.timesfm:TimesFM25", "tsfm", True),
    "chronos_bolt": ("tsbench.models.tsfm.chronos:ChronosBolt", "tsfm", True),
}

FAKE_MODULE_SOURCE = '''
import numpy as np

from tsbench.base import ModelInfo


class Good:
    name = "good"

    def fit(self, y):
        self._y = np.asarray(y, dtype=float)
        return self

    def predict(self, h):
        return np.full(h, float(self._y[-1]))

    def info(self):
        return ModelInfo("good", "baseline", True, 0, "MIT", "test-pin")


class NotAForecaster:
    name = "not-a-forecaster"

    def predict(self, h):
        return np.zeros(h)
'''


@pytest.fixture(autouse=True)
def _clear_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALLOW_NONCOMMERCIAL_ENV, raising=False)


@pytest.fixture
def fake_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    module_name = f"tsbench_fake_{uuid.uuid4().hex}"
    (tmp_path / f"{module_name}.py").write_text(FAKE_MODULE_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    yield module_name
    monkeypatch.delitem(sys.modules, module_name, raising=False)


def _entry(entrypoint: str, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "entrypoint": entrypoint,
        "family": "baseline",
        "zero_shot": True,
        "license": "MIT",
        "revision": "test-pin",
    }
    entry.update(overrides)
    return entry


def _write_config(tmp_path: Path, models: dict[str, Any]) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"models": models}), encoding="utf-8")
    return path


def test_result_columns_are_frozen() -> None:
    assert RESULT_COLUMNS == (
        "run_id",
        "dataset",
        "series_id",
        "model",
        "family",
        "context_length",
        "horizon",
        "window_index",
        "mae",
        "rmse",
        "mase",
        "smape",
        "latency_ms",
        "peak_mem_mb",
        "train_seconds",
        "params",
        "zero_shot",
        "git_sha",
        "config_hash",
        "measured",
        "timestamp",
    )


def test_model_info_carries_contract_fields() -> None:
    info = ModelInfo("naive", "baseline", True, 0, "Apache-2.0", "repo-local:0.0.1")
    assert info.name == "naive"
    assert info.family == "baseline"
    assert info.zero_shot is True
    assert info.params == 0
    assert info.license == "Apache-2.0"
    assert info.revision == "repo-local:0.0.1"
    assert info.extra == {}


def test_shipped_manifest_matches_frozen_registry() -> None:
    registry = load_registry(MODELS_CONFIG)
    assert set(registry.keys()) == set(FROZEN_MODELS)
    for key, (entrypoint, family, zero_shot) in FROZEN_MODELS.items():
        spec = registry.spec(key)
        assert spec.entrypoint == entrypoint
        assert spec.family == family
        assert spec.zero_shot is zero_shot
        assert spec.license.strip()
        assert spec.revision.strip()
        assert spec.permissive, f"{key} must stay on the permissive allowlist"


def test_shipped_tsfm_entries_pin_weights_and_revision() -> None:
    registry = load_registry(MODELS_CONFIG)
    timesfm = registry.spec("timesfm25")
    assert timesfm.weights == "google/timesfm-2.5-200m-pytorch"
    assert timesfm.params == 200_000_000
    chronos = registry.spec("chronos_bolt")
    assert chronos.weights == "amazon/chronos-bolt-base"
    assert chronos.params == 205_000_000
    for key in ("timesfm25", "chronos_bolt"):
        assert registry.spec(key).revision.strip()


def test_missing_license_is_refused(tmp_path: Path) -> None:
    entry = _entry("some.module:Model", family="baseline")
    del entry["license"]
    with pytest.raises(ModelConfigError, match="license"):
        load_registry(_write_config(tmp_path, {"m": entry}))


def test_missing_revision_is_refused(tmp_path: Path) -> None:
    entry = _entry("some.module:Model")
    del entry["revision"]
    with pytest.raises(ModelConfigError, match="revision"):
        load_registry(_write_config(tmp_path, {"m": entry}))


def test_blank_license_is_refused(tmp_path: Path) -> None:
    entry = _entry("some.module:Model", license="   ")
    with pytest.raises(ModelConfigError, match="license"):
        load_registry(_write_config(tmp_path, {"m": entry}))


def test_unknown_field_is_refused(tmp_path: Path) -> None:
    entry = _entry("some.module:Model", licence="MIT")
    with pytest.raises(ModelConfigError, match="unknown field"):
        load_registry(_write_config(tmp_path, {"m": entry}))


def test_invalid_entrypoint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ModelConfigError, match="module:attribute"):
        load_registry(_write_config(tmp_path, {"m": _entry("some.module.Model")}))


def test_unknown_family_is_refused(tmp_path: Path) -> None:
    entry = _entry("some.module:Model", family="foundation")
    with pytest.raises(ModelConfigError, match="family"):
        load_registry(_write_config(tmp_path, {"m": entry}))


def test_non_boolean_zero_shot_is_refused(tmp_path: Path) -> None:
    entry = _entry("some.module:Model", zero_shot="yes")
    with pytest.raises(ModelConfigError, match="zero_shot"):
        load_registry(_write_config(tmp_path, {"m": entry}))


def test_no_public_ungated_resolver(tmp_path: Path, fake_module: str) -> None:
    registry = load_registry(
        _write_config(tmp_path, {"nc": _entry(f"{fake_module}:Good", license="CC-BY-NC-4.0")})
    )
    assert not hasattr(registry, "entrypoint")
    with pytest.raises(LicenseError, match="CC-BY-NC-4.0"):
        registry.instantiate("nc")


def test_instantiate_resolves_entrypoint_into_forecaster(
    tmp_path: Path, fake_module: str
) -> None:
    registry = load_registry(
        _write_config(tmp_path, {"good": _entry(f"{fake_module}:Good")})
    )
    model = registry.instantiate("good")
    assert isinstance(model, Forecaster)
    assert model.name == "good"
    fitted = model.fit([1.0, 2.0, 3.0])
    assert fitted is model
    assert list(model.predict(3)) == [3.0, 3.0, 3.0]
    assert model.info().license == "MIT"


def test_instantiate_rejects_non_forecaster(tmp_path: Path, fake_module: str) -> None:
    registry = load_registry(
        _write_config(tmp_path, {"bad": _entry(f"{fake_module}:NotAForecaster")})
    )
    with pytest.raises(ModelImportError, match="Forecaster"):
        registry.instantiate("bad")


def test_missing_module_names_the_model(tmp_path: Path) -> None:
    registry = load_registry(
        _write_config(
            tmp_path,
            {"ghost": _entry("tsbench_missing_module_for_tests:Model")},
        )
    )
    with pytest.raises(ModelImportError, match="ghost"):
        registry.instantiate("ghost")


def test_unknown_model_name_is_refused(tmp_path: Path, fake_module: str) -> None:
    registry = load_registry(
        _write_config(tmp_path, {"good": _entry(f"{fake_module}:Good")})
    )
    with pytest.raises(RegistryError, match="unknown model"):
        registry.instantiate("nope")


def test_noncommercial_license_blocks_until_override(
    tmp_path: Path, fake_module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = load_registry(
        _write_config(
            tmp_path,
            {"nc": _entry(f"{fake_module}:Good", license="CC-BY-NC-4.0")},
        )
    )
    with pytest.raises(LicenseError, match="CC-BY-NC-4.0"):
        registry.instantiate("nc")
    monkeypatch.setenv(ALLOW_NONCOMMERCIAL_ENV, "1")
    assert registry.instantiate("nc").name == "good"
    monkeypatch.delenv(ALLOW_NONCOMMERCIAL_ENV)
    assert registry.instantiate("nc", allow_noncommercial=True).name == "good"


def test_unknown_license_blocks_until_override(
    tmp_path: Path, fake_module: str
) -> None:
    registry = load_registry(
        _write_config(
            tmp_path,
            {"custom": _entry(f"{fake_module}:Good", license="Acme-Research-Only")},
        )
    )
    with pytest.raises(LicenseError, match="Acme-Research-Only"):
        registry.instantiate("custom")
    assert registry.instantiate("custom", allow_noncommercial=True).name == "good"


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_noncommercial_env_truthy(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ALLOW_NONCOMMERCIAL_ENV, value)
    assert noncommercial_allowed() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "banana"])
def test_noncommercial_env_falsy(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ALLOW_NONCOMMERCIAL_ENV, value)
    assert noncommercial_allowed() is False


def test_explicit_override_beats_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOW_NONCOMMERCIAL_ENV, "1")
    assert noncommercial_allowed(False) is False


def test_permissive_allowlist_normalizes_spellings() -> None:
    assert is_permissive_license("Apache-2.0")
    assert is_permissive_license("apache 2.0")
    assert is_permissive_license("MIT")
    assert is_permissive_license("BSD-3-Clause")
    assert not is_permissive_license("CC-BY-NC-4.0")
    assert not is_permissive_license("GPL-3.0")
    assert not is_permissive_license("unknown")


def test_smoke_config_runs_the_local_csv_fixture() -> None:
    config = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    assert dataset["loader"] == "local_csv"
    fixture = (REPO_ROOT / dataset["path"]).resolve()
    assert fixture.is_file()
    assert fixture.parent == (REPO_ROOT / "tests" / "fixtures").resolve()
    assert config["models"] == ["naive"]
    assert config["context_length"] > 0
    assert config["horizon"] > 0

    rows = list(csv.DictReader(fixture.open(encoding="utf-8")))
    assert list(rows[0]) == ["timestamp", "value"]
    assert len(rows) >= config["context_length"] + config["horizon"]
    values = [float(row["value"]) for row in rows]
    assert all(value != 0.0 for value in values)
    stamps = [dt.date.fromisoformat(row["timestamp"]) for row in rows]
    assert stamps == sorted(stamps)


def test_cli_licenses_lists_every_registered_model(
    capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    assert main(["licenses"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    for key in FROZEN_MODELS:
        assert key in captured.out
    assert "Apache-2.0" in captured.out
    assert "TBD-at-download" in captured.out


def test_cli_run_rejects_a_missing_config(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    exit_code = main(["run", "--config", str(tmp_path / "absent.yaml")])
    assert exit_code == 2
    assert "not found" in capsys.readouterr().err
