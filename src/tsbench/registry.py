"""Model registry and license gate for tsfm-benchmark.

``configs/models.yaml`` is the single source of truth for model entrypoints
and their licenses. Loading validates the manifest; instantiation enforces the
license gate.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .base import Forecaster

__all__ = [
    "ALLOWED_FAMILIES",
    "ALLOW_NONCOMMERCIAL_ENV",
    "DEFAULT_MODELS_CONFIG",
    "PERMISSIVE_LICENSES",
    "LicenseError",
    "ModelConfigError",
    "ModelImportError",
    "ModelRegistry",
    "ModelSpec",
    "RegistryError",
    "check_license",
    "is_permissive_license",
    "load_registry",
    "noncommercial_allowed",
]

ALLOW_NONCOMMERCIAL_ENV = "TSBENCH_ALLOW_NONCOMMERCIAL"
DEFAULT_MODELS_CONFIG = Path("configs") / "models.yaml"

ALLOWED_FAMILIES = frozenset({"baseline", "classical", "ml", "deep", "tsfm"})

PERMISSIVE_LICENSES = frozenset(
    {
        "0bsd",
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc-by-4.0",
        "cc0-1.0",
        "isc",
        "mit",
        "unlicense",
    }
)

_REQUIRED_FIELDS = ("entrypoint", "family", "zero_shot", "license", "revision")
_OPTIONAL_FIELDS = ("params", "weights", "notes")
_FORECASTER_ATTRIBUTES = ("name", "fit", "predict", "info")
_FORECASTER_METHODS = ("fit", "predict", "info")


class RegistryError(RuntimeError):
    """Base class for registry and license-gate failures."""


class ModelConfigError(RegistryError):
    """A model manifest is malformed or missing required fields."""


class ModelImportError(RegistryError):
    """A model entrypoint cannot be imported or is not a Forecaster."""


class LicenseError(RegistryError):
    """A model license blocks instantiation and no override is set."""


def normalize_license(value: str) -> str:
    """Return a canonical SPDX-ish spelling used for allowlist comparison."""
    return "-".join(str(value).strip().lower().replace("_", "-").split())


def is_permissive_license(value: str) -> bool:
    """True when ``value`` is on the explicit permissive allowlist."""
    return normalize_license(value) in PERMISSIVE_LICENSES


def noncommercial_allowed(override: bool | None = None) -> bool:
    """True when non-permissive licenses may run (argument or environment)."""
    if override is not None:
        return bool(override)
    raw = os.environ.get(ALLOW_NONCOMMERCIAL_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelSpec:
    """One validated entry from the model manifest."""

    key: str
    entrypoint: str
    family: str
    zero_shot: bool
    license: str
    revision: str
    params: int | float | str | None = None
    weights: str | None = None
    notes: str | None = None

    @property
    def module(self) -> str:
        return self.entrypoint.split(":", 1)[0]

    @property
    def attribute(self) -> str:
        return self.entrypoint.split(":", 1)[1]

    @property
    def permissive(self) -> bool:
        return is_permissive_license(self.license)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.key,
            "entrypoint": self.entrypoint,
            "family": self.family,
            "zero_shot": self.zero_shot,
            "params": self.params,
            "license": self.license,
            "revision": self.revision,
            "weights": self.weights,
            "permissive": self.permissive,
        }


def check_license(spec: ModelSpec, *, allow_noncommercial: bool | None = None) -> None:
    """Raise ``LicenseError`` unless the spec's license is allowed to run."""
    if spec.permissive or noncommercial_allowed(allow_noncommercial):
        return
    raise LicenseError(
        f"model {spec.key!r}: license {spec.license!r} is not on the permissive "
        f"allowlist ({', '.join(sorted(PERMISSIVE_LICENSES))}); set "
        f"{ALLOW_NONCOMMERCIAL_ENV}=1 to instantiate it anyway"
    )


def _parse_spec(key: str, raw: Any) -> ModelSpec:
    if not isinstance(raw, Mapping):
        raise ModelConfigError(
            f"model {key!r}: entry must be a mapping, got {type(raw).__name__}"
        )
    missing = [
        name
        for name in _REQUIRED_FIELDS
        if name not in raw
        or raw[name] is None
        or (isinstance(raw[name], str) and not raw[name].strip())
    ]
    if missing:
        raise ModelConfigError(
            f"model {key!r}: missing required field(s): {', '.join(missing)}; "
            "every model must declare a license and a revision"
        )
    unknown = sorted(set(raw) - set(_REQUIRED_FIELDS) - set(_OPTIONAL_FIELDS))
    if unknown:
        raise ModelConfigError(
            f"model {key!r}: unknown field(s): {', '.join(unknown)} "
            f"(allowed: {', '.join(_REQUIRED_FIELDS + _OPTIONAL_FIELDS)})"
        )
    entrypoint = str(raw["entrypoint"])
    if ":" not in entrypoint:
        raise ModelConfigError(
            f"model {key!r}: entrypoint must look like 'module:attribute', got {entrypoint!r}"
        )
    module, _, attribute = entrypoint.partition(":")
    if not module.strip() or not attribute.strip():
        raise ModelConfigError(
            f"model {key!r}: entrypoint must look like 'module:attribute', got {entrypoint!r}"
        )
    family = str(raw["family"])
    if family not in ALLOWED_FAMILIES:
        raise ModelConfigError(
            f"model {key!r}: family must be one of {sorted(ALLOWED_FAMILIES)}, got {family!r}"
        )
    zero_shot = raw["zero_shot"]
    if not isinstance(zero_shot, bool):
        raise ModelConfigError(
            f"model {key!r}: zero_shot must be a boolean, got {zero_shot!r}"
        )
    return ModelSpec(
        key=key,
        entrypoint=entrypoint,
        family=family,
        zero_shot=zero_shot,
        license=str(raw["license"]).strip(),
        revision=str(raw["revision"]).strip(),
        params=raw.get("params"),
        weights=raw.get("weights"),
        notes=raw.get("notes"),
    )


def _import_entrypoint(spec: ModelSpec) -> Any:
    try:
        module = importlib.import_module(spec.module)
    except Exception as exc:
        raise ModelImportError(
            f"model {spec.key!r}: cannot import module {spec.module!r}: {exc}"
        ) from exc
    try:
        return getattr(module, spec.attribute)
    except AttributeError as exc:
        raise ModelImportError(
            f"model {spec.key!r}: module {spec.module!r} has no attribute "
            f"{spec.attribute!r}"
        ) from exc


class ModelRegistry:
    """Validated view over a model manifest with a license-gated factory."""

    def __init__(
        self, specs: Mapping[str, ModelSpec], source: str | Path | None = None
    ) -> None:
        self._specs = dict(specs)
        self.source = Path(source) if source is not None else None

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self._specs.values())

    def __contains__(self, key: object) -> bool:
        return key in self._specs

    def keys(self) -> list[str]:
        return list(self._specs)

    def spec(self, name: str) -> ModelSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise RegistryError(
                f"unknown model {name!r}; available: {', '.join(sorted(self._specs))}"
            ) from exc

    def entrypoint(self, name: str) -> Any:
        return _import_entrypoint(self.spec(name))

    def instantiate(
        self,
        name: str,
        *,
        allow_noncommercial: bool | None = None,
        **kwargs: Any,
    ) -> Forecaster:
        spec = self.spec(name)
        check_license(spec, allow_noncommercial=allow_noncommercial)
        target = _import_entrypoint(spec)
        if not callable(target):
            raise ModelImportError(
                f"model {name!r}: entrypoint {spec.entrypoint!r} is not callable"
            )
        model = target(**kwargs) if kwargs else target()
        missing = [attr for attr in _FORECASTER_ATTRIBUTES if not hasattr(model, attr)]
        if missing:
            raise ModelImportError(
                f"model {name!r}: entrypoint {spec.entrypoint!r} is missing "
                f"Forecaster attribute(s): {', '.join(missing)}"
            )
        not_callable = [
            attr for attr in _FORECASTER_METHODS if not callable(getattr(model, attr))
        ]
        if not_callable:
            raise ModelImportError(
                f"model {name!r}: Forecaster attribute(s) not callable: "
                f"{', '.join(not_callable)}"
            )
        return model

    def manifest(self) -> list[dict[str, Any]]:
        return [spec.as_dict() for spec in self._specs.values()]


def load_registry(path: str | Path | None = None) -> ModelRegistry:
    """Load and validate a model manifest; default is ``configs/models.yaml``."""
    config_path = Path(path) if path is not None else DEFAULT_MODELS_CONFIG
    if not config_path.is_file():
        raise ModelConfigError(f"model config not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ModelConfigError(f"{config_path}: invalid YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ModelConfigError(f"{config_path}: expected a mapping at the document root")
    entries = data.get("models")
    if not isinstance(entries, Mapping):
        raise ModelConfigError(f"{config_path}: expected a top-level 'models:' mapping")
    specs = {str(key): _parse_spec(str(key), raw) for key, raw in entries.items()}
    if not specs:
        raise ModelConfigError(f"{config_path}: no models declared")
    return ModelRegistry(specs, source=config_path)
