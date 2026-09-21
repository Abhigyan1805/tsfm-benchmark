"""Dataset catalogue: pinned sources, licenses, checksums, and caching.

``configs/datasets.yaml`` is the single source of truth for dataset
provenance. This module parses it, resolves local cache paths, and downloads /
verifies archives only when a caller explicitly permits it. Download failure is
reported, never papered over: callers mark the dataset ``PRELIMINARY`` instead
of fabricating observations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .loaders import (
    DataError,
    DownloadError,
    download_file,
    extract_zip_member,
    sha256_file,
    verify_checksums,
)

__all__ = [
    "DEFAULT_DATASETS_CONFIG",
    "CatalogError",
    "DatasetSpec",
    "ensure_dataset",
    "load_catalog",
]

DEFAULT_DATASETS_CONFIG = Path("configs") / "datasets.yaml"

_KNOWN_LOADERS = frozenset({"local_csv", "monash_tsf"})
_REQUIRED_FIELDS = ("loader", "license")


class CatalogError(DataError):
    """``datasets.yaml`` is malformed or references an undeclared dataset."""


@dataclass(frozen=True)
class DatasetSpec:
    """One validated entry from the dataset catalogue."""

    name: str
    loader: str
    license: str
    path: str | None = None
    url: str | None = None
    archive_filename: str | None = None
    member_filename: str | None = None
    download_dir: str | None = None
    license_url: str | None = None
    sha256: str | None = None
    md5: str | None = None
    member_sha256: str | None = None
    format: str | None = None
    frequency: str | None = None
    series_count: int | None = None
    series_length: int | None = None
    split_manifest: str | None = None
    preliminary: bool = False
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_preliminary(self) -> bool:
        return bool(self.preliminary) or (self.url is not None and self.sha256 is None)

    @property
    def cache_dir(self) -> str:
        return self.download_dir or f"data/{self.name}"

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "loader": self.loader,
            "license": self.license,
            "path": self.path,
            "url": self.url,
            "archive_filename": self.archive_filename,
            "member_filename": self.member_filename,
            "download_dir": self.download_dir,
            "license_url": self.license_url,
            "sha256": self.sha256,
            "md5": self.md5,
            "member_sha256": self.member_sha256,
            "format": self.format,
            "frequency": self.frequency,
            "series_count": self.series_count,
            "series_length": self.series_length,
            "split_manifest": self.split_manifest,
            "preliminary": self.preliminary,
            "notes": self.notes,
        }
        payload.update(self.extra)
        return payload


def _parse_spec(name: str, raw: Any) -> DatasetSpec:
    if not isinstance(raw, Mapping):
        raise CatalogError(f"dataset {name!r}: entry must be a mapping")
    missing = [f for f in _REQUIRED_FIELDS if not raw.get(f)]
    if missing:
        raise CatalogError(
            f"dataset {name!r}: missing required field(s): {', '.join(missing)}"
        )
    loader = str(raw["loader"]).strip()
    if loader not in _KNOWN_LOADERS:
        raise CatalogError(
            f"dataset {name!r}: unknown loader {loader!r} "
            f"(known: {', '.join(sorted(_KNOWN_LOADERS))})"
        )
    known = {
        "loader",
        "license",
        "path",
        "url",
        "archive_filename",
        "member_filename",
        "download_dir",
        "license_url",
        "sha256",
        "md5",
        "member_sha256",
        "format",
        "frequency",
        "series_count",
        "series_length",
        "split_manifest",
        "preliminary",
        "notes",
    }
    extra = {k: v for k, v in raw.items() if k not in known}
    return DatasetSpec(
        name=name,
        loader=loader,
        license=str(raw["license"]).strip(),
        path=raw.get("path"),
        url=raw.get("url"),
        archive_filename=raw.get("archive_filename"),
        member_filename=raw.get("member_filename"),
        download_dir=raw.get("download_dir"),
        license_url=raw.get("license_url"),
        sha256=raw.get("sha256"),
        md5=raw.get("md5"),
        member_sha256=raw.get("member_sha256"),
        format=raw.get("format"),
        frequency=raw.get("frequency"),
        series_count=raw.get("series_count"),
        series_length=raw.get("series_length"),
        split_manifest=raw.get("split_manifest"),
        preliminary=bool(raw.get("preliminary", False)),
        notes=raw.get("notes"),
        extra=extra,
    )


def load_catalog(
    path: str | Path | None = None,
) -> tuple[dict[str, DatasetSpec], dict[str, Any]]:
    """Load and validate the dataset catalogue.

    Returns ``(datasets, metadata)`` where ``metadata`` holds the top-level
    document context (``default_dataset``, ``version``, ...).
    """
    config_path = Path(path) if path is not None else DEFAULT_DATASETS_CONFIG
    if not config_path.is_file():
        raise CatalogError(f"dataset catalogue not found: {config_path}")
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CatalogError(f"{config_path}: invalid YAML: {exc}") from exc
    if not isinstance(document, Mapping):
        raise CatalogError(f"{config_path}: expected a mapping at the document root")
    entries = document.get("datasets")
    if not isinstance(entries, Mapping) or not entries:
        raise CatalogError(f"{config_path}: expected a non-empty top-level 'datasets:'")
    specs = {str(k): _parse_spec(str(k), v) for k, v in entries.items()}
    metadata = {k: v for k, v in document.items() if k != "datasets"}
    default = metadata.get("default_dataset")
    if default is not None and str(default) not in specs:
        raise CatalogError(f"{config_path}: default_dataset {default!r} is not declared")
    return specs, metadata


def _resolve_spec_path(spec: DatasetSpec, root: Path) -> Path | None:
    if spec.path is not None:
        return root / spec.path
    if spec.member_filename is not None:
        return root / spec.cache_dir / spec.member_filename
    return None


def ensure_dataset(
    spec: DatasetSpec,
    *,
    root: str | Path = ".",
    allow_download: bool = True,
    verify: bool = True,
) -> Path | None:
    """Return a local path to ``spec``'s data, downloading when permitted.

    Recoverable failures return ``None`` so the caller can surface the dataset
    as PRELIMINARY: the data is absent, downloads are not permitted, or the
    fetch itself fails (such as no network, a :class:`DownloadError`). Fatal
    failures raise instead: present-but-tampered data or a downloaded archive
    failing pinned verification (:class:`ChecksumError`) must never be treated
    as acceptable and silently marked preliminary.
    """
    base = Path(root)
    target = _resolve_spec_path(spec, base)
    if target is None:
        raise CatalogError(f"dataset {spec.name!r}: no path or member_filename declared")
    if target.is_file():
        if verify and spec.member_sha256:
            verify_checksums(target, sha256=spec.member_sha256)
        return target
    if not allow_download or spec.url is None:
        return None
    archive_dir = base / spec.cache_dir
    archive_name = spec.archive_filename or Path(spec.url).name
    archive = archive_dir / archive_name
    try:
        if not archive.is_file():
            download_file(spec.url, archive)
        if verify and (spec.sha256 or spec.md5):
            verify_checksums(archive, sha256=spec.sha256, md5=spec.md5)
        if spec.member_filename is None:
            return archive
        extract_zip_member(archive, spec.member_filename, target)
        if verify and spec.member_sha256:
            verify_checksums(target, sha256=spec.member_sha256)
    except DownloadError:
        return None
    return target


def resolve_member_path(spec: DatasetSpec, *, root: str | Path = ".") -> Path | None:
    """Return the expected local path for a dataset without touching the network."""
    return _resolve_spec_path(spec, Path(root))


def verify_local_dataset(spec: DatasetSpec, *, root: str | Path = ".") -> dict[str, Any]:
    """Verify an already-downloaded dataset and report provenance.

    Never downloads. The result carries ``preliminary=True`` when the data is
    absent or its pinned member checksum does not match, so callers can surface
    the gap instead of pretending the split is frozen. When there is no member
    checksum to check, the catalogue's own ``preliminary`` declaration decides:
    a committed local fixture is not marked preliminary just because it has no
    sha256 pin, while a remote source without one still is.
    """
    target = _resolve_spec_path(spec, Path(root))
    report: dict[str, Any] = {
        "dataset": spec.name,
        "loader": spec.loader,
        "license": spec.license,
        "expected_path": str(target) if target is not None else None,
        "present": bool(target and target.is_file()),
        "preliminary": True,
        "sha256": None,
    }
    if target is not None and target.is_file():
        report["sha256"] = sha256_file(target)
        if spec.member_sha256:
            expected = str(spec.member_sha256).lower()
            report["preliminary"] = report["sha256"] != expected
        else:
            report["preliminary"] = spec.is_preliminary
    return report
