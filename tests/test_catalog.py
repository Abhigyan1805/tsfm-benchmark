"""Dataset catalogue download-failure contract.

``ensure_dataset`` must distinguish recoverable fetch failures (no network, no
permission) from fatal integrity failures (tampered/mismatched data): the
former returns ``None`` so the caller can surface the dataset as PRELIMINARY,
the latter raises so a bad artifact is never silently accepted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tsbench.data import catalog
from tsbench.data.catalog import DatasetSpec, ensure_dataset
from tsbench.data.loaders import ChecksumError, DownloadError


def _spec(**overrides) -> DatasetSpec:
    payload = {
        "name": "demo",
        "loader": "local_csv",
        "license": "CC0-1.0",
        "url": "https://example.invalid/demo.zip",
        "archive_filename": "demo.zip",
        "member_filename": "demo.csv",
        "sha256": "0" * 64,
    }
    payload.update(overrides)
    return DatasetSpec(**payload)


def test_ensure_dataset_returns_none_when_download_fails(tmp_path, monkeypatch):
    def failing_download(url, dest, **kwargs):
        raise DownloadError(f"cannot download {url}")

    monkeypatch.setattr(catalog, "download_file", failing_download)

    assert ensure_dataset(_spec(), root=tmp_path) is None


def test_ensure_dataset_raises_on_checksum_mismatch(tmp_path, monkeypatch):
    def tampered_download(url, dest, **kwargs):
        destination = Path(dest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"tampered bytes")
        return destination

    monkeypatch.setattr(catalog, "download_file", tampered_download)

    with pytest.raises(ChecksumError):
        ensure_dataset(_spec(), root=tmp_path)


def test_verify_local_dataset_defers_to_the_catalogue_preliminary_flag(tmp_path):
    """A local fixture with no sha256 pin is not preliminary by default."""
    from tsbench.data.catalog import verify_local_dataset

    fixture = tmp_path / "series.csv"
    fixture.write_text("timestamp,value\n2020-01-01,1.0\n", encoding="utf-8")
    committed = DatasetSpec(
        name="smoke",
        loader="local_csv",
        license="Apache-2.0",
        path="series.csv",
        preliminary=False,
    )
    report = verify_local_dataset(committed, root=tmp_path)
    assert report["present"] is True
    assert report["preliminary"] is False

    # A remote source with no pinned sha256 stays preliminary.
    remote = DatasetSpec(
        name="remote",
        loader="monash_tsf",
        license="CC-BY-4.0",
        url="https://example.invalid/x.zip",
        member_filename="x.tsf",
    )
    report = verify_local_dataset(remote, root=tmp_path)
    assert report["preliminary"] is True


def test_materialize_marks_fetch_failure_preliminary(tmp_path, monkeypatch):
    from tsbench.data import build_catalog

    def failing_download(url, dest, **kwargs):
        raise DownloadError(f"cannot download {url}")

    monkeypatch.setattr(catalog, "download_file", failing_download)
    config = tmp_path / "datasets.yaml"
    config.write_text(
        "datasets:\n"
        "  demo:\n"
        "    loader: local_csv\n"
        "    license: CC0-1.0\n"
        "    url: https://example.invalid/demo.zip\n"
        "    archive_filename: demo.zip\n"
        "    member_filename: demo.csv\n"
        "    sha256: " + "0" * 64 + "\n",
        encoding="utf-8",
    )

    report = build_catalog.materialize(config, root=tmp_path)

    entry = report["datasets"]["demo"]
    assert entry["preliminary"] is True
    assert entry["path"] is None
    assert entry["sha256"] is None
