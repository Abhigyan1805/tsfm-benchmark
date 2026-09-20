"""Dataset loaders for tsfm-benchmark.

Two formats are supported:

* ``local_csv`` -- a single column of timestamps and a single column of values.
  This is the smoke-test path and the format every loader ultimately normalises
  to.
* ``monash_tsf`` -- the ``.tsf`` format used by the Monash Time Series
  Forecasting Repository (one line per series,
  ``name:start_timestamp:v1,v2,...``).

Every loader returns one or more :class:`Series` objects. A ``Series`` is a
plain value object: a stable ``series_id``, an aligned timestamp array, and a
finite float64 value array. Loaders never impute, resample, or fabricate data;
malformed input raises :class:`ParseError`.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:  # pandas is a hard dependency of the project, but keep the import lazy-friendly.
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is declared in pyproject
    pd = None  # type: ignore[assignment]

__all__ = [
    "ChecksumError",
    "DataError",
    "DownloadError",
    "ParseError",
    "Series",
    "extract_zip_member",
    "load_local_csv",
    "load_monash_tsf",
    "md5_file",
    "parse_tsf_text",
    "sha256_file",
    "verify_checksums",
]


class DataError(RuntimeError):
    """Base class for data-layer failures."""


class ParseError(DataError):
    """A dataset file is malformed or does not match its declared schema."""


class ChecksumError(DataError):
    """A downloaded artifact does not match its pinned checksum."""


class DownloadError(DataError):
    """A dataset could not be fetched; callers must mark it PRELIMINARY."""


@dataclass(frozen=True)
class Series:
    """One univariate time series with aligned timestamps."""

    series_id: str
    timestamps: np.ndarray
    values: np.ndarray
    frequency: str | None = None

    def __len__(self) -> int:
        return int(self.values.size)

    @property
    def n_observations(self) -> int:
        return int(self.values.size)


def _finalize_series(
    series_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    frequency: str | None = None,
) -> Series:
    if not series_id:
        raise ParseError("series_id must be a non-empty string")
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ParseError(f"series {series_id!r}: no observations")
    if not np.all(np.isfinite(array)):
        raise ParseError(f"series {series_id!r}: values must be finite")
    stamps = np.asarray(timestamps).reshape(-1)
    if stamps.size != array.size:
        raise ParseError(
            f"series {series_id!r}: {stamps.size} timestamps for {array.size} values"
        )
    return Series(
        series_id=str(series_id),
        timestamps=stamps,
        values=array.astype(np.float64, copy=True),
        frequency=frequency,
    )


def _parse_timestamps(column: object) -> np.ndarray:
    """Parse a CSV timestamp column to datetime64 when possible."""
    if pd is None:  # pragma: no cover - defensive
        return np.asarray(column)
    try:
        return pd.to_datetime(column).to_numpy()
    except (ValueError, TypeError, OverflowError):
        return np.asarray(column)


def load_local_csv(
    path: str | Path,
    *,
    series_id: str | None = None,
    timestamp_column: str = "timestamp",
    value_column: str = "value",
) -> Series:
    """Load a single ``timestamp,value`` CSV into a :class:`Series`.

    ``timestamp_column`` and ``value_column`` name the columns to read. The
    series id defaults to the file stem when not supplied.
    """
    if pd is None:  # pragma: no cover - defensive
        raise DataError("pandas is required to read CSV datasets")
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"dataset not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    missing = [c for c in (timestamp_column, value_column) if c not in frame.columns]
    if missing:
        raise ParseError(
            f"{csv_path}: missing column(s) {missing}; found {list(frame.columns)}"
        )
    if frame.empty:
        raise ParseError(f"{csv_path}: file contains no rows")
    return _finalize_series(
        series_id or csv_path.stem,
        _parse_timestamps(frame[timestamp_column]),
        frame[value_column].to_numpy(),
    )


# Monash .tsf frequencies spelled out in the header, mapped to pandas offsets.
_TSF_FREQUENCIES: dict[str, str] = {
    "yearly": "YS",
    "quarterly": "QS",
    "monthly": "MS",
    "weekly": "W",
    "daily": "D",
    "hourly": "h",
    "half_hourly": "30min",
    "minutely": "min",
    "10_minutely": "10min",
    "5_minutely": "5min",
}

_TSF_TIMESTAMP_FORMAT = "%Y-%m-%d %H-%M-%S"


def _tsf_timestamps(start: str, n_obs: int, frequency: str, series_id: str) -> np.ndarray:
    offset = _TSF_FREQUENCIES.get(frequency.lower())
    if pd is None or offset is None:
        return np.arange(n_obs, dtype=np.int64)
    try:
        begin = datetime.strptime(start.strip(), _TSF_TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise ParseError(
            f"series {series_id!r}: cannot parse start timestamp {start!r}"
        ) from exc
    return pd.date_range(begin, periods=n_obs, freq=offset).to_numpy()


def parse_tsf_text(text: str, *, source: str = "<string>") -> list[Series]:
    """Parse Monash ``.tsf`` content into a list of :class:`Series`."""
    frequency = ""
    in_data = False
    series: list[Series] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if not in_data:
            if line.startswith("@"):
                directive, _, argument = line[1:].partition(" ")
                if directive.lower() == "frequency":
                    frequency = argument.strip()
                elif directive.lower() == "data":
                    in_data = True
                continue
            continue  # comments and blank lines before @data
        if line.startswith("#"):
            continue
        parts = line.split(":", 2)
        if len(parts) != 3:
            raise ParseError(f"{source}:{lineno}: malformed data row")
        series_id, start, payload = (part.strip() for part in parts)
        try:
            values = np.array(payload.split(","), dtype=np.float64)
        except ValueError as exc:
            raise ParseError(f"{source}:{lineno}: non-numeric values") from exc
        if values.size == 0:
            raise ParseError(f"{source}:{lineno}: series {series_id!r} has no values")
        series.append(
            _finalize_series(
                series_id,
                _tsf_timestamps(start, values.size, frequency, series_id),
                values,
                frequency=frequency or None,
            )
        )
    if not series:
        raise ParseError(f"{source}: no series found (missing @data section?)")
    return series


def load_monash_tsf(
    path: str | Path,
    *,
    series_ids: Sequence[str] | None = None,
    max_series: int | None = None,
) -> list[Series]:
    """Load selected series from a Monash ``.tsf`` file.

    ``series_ids`` selects (and orders) explicit series; ``max_series`` caps how
    many are returned when no explicit selection is given.
    """
    tsf_path = Path(path)
    if not tsf_path.is_file():
        raise FileNotFoundError(f"dataset not found: {tsf_path}")
    series = parse_tsf_text(tsf_path.read_text(encoding="utf-8"), source=str(tsf_path))
    if series_ids is not None:
        wanted = {str(sid): True for sid in series_ids}
        by_id = {s.series_id: s for s in series}
        missing = [sid for sid in wanted if sid not in by_id]
        if missing:
            raise ParseError(f"{tsf_path}: unknown series id(s): {missing[:10]}")
        return [by_id[str(sid)] for sid in series_ids]
    if max_series is not None:
        if max_series < 1:
            raise ValueError("max_series must be a positive integer")
        return series[:max_series]
    return series


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex md5 of a file (used only for archive provenance)."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(
    path: str | Path,
    *,
    sha256: str | None = None,
    md5: str | None = None,
) -> None:
    """Raise :class:`ChecksumError` when ``path`` fails a pinned checksum."""
    target = Path(path)
    if sha256 is not None:
        actual = sha256_file(target)
        if actual.lower() != str(sha256).strip().lower():
            raise ChecksumError(
                f"{target}: sha256 mismatch (expected {sha256}, got {actual})"
            )
    if md5 is not None:
        actual_md5 = md5_file(target)
        if actual_md5.lower() != str(md5).strip().lower():
            raise ChecksumError(
                f"{target}: md5 mismatch (expected {md5}, got {actual_md5})"
            )


def download_file(url: str, dest: str | Path, *, timeout: float = 120.0) -> Path:
    """Download ``url`` to ``dest`` atomically; raises :class:`DownloadError`."""
    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "tsbench/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"cannot download {url}: {exc}") from exc
    partial.replace(destination)
    return destination


def extract_zip_member(archive: str | Path, member: str, dest: str | Path) -> Path:
    """Extract ``member`` from a zip ``archive`` to ``dest``."""
    archive_path = Path(archive)
    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            names = bundle.namelist()
            if member not in names:
                raise ParseError(f"{archive_path}: no member {member!r} (has {names})")
            with bundle.open(member) as source, open(destination, "wb") as handle:
                shutil.copyfileobj(source, handle)
    except zipfile.BadZipFile as exc:
        raise ParseError(f"{archive_path}: not a valid zip archive") from exc
    return destination


def iter_values(series: Iterable[Series]) -> np.ndarray:
    """Concatenate every series' values (used by integrity digests)."""
    chunks = [np.asarray(s.values, dtype=np.float64) for s in series]
    if not chunks:
        raise ValueError("no series to concatenate")
    return np.concatenate(chunks)
