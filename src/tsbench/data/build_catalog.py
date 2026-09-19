"""Build the pinned dataset catalogue from ``configs/datasets.yaml``.

The catalogue is intentionally dataset-specific: the primary electricity demand
source is Monash's *Electricity Hourly* dataset (CC-BY-4.0), which is itself an
aggregation of the UCI ElectricityLoadDiagrams20112014 data set. The loader is
the same Monash ``.tsf`` reader used everywhere else, so the primary dataset and
the smoke fixture share a code path.

Run ``python -m tsbench.data.build_catalog`` to print the pinned manifest
without touching the network. Downloads happen only through ``--materialize``
and are checksum-verified, never assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import (
    DEFAULT_DATASETS_CONFIG,
    DatasetSpec,
    ensure_dataset,
    load_catalog,
    resolve_member_path,
    verify_local_dataset,
)
from .loaders import ChecksumError, DataError, sha256_file

__all__ = ["N_SERIES", "build_report", "main", "materialize", "record_series", "select_series"]

# A deterministic, evenly-spaced spread across the 321 Monash electricity
# series. Sampling by index (not by name) means the selection is stable even if
# a future release relabels series; the frozen manifest records the resolved
# ids and their per-series digests.
N_SERIES = 24


def select_series(all_ids: list[str], n: int = N_SERIES) -> list[str]:
    """Pick ``n`` series ids spread deterministically across ``all_ids``."""
    if not all_ids:
        raise ValueError("no series available to select from")
    if n >= len(all_ids):
        return list(all_ids)
    step = len(all_ids) / n
    indices = sorted({int(i * step) for i in range(n)})
    while len(indices) < n:
        for candidate in range(len(all_ids)):
            if candidate not in indices:
                indices.append(candidate)
                indices.sort()
                if len(indices) == n:
                    break
    return [all_ids[i] for i in indices[:n]]


def _series_sha256(values) -> str:
    import hashlib

    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(array.tobytes())
    return digest.hexdigest()


def _record_from_path(spec: DatasetSpec, path: Path) -> dict:
    series = _load_selected(spec, path)
    return {
        "dataset": spec.name,
        "status": "OK",
        "loader": spec.loader,
        "series_count": len(series),
        "series_length": len(series[0]),
        "series": [
            {
                "series_id": s.series_id,
                "length": len(s),
                "value_sha256": _series_sha256(s.values),
            }
            for s in series
        ],
    }


def build_report(
    config: str | Path = DEFAULT_DATASETS_CONFIG,
    *,
    root: str | Path = ".",
) -> dict:
    """Return the provenance report for every dataset in the catalogue."""
    specs, metadata = load_catalog(config)
    datasets: dict[str, object] = {}
    for name, spec in specs.items():
        report = verify_local_dataset(spec, root=root)
        report["url"] = spec.url
        report["license_url"] = spec.license_url
        report["split_manifest"] = spec.split_manifest
        datasets[name] = report
    return {"version": metadata.get("version"), "datasets": datasets}


def _load_selected(spec: DatasetSpec, path: Path):
    if spec.loader == "monash_tsf":
        from .loaders import load_monash_tsf

        available = load_monash_tsf(path)
        ids = select_series([s.series_id for s in available])
        return load_monash_tsf(path, series_ids=ids)
    if spec.loader == "local_csv":
        from .loaders import load_local_csv

        return [load_local_csv(path, series_id=spec.extra.get("series_id", Path(path).stem))]
    raise DataError(f"dataset {spec.name!r}: loader {spec.loader!r} has no series builder")


def record_series(datasets_config: str | Path, name: str, *, root: str | Path = ".") -> dict:
    """Compute the frozen-series digest block for one dataset."""
    specs, _ = load_catalog(datasets_config)
    if name not in specs:
        raise DataError(f"unknown dataset {name!r}")
    spec = specs[name]
    path = resolve_member_path(spec, root=root)
    if path is None or not path.is_file():
        return {
            "dataset": name,
            "status": "PRELIMINARY",
            "reason": "dataset file absent; run with --materialize to fetch it",
        }
    return _record_from_path(spec, path)


def materialize(config: str | Path, *, root: str | Path = ".") -> dict:
    """Download and verify every catalogue dataset, reporting per-dataset status.

    Each entry records the local path and file sha256, or is marked
    ``preliminary`` when the fetch was not possible.
    """
    specs, metadata = load_catalog(config)
    out: dict[str, object] = {"version": metadata.get("version"), "datasets": {}}
    for name, spec in specs.items():
        target = ensure_dataset(spec, root=root, allow_download=True)
        entry = {
            "path": str(target) if target is not None else None,
            "sha256": None,
            "preliminary": False,
        }
        if target is not None and target.is_file():
            entry["sha256"] = sha256_file(target)
        else:
            entry["preliminary"] = True
        out["datasets"][name] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsbench.data.build_catalog")
    parser.add_argument("--config", type=Path, default=DEFAULT_DATASETS_CONFIG)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="download and checksum-verify datasets (network required)",
    )
    parser.add_argument(
        "--record",
        metavar="DATASET",
        default=None,
        help="print the frozen per-series digest block for one dataset",
    )
    args = parser.parse_args(argv)
    try:
        if args.record:
            payload = record_series(args.config, args.record, root=args.root)
        elif args.materialize:
            payload = materialize(args.config, root=args.root)
        else:
            payload = build_report(args.config, root=args.root)
    except (DataError, ChecksumError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
