# Data provenance

This file records how the primary dataset was obtained and how its frozen split
is committed. The authoritative machine-readable record lives in
`configs/datasets.yaml`; this file is the human-readable audit trail.

## Primary dataset — Monash Electricity Hourly

| Field | Value |
| --- | --- |
| Source | Monash Time Series Forecasting Repository |
| Record | https://zenodo.org/records/4656140 |
| Download | https://zenodo.org/records/4656140/files/electricity_hourly_dataset.zip?download=1 |
| License | CC-BY-4.0 |
| License URL | https://creativecommons.org/licenses/by/4.0/ |
| Content | 321 univariate hourly series, 2012-01-01 to 2014-12-31 (26,304 points each) |
| Upstream | UCI ElectricityLoadDiagrams20112014 (also CC-BY-4.0), aggregated by Lai et al. (2017) |

### Checksums (verified on download)

| Artifact | Algorithm | Digest |
| --- | --- | --- |
| `electricity_hourly_dataset.zip` | md5 (Zenodo-published) | `18096614662b02640d265ad2a6a416bd` |
| `electricity_hourly_dataset.zip` | sha256 | `eff447075dde68dca0105ab7e2851c5637967ae3bb21556fd8b931f196d5968c` |
| `electricity_hourly_dataset.tsf` | sha256 | `bc33039133b9f1ca2e73d6b58d97306dd063f865fd12fbb0920ab312f047ad28` |

The download was verified against Zenodo's published md5 and a locally computed
sha256 was pinned at the same time. `python -m tsbench.data.build_catalog
--materialize` re-verifies both, and refuses to proceed on a mismatch.

## Frozen split

The split manifest is committed at
`data/manifests/electricity_hourly_split_manifest.json` (and materialized at the
runtime path `data/electricity_hourly/split_manifest.json`). It records, for
each of the 24 evenly-spaced evaluation series (T1, T14, T27, ... T308):

* series length and a sha256 over the raw float64 value block;
* the exclusive `train_end` / `val_end` / `test_end` boundaries; and
* a sha256 over the three values immediately preceding each boundary.

`data/manifests/checksums.sha256` pins the manifest file itself. The runner
verifies the manifest against the on-disk series before producing any result, so
a changed series or a tampered split is a hard failure rather than a silently
different experiment.

Split geometry: 60% / 20% / 20% train/val/test, `context_length=168`,
`horizon=24`, `stride=24`. Windows are enumerated at run time from these frozen
boundaries and never stored; no window reads a value at or after its target
block (enforced by `tsbench.data.splits.assert_no_future_data` and the leakage
tests).

## Regenerating

```sh
# network required; verifies checksums before use
python -m tsbench.data.build_catalog --materialize

# offline provenance report
python -m tsbench.data.build_catalog
```
