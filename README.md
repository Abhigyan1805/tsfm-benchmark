# tsfm-benchmark

**Research question:** when do time-series foundation models (TSFMs) outperform conventional forecasting models, and what is the accuracy/latency trade-off?

A reproducible, leakage-audited head-to-head of classical, ML, deep, and foundation
forecasters across forecasting horizons — measured on accuracy *and* inference cost,
with a partial reproduction of the TimesFM evaluation methodology, producing an
accuracy-vs-cost Pareto frontier rather than a single leaderboard row.

**Status:** P0 foundation, the local model tier, the GPU tier, and the data/evaluation
spine are in place — packaging, configs, the license-gated model registry, naive and
seasonal-naive floors, classical ETS/ARIMA, the XGBoost lag-feature baseline, seeded
LSTM and small Transformer forecasters plus zero-shot TimesFM 2.5 / Chronos-Bolt
wrappers behind the weight-download gate, the electricity-demand data layer with frozen
leakage-audited splits, metrics, rolling-origin backtest, results schema, paired
statistics, and CI. `make smoke` runs end-to-end on the committed fixture through the
real `naive` baseline; the full GPU tier has now run for real on Kaggle T4 through the
compute handoff in `docs/colab-handoff.md`.

**Complete benchmark results.** All five local families **and** the GPU tier (seeded
LSTM/Transformer, zero-shot TimesFM 2.5 / Chronos-Bolt) now run end-to-end on the real
Monash *Electricity Hourly* dataset over horizons 24/48/96/192, on the **same** bounded
origin sample of the frozen splits (identical split-manifest digest across both runs).
The zero-shot TSFMs win outright: at the 24-step horizon `chronos_bolt` leads (seasonal
MASE **0.79** at ≈35 ms/forecast), `timesfm25` is second (**0.88** at ≈122 ms), both
ahead of the `seasonal_naive` floor (**1.00**) and every trained model. Classical
`auto_ets` is the best trained model (**1.14** at ≈177 ms), `xgboost_lags` is
competitive (**1.29** at ≈104 ms), and the from-scratch deep models are the negative
result — LSTM **1.75** and Transformer **1.74** at ≈342/600 ms — behind the seasonal
floor. The zero-shot lead holds at every horizon (Chronos MASE 0.79→1.04, TimesFM
0.88→1.11, seasonal floor 1.00→1.30). Full table, figures, and the measured-vs-pending
list: [`REPORT.md`](REPORT.md) and [`docs/results_summary.csv`](docs/results_summary.csv).

## Quickstart

The `models` extra installs the local model tier's runtime deps (`xgboost`,
`scikit-learn`, `statsforecast`, `statsmodels`); without it the tier's tests skip.
The `report` extra adds `matplotlib` for the figures (it is also in `dev`, so CI
exercises the report script).

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,models,report]"
make test
make lint
```

## Make targets

| Target | What it does |
| --- | --- |
| `make test` | run the pytest suite |
| `make lint` | run ruff |
| `make smoke` | tiny end-to-end run: `python -m tsbench run --config configs/experiments/smoke.yaml` over `tests/fixtures/smoke_series.csv` through the `local_csv` loader |
| `make data` | fetch and checksum-verify the dataset catalogue (`python -m tsbench.data.build_catalog --materialize`; see `data/manifests/PROVENANCE.md`) |
| `make backtest` | 24h rolling-origin backtest, all five local families (`configs/experiments/backtest.yaml`); bounded origin stride, see `REPORT.md` |
| `make horizons` | complete the primary sweep at 48/96/192 on the same frozen boundaries (`configs/experiments/backtest_h{48,96,192}.yaml`) |
| `make gpu` | deep LSTM/Transformer + zero-shot TimesFM 2.5/Chronos-Bolt on the CPU families' frozen windows (`configs/experiments/gpu.yaml`; run in a GPU session via `docs/colab-handoff.md`) |
| `make gpu-horizons` | remaining GPU-tier horizons (`configs/experiments/gpu_h{48,96,192}.yaml`) |
| `make report` | merge the live `results/` store with the committed `docs/telemetry/` store and regenerate `docs/results_summary.csv` + `docs/figures/` (`scripts/make_plots.py`) |
| `make licenses` | print the model license manifest from `configs/models.yaml` |
| `make reproduce` | rerun the documented end-to-end path |

Experiment stages are dispatched through `python -m tsbench run --config <experiment.yaml>`;
dataset materialization runs through `python -m tsbench.data.build_catalog --materialize`,
and configs for later phases land with their slices. CI runs the same commands the
quickstart does: `ruff check .` and `python -m pytest`.

## Frozen interfaces

`src/tsbench/base.py` is the fleet-wide contract that the data, model, and
evaluation slices build against:

- `Forecaster` protocol — `.name`, `.fit(y) -> self`, `.predict(h) -> np.ndarray`,
  `.info() -> ModelInfo`
- `ModelInfo(name, family, zero_shot, params, license, revision, extra)` — `extra`
  optionally carries engine/backend details; `zero_shot` marks forecasters with no
  learned parameters (reference floors and untuned TSFMs)
- `RESULT_COLUMNS` — every result row carries those columns in that exact order.
  `train_seconds` sits immediately after `peak_mem_mb`; it is the measured fit/train
  wall-clock for families that train or fit (classical, ML, deep) and is left empty
  only for the `baseline` and zero-shot `tsfm` families.

The experiment runner is resolved from `tsbench.evaluation.runner.run_experiment`
and called with the experiment config path. The registry is
`tsbench.registry.load_registry`; `ModelRegistry.instantiate` is the only public
construction path and every call passes the license gate.

A config-declared `entrypoint` for a name the registry does not know is the
documented not-yet-landed-module fallback; it is imported directly and is
therefore **outside** the registry's license gate. Registry-known names still
pass the gate (their not-yet-landed fallback calls `check_license` before
importing the config entrypoint), and the TSFM wrappers still refuse unverified
checkpoint ids through the fallback.

## Models and the license gate

`configs/models.yaml` declares nine keys: `naive`, `seasonal_naive`, `auto_ets`,
`auto_arima`, `xgboost_lags`, `lstm`, `transformer`, `timesfm25`, `chronos_bolt`.
Each entry must declare `entrypoint`, `family`, `zero_shot`, `license`, and
`revision`; an entry missing a license or revision is refused at load time.
Instantiation is refused for any license outside the explicit permissive allowlist
(Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC, 0BSD, Unlicense, CC0-1.0,
CC-BY-4.0) unless `TSBENCH_ALLOW_NONCOMMERCIAL=1` (or `run --allow-noncommercial`)
is set. `TBD-at-download` marks a revision pinned at fetch time; `make licenses`
prints the manifest revision for every key. The results schema does not carry a
revision column. TimesFM 2.5 (Apache-2.0) is in scope; TimesFM 3.0 weights are
non-commercial and deliberately out of scope.

## Environment variables

| Variable | Effect |
| --- | --- |
| `TSBENCH_ALLOW_NONCOMMERCIAL=1` | instantiate non-permissive or unknown-license models |
| `TSBENCH_ALLOW_MODEL_DOWNLOAD=1` | permit TSFM weight downloads (GPU slice gate) |
| `TSBENCH_RESULTS_DIR` | override the results root (default `results/`) |

## Layout

```
configs/models.yaml            model keys, entrypoints, licenses, revisions
configs/experiments/           smoke, backtest/gpu (24h), backtest_h{48,96,192}, gpu_h{48,96,192}, gpu_probe/pilot
src/tsbench/base.py            frozen contract: Forecaster, ModelInfo, RESULT_COLUMNS
src/tsbench/registry.py        manifest validation + license-gated factory
src/tsbench/cli.py             python -m tsbench run|licenses
src/tsbench/data/              dataset loaders and splits (data slice)
src/tsbench/models/            model implementations (model and GPU slices)
src/tsbench/evaluation/        metrics, backtest, stats, results, runner (data slice)
scripts/make_plots.py          telemetry store -> summary CSV + figures (make report)
scripts/kaggle_run.py          resumable Kaggle GPU batch route (see docs/colab-handoff.md)
docs/telemetry/{cpu,gpu}/      committed per-run raw telemetry (results.csv + run.json)
docs/results_summary.csv       committed per-family per-horizon summary (real numbers)
docs/figures/                  committed MASE and accuracy-vs-cost figures
REPORT.md                      setup, results, measured-vs-pending, limitations
tests/                         pytest suite and the smoke fixture
```

See `PLAN.md` for the condensed plan of record: scope, experiment matrix,
phases, risks, and deliverables.
