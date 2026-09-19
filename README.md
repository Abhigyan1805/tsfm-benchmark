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
statistics, and CI. `make smoke` runs end-to-end on the committed fixture; GPU runs go
through the compute handoff in `docs/colab-handoff.md`.

## Quickstart

The `models` extra installs the local model tier's runtime deps (`xgboost`,
`scikit-learn`, `statsforecast`, `statsmodels`); without it the tier's tests skip.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,models]"
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
| `make backtest` | rolling-origin backtest (`configs/experiments/backtest.yaml`) |
| `make deep` | LSTM / small Transformer run (`configs/experiments/deep.yaml`) |
| `make tsfm` | TimesFM 2.5 / Chronos-Bolt zero-shot run (`configs/experiments/tsfm.yaml`) |
| `make report` | aggregate results and regenerate figures |
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
  `train_seconds` sits immediately after `peak_mem_mb`; it is populated for trained
  families and left empty for zero-shot and statistical models
  (`zero_shot: true` and the classical/ML families).

The experiment runner is resolved from `tsbench.evaluation.runner.run_experiment`
and called with the experiment config path. The registry is
`tsbench.registry.load_registry`; `ModelRegistry.instantiate` is the only public
construction path and every call passes the license gate.

## Models and the license gate

`configs/models.yaml` declares nine keys: `naive`, `seasonal_naive`, `auto_ets`,
`auto_arima`, `xgboost_lags`, `lstm`, `transformer`, `timesfm25`, `chronos_bolt`.
Each entry must declare `entrypoint`, `family`, `zero_shot`, `license`, and
`revision`; an entry missing a license or revision is refused at load time.
Instantiation is refused for any license outside the explicit permissive allowlist
(Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC, 0BSD, Unlicense, CC0-1.0,
CC-BY-4.0) unless `TSBENCH_ALLOW_NONCOMMERCIAL=1` (or `run --allow-noncommercial`)
is set. `TBD-at-download` marks a revision pinned at fetch time; pinned revisions
are recorded with every result row. TimesFM 2.5 (Apache-2.0) is in scope; TimesFM
3.0 weights are non-commercial and deliberately out of scope.

## Environment variables

| Variable | Effect |
| --- | --- |
| `TSBENCH_ALLOW_NONCOMMERCIAL=1` | instantiate non-permissive or unknown-license models |
| `TSBENCH_ALLOW_MODEL_DOWNLOAD=1` | permit TSFM weight downloads (GPU slice gate) |
| `TSBENCH_RESULTS_DIR` | override the results root (default `results/`) |

## Layout

```
configs/models.yaml            model keys, entrypoints, licenses, revisions
configs/experiments/smoke.yaml smoke experiment
src/tsbench/base.py            frozen contract: Forecaster, ModelInfo, RESULT_COLUMNS
src/tsbench/registry.py        manifest validation + license-gated factory
src/tsbench/cli.py             python -m tsbench run|licenses
src/tsbench/data/              dataset loaders and splits (data slice)
src/tsbench/models/            model implementations (model and GPU slices)
src/tsbench/evaluation/        metrics, backtest, stats, results, runner (data slice)
tests/                         pytest suite and the smoke fixture
```

See `PLAN.md` for the condensed plan of record: scope, experiment matrix,
phases, risks, and deliverables.
