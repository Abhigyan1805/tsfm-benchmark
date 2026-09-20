# PLAN — tsfm-benchmark (condensed plan of record)

Condensed from the captain-approved orchestration plan (2026-09-19). The plan of
record governs until the captain approves changes; this file tracks the in-repo
summary.

## 0. Research question and positioning

**Question:** when do time-series foundation models (TSFMs) outperform conventional
forecasting models, and what is the accuracy/latency trade-off?

**Positioning:** a reproducible, leakage-audited head-to-head of classical, ML, deep,
and foundation forecasters across forecasting horizons — measured on accuracy *and*
inference cost, with a partial reproduction of the TimesFM evaluation methodology,
producing an accuracy-vs-cost Pareto frontier rather than a single leaderboard row.

The defensible deltas: (a) cost-aware evaluation at fixed accuracy, (b) zero-shot vs
trained separation, (c) an explicit paper-reproduction lane, (d) leakage audits
anyone can rerun.

## 1. Scope decisions (locked at intake)

- **TSFMs:** TimesFM 2.5 (200M, long context) and Chronos-Bolt. TimesFM 3.0 is out of
  scope because its pretrained weights are non-commercial. Optional third: Moirai.
- **License gate:** every model and dataset license is verified at pin time and
  recorded in a committed manifest before any download. A license failure is a
  blocker, not a footnote.
- **First dataset:** one electricity-demand series set with obvious temporal
  structure, from a standard benchmark source (Monash class), so results are
  comparable to literature. Genuinely new data — not the captain's old EV project.
- **Families:** naive/seasonal-naive · classical (AutoETS, AutoARIMA via
  statsforecast) · gradient boosting (XGBoost with lag/calendar features) · deep
  trained (LSTM, small Transformer) · TSFM zero-shot.
- **Horizons:** primary sweep 24/48/96/192 steps; report per-horizon, never one
  number.
- **Metrics:** MAE, RMSE, seasonal MASE, sMAPE; latency p50/p95; peak memory;
  parameter count; training wall-clock for trained families.
- **Statistics:** MASE ratio/rank aggregation; Wilcoxon signed-rank across series;
  Friedman test with critical-difference diagram.
- **Backtest:** rolling-origin evaluation with a frozen split manifest (hashes
  committed). Context length is a first-class variable for TSFMs (512/1024/2048).
- **Compute:** classical + XGBoost local CPU; LSTM/Transformer training and TSFM
  inference on Colab GPU through the proven handoff pattern. No paid cloud.
- **Everything on D drive:** clone, results, artifacts at `/mnt/d/tsfm-benchmark`;
  transient crew worktrees stay in the normal treehouse pool.

## 2. Repository layout

```
README.md            research question, status, make targets
PLAN.md              this file
REPORT.md            full write-up, negative results, limitations
pyproject.toml
Makefile             smoke | data | backtest | horizons | gpu | gpu-horizons | report | licenses | reproduce
.github/workflows/   CI: ruff + pytest
configs/
  datasets.yaml      pinned sources + license manifest refs
  models.yaml        pinned model revisions + licenses
  experiments/       smoke and phase experiment configs
src/tsbench/
  data/              loaders, splitter, windowing, leakage checks
  models/            naive, stats, gbdt, deep/, tsfm/
  evaluation/        metrics, backtest, runner, stats, report
  base.py            frozen Forecaster/ModelInfo/RESULT_COLUMNS contract
  registry.py        model registry + license manifest enforcement
  cli.py             config-driven entry point
scripts/             Colab handoff, plot generation
results/             gitignored run outputs; committed aggregates + split hashes
tests/
notebooks/           exploration only, nothing load-bearing
```

Every result row carries dataset, series id, model, family, context length, horizon,
window index, metrics, latency, memory, training wall-clock, params, zero-shot flag,
git sha, config hash, measured-vs-estimated flag, and timestamp.

## 3. Experiment matrix

| # | Experiment | Question |
| --- | --- | --- |
| E1 | Naive/seasonal-naive + leakage audits | Sanity floor; is the backtest honest? |
| E2 | Classical (ETS/ARIMA) | Cheap statistical reference strength |
| E3 | XGBoost + feature ablation | Do lag features close the gap? |
| E4 | LSTM + small Transformer | Trained-deep reference at fixed budget |
| E5 | TSFM zero-shot + context sweep | Do foundation models win out of the box? |
| E6 | Horizon scaling 24/48/96/192 | Where does each family break? |
| E7 | Accuracy vs inference cost Pareto | The headline artifact |
| E8 | Paired statistics + CD diagram | Is any difference real? |
| E9 | TimesFM paper reproduction subset | Does the published methodology reproduce? |
| E10 | Second dataset (stretch) | Does the ranking transfer? |

## 4. Phases and exit criteria

- **P0 — Foundation.** Skeleton, config system, results schema, license manifest, CI,
  Makefile. *Exit:* `make smoke` runs the naive baseline on the tiny fixture.
- **P1 — Data + baselines.** Primary dataset loader, frozen splits, rolling-origin
  backtest, E1. *Exit:* first results table + leakage checks pass.
- **P2 — Classical + ML.** E2 + E3 with feature ablation. *Exit:* local-run table.
- **P3 — Deep.** LSTM + Transformer on Colab, seed-reproducible. *Exit:* checkpoints
  reproducible; results copied back to D.
- **P4 — TSFM zero-shot.** TimesFM 2.5 + Chronos, latency/memory harness, E5/E6.
  *Exit:* zero-shot table + cost numbers.
- **P5 — Headline.** E7 Pareto + E8 statistics + E9 reproduction subset. *Exit:*
  README headline plot + REPORT draft.
- **P6 — Publish.** Blog post, `make reproduce`, docs polish. *Exit:* a stranger can
  reproduce the headline number.

## 5. Orchestration model

- Each phase is 1–2 parallel ship tasks in isolated worktrees with disjoint owned
  paths; local phases run parallel with Colab-dependent prep so GPU waits never
  block CPU work.
- Colab boundary as in costsmart-rag: prep first, request endpoint, captain connects,
  resumable cached runs.
- Delivery posture `no-mistakes-prod-only`: full validation pipeline; the captain
  merges every PR.

## 6. Risk register

| Risk | Signal | Mitigation |
| --- | --- | --- |
| License trap (TimesFM 3.0 non-commercial) | non-permissive license detected | pin 2.5/Chronos; manifest gate blocks download |
| Colab session churn / GPU limits | endpoint drops mid-run | resumable caches; CPU work continues |
| Backtest leakage | splits use future data | explicit leakage tests in E1 |
| Not-comparable baselines | different context/normalization | one protocol, documented per model |
| Deep-model compute ceiling | training exceeds budget | small models, fixed budget, one dataset |
| Version drift | results not reproducible | pinned revisions + environment record |

## 7. Deliverables

README with the Pareto plot above the fold, REPORT.md (all experiments, negative
results, limitations), critical-difference diagram, blog post, committed split
hashes, `make reproduce`.
