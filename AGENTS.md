# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## Sharp edges

- `src/tsbench/base.py` is the frozen fleet-wide contract (`Forecaster`, `ModelInfo`, `RESULT_COLUMNS`). The data, model, and GPU slices import these names; never redefine or reorder them. `train_seconds` sits immediately after `peak_mem_mb`; it is the measured fit/train wall-clock for families that train or fit (classical, ML, deep) and empty only for the `baseline` and zero-shot `tsfm` families.
- `tsbench.evaluation.contract` is the single import site for the frozen `tsbench.base` contract: it uses `tsbench.base` when importable and otherwise falls back to a byte-identical schema, so the rest of the data/evaluation code and its tests never import `tsbench.base` directly. Keep the fallback in lockstep.
- `configs/models.yaml` keys and entrypoints are frozen fleet-wide. The registry refuses entries missing `license` or `revision` at load time, and refuses instantiation of non-permissive or unknown licenses unless `TSBENCH_ALLOW_NONCOMMERCIAL=1`. `ModelRegistry.instantiate` is the only public construction path (the resolver is private) so every registry construction passes the license gate. The runner routes a registry-known model through `instantiate` first and falls back to a config-declared `entrypoint` only when the registered module cannot be imported (the not-yet-landed-module stub fallback); registry-unknown models may declare their own entrypoint. Run `make licenses` to print the manifest.
- Experiments run through `python -m tsbench run --config <experiment.yaml>`, which dispatches to `tsbench.evaluation.runner.run_experiment`. `src/tsbench/__main__.py` is the required shim for `python -m tsbench`. The CLI catches the runner's `RunnerError` and prints an `error: ...` line instead of a traceback; other exceptions propagate.
- The local-family backtest configs are `configs/experiments/backtest.yaml` (24h) plus `backtest_h{48,96,192}.yaml` (the primary 24/48/96/192 sweep; `make backtest` and `make horizons`). They are deliberately **bounded**: `stride: 240` samples 21-22 origins per series instead of the manifest's 24-step stride, and `track_memory: false` because `tracemalloc` makes statsforecast `auto_arima` ~3.5x slower (41s vs 12s for one 22-origin series). The bound is declared in each config's `subset:` block (stored verbatim in `run.json`) and in `REPORT.md`; `peak_mem_mb` is empty when tracking is off. `ExperimentConfig.track_memory` threads through to `backtest_windows`.
- Slices own disjoint paths by design (data/evaluation, local models, GPU models). Check `PLAN.md` and the current task brief before editing outside your slice.
- GPU tier: `src/tsbench/models/deep/` (seeded LSTM/Transformer) and `src/tsbench/models/tsfm/` (zero-shot TimesFM 2.5 / Chronos-Bolt). Heavy deps import lazily and are not CI extras, so trained-path tests skip without torch; TSFM tests use fake backends and never touch the network. Weights require `TSBENCH_ALLOW_MODEL_DOWNLOAD=1`, and TimesFM 3.0 ids are refused. Both compute routes with exact commands live in `docs/colab-handoff.md`; runners are `scripts/serve_endpoint.py`, `scripts/kaggle_run.py`, and the shared `scripts/colab_run.py`.
- `.gitignore`'s `/data/` pattern anchors to the repo root, so `src/tsbench/data/` is linted and tracked normally; the committed `data/manifests/` files were force-added and any regeneration still needs `git add -f`.
- Dataset downloads and run outputs live under the gitignored `data/` and `results/`. The frozen split manifest is the exception and is committed under `data/manifests/`; keep `data/manifests/checksums.sha256` in sync after regenerating.

## Commands

- Setup: `pip install -e ".[dev,models]"`; then `make test`, `make lint`, `make smoke`, `make licenses`. The `models` extra installs the local model tier's runtime deps (`xgboost`, `scikit-learn`, `statsforecast`, `statsmodels`); without it the tier's tests skip.
- Tests import `tsbench` without an install because pytest injects `src` via `pythonpath`; CI does a full editable install and runs ruff + pytest on 3.10 and 3.12. Local test/lint: `python -m pytest`, `python -m ruff check .`.
- Primary dataset: `python -m tsbench.data.build_catalog --materialize` (network, checksum-verified), then `make backtest`. Offline provenance: `python -m tsbench.data.build_catalog`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
