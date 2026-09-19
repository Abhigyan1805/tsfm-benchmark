# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## Sharp edges

- `src/tsbench/base.py` is the frozen fleet-wide contract (`Forecaster`, `ModelInfo`, `RESULT_COLUMNS`). The data, model, and GPU slices import these names; never redefine or reorder them. `train_seconds` sits immediately after `peak_mem_mb`; it is empty for zero-shot and statistical models and populated for trained families.
- `configs/models.yaml` keys and entrypoints are frozen fleet-wide. The registry refuses entries missing `license` or `revision` at load time, and refuses instantiation of non-permissive or unknown licenses unless `TSBENCH_ALLOW_NONCOMMERCIAL=1`. `ModelRegistry.instantiate` is the only public construction path (the resolver is private) so every construction passes the license gate. Run `make licenses` to print the manifest.
- Experiments run through `python -m tsbench run --config <experiment.yaml>`, which dispatches to `tsbench.evaluation.runner.run_experiment` with the config path. Until the evaluation slice lands, the command fails with an explicit `error: cannot execute experiments` message; that is expected, not a regression. `src/tsbench/__main__.py` is the required shim for `python -m tsbench`.
- Slices own disjoint paths by design (data/evaluation, local models, GPU models). Check `PLAN.md` and the current task brief before editing outside your slice.

## Commands

- Setup: `pip install -e ".[dev]"`; then `make test`, `make lint`, `make smoke`, `make licenses`.
- Tests import `tsbench` without an install because pytest injects `src` via `pythonpath`; CI does a full editable install and runs ruff + pytest on 3.10 and 3.12.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
