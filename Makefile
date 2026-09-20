PYTHON ?= python

.PHONY: test lint smoke data backtest horizons gpu gpu-horizons report licenses reproduce

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

smoke:
	$(PYTHON) -m tsbench run --config configs/experiments/smoke.yaml

data:
	$(PYTHON) -m tsbench.data.build_catalog --materialize

backtest:
	$(PYTHON) -m tsbench run --config configs/experiments/backtest.yaml

# The remaining horizons of the primary 24/48/96/192 sweep. `make backtest`
# runs the 24h primary; this target completes the sweep on the same frozen
# boundaries.
horizons:
	$(PYTHON) -m tsbench run --config configs/experiments/backtest_h48.yaml
	$(PYTHON) -m tsbench run --config configs/experiments/backtest_h96.yaml
	$(PYTHON) -m tsbench run --config configs/experiments/backtest_h192.yaml

# GPU tier (deep LSTM/Transformer + zero-shot TimesFM 2.5/Chronos-Bolt) on the
# CPU families' frozen windows. These need torch and, for the TSFM keys, the
# licensed checkpoints; run them in a GPU session via the compute handoff
# (docs/colab-handoff.md) or `python scripts/kaggle_run.py experiments ...`.
gpu:
	$(PYTHON) -m tsbench run --config configs/experiments/gpu.yaml

# Remaining horizons of the GPU tier's primary 24/48/96/192 sweep.
gpu-horizons:
	$(PYTHON) -m tsbench run --config configs/experiments/gpu_h48.yaml
	$(PYTHON) -m tsbench run --config configs/experiments/gpu_h96.yaml
	$(PYTHON) -m tsbench run --config configs/experiments/gpu_h192.yaml

# Aggregate the live store plus the committed telemetry of both runs.
report:
	$(PYTHON) scripts/make_plots.py

licenses:
	$(PYTHON) -m tsbench licenses

# The documented end-to-end path: fetch the frozen dataset, run every family
# (local CPU here; the GPU tier via the handoff), then regenerate the summary.
reproduce: data backtest horizons report
