PYTHON ?= python

.PHONY: test lint smoke data backtest horizons deep tsfm report licenses reproduce

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

deep:
	$(PYTHON) -m tsbench run --config configs/experiments/deep.yaml

tsfm:
	$(PYTHON) -m tsbench run --config configs/experiments/tsfm.yaml

report:
	$(PYTHON) scripts/make_plots.py

licenses:
	$(PYTHON) -m tsbench licenses

reproduce:
	$(PYTHON) -m tsbench run --config configs/experiments/reproduce.yaml
