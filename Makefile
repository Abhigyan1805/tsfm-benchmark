PYTHON ?= python

.PHONY: test lint smoke data backtest deep tsfm report licenses reproduce

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

deep:
	$(PYTHON) -m tsbench run --config configs/experiments/deep.yaml

tsfm:
	$(PYTHON) -m tsbench run --config configs/experiments/tsfm.yaml

report:
	$(PYTHON) -m tsbench run --config configs/experiments/report.yaml

licenses:
	$(PYTHON) -m tsbench licenses

reproduce:
	$(PYTHON) -m tsbench run --config configs/experiments/reproduce.yaml
