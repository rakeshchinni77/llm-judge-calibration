# ===========================================================================
# Makefile — LLM-as-a-Judge Calibration Pipeline
# ===========================================================================
# Targets:
#   make install   - install Python dependencies into a virtual env
#   make up        - docker compose up (end-to-end test run)
#   make split     - run the deterministic data splitter
#   make dev       - evaluate against the dev split
#   make test      - evaluate against the frozen test split
#   make bench     - run bias probes
#   make leakage   - run the induced-failure experiment
#   make pytest    - run the unit test suite
#   make lint      - run ruff/mypy checks
#   make clean     - remove generated results
# ===========================================================================

PYTHON     ?= python3.11
VENV       := .venv
PIP        := $(VENV)/bin/pip
PY         := $(VENV)/bin/python

# ---- Environment setup -----------------------------------------------------

.PHONY: install
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅  Virtual environment ready. Activate with: source $(VENV)/bin/activate"

# ---- Docker targets ---------------------------------------------------------

.PHONY: up
up:
	docker compose up --build

# ---- Pipeline targets -------------------------------------------------------

.PHONY: split
split:
	$(PY) src/split_data.py --input data/labels.csv --output-dir data/splits/

.PHONY: dev
dev:
	$(PY) evaluate.py --split dev

.PHONY: test
test:
	$(PY) evaluate.py --split test

.PHONY: bench
bench:
	$(PY) src/bias_probes.py

.PHONY: leakage
leakage:
	$(PY) scripts/induce_leakage.py

# ---- Quality targets --------------------------------------------------------

.PHONY: pytest
pytest:
	$(PY) -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

.PHONY: lint
lint:
	$(PY) -m ruff check src/ scripts/ evaluate.py
	$(PY) -m mypy src/ scripts/ evaluate.py --ignore-missing-imports

# ---- Housekeeping -----------------------------------------------------------

.PHONY: clean
clean:
	rm -f results/metrics.csv results/bias.csv results/judge_test.csv
	rm -f results/induced_failure.json
	@echo "🧹  results/ cleaned (labels.csv preserved)"
