# I2AS development checks — single source of truth for all quality gates.
#
# CI (GitHub Actions now, GitLab CI after the migration) calls these targets;
# keep all check logic HERE so the CI configs stay thin wrappers that never
# need to change when a check is added or adjusted.
#
# Usage (from an activated .venv, or any environment with the dev deps):
#   make check       run every blocking gate (lint + contracts + tests)
#   make test        run the pytest suite (hardware-marked tests excluded),
#                    with the instrument stack on its own thread — the default
#   make test-instrument-inline
#                    run the GUI suite again in the temporary inline mode
#                    (I2AS_INSTRUMENT_THREAD=0) — the same assertions
#   make contracts   verify the layer import contracts (import-linter)
#   make lint        ruff error-level lint (undefined names, unused imports)
#   make typecheck   mypy basic mode — advisory, not part of `check` yet
#   make install     editable install with dev dependencies
#
# Windows note: install GNU make once via `scoop install make`. Every target
# is a single command that can also be run directly without make.

PYTHON ?= python

.PHONY: install test test-instrument-inline contracts lint typecheck check

install:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest -m "not hardware"

# The instrument-thread flag's second half. `make test` runs everything on the
# instrument thread, which is the default; this runs the GUI suite again with
# `I2AS_INSTRUMENT_THREAD=0`, the temporary `inline` mode, which is the
# one difference the windows are supposed not to be able to see. Only the GUI
# suite, because it is the suite whose fixtures build through the
# InstrumentHost and therefore honour the flag — see
# tests/instrument_modes.py. This leg goes when `inline` does.
test-instrument-inline:
	I2AS_INSTRUMENT_THREAD=0 $(PYTHON) -m pytest -m "not hardware" \
		tests/test_gui.py tests/test_instrument_thread.py

contracts:
	lint-imports

lint:
	ruff check .

typecheck:
	-$(PYTHON) -m mypy

check: lint contracts test test-instrument-inline
	@echo "All blocking checks passed."
