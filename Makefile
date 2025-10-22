SHELL := /bin/bash
VENV ?= .venv
SYS_PYTHON ?= python3
UV ?= uv
PYTHON ?= $(VENV)/bin/python

.PHONY: install data train benchmark

install:
	@# Ensure uv is available, then create a dedicated venv and install deps into it
	@if ! command -v $(UV) >/dev/null 2>&1; then $(SYS_PYTHON) -m pip install --user uv; fi
	$(UV) venv $(VENV)
	$(UV) pip install -r src/install/requirements.txt --python $(PYTHON)
	$(MAKE) data

data:
	@if [ -d data/aesthetic4k ]; then \
		echo "Aesthetic4K dataset already present at data/aesthetic4k"; \
	else \
		if [ -n "$$AESTHETIC4K_SOURCE_DIR" ]; then \
			$(PYTHON) -m src.install.fetch_aesthetic4k --output data/aesthetic4k --source-dir "$$AESTHETIC4K_SOURCE_DIR"; \
		elif [ -n "$$AESTHETIC4K_ARCHIVE" ]; then \
			$(PYTHON) -m src.install.fetch_aesthetic4k --output data/aesthetic4k --archive "$$AESTHETIC4K_ARCHIVE"; \
		elif [ -n "$$AESTHETIC4K_URL" ]; then \
			$(PYTHON) -m src.install.fetch_aesthetic4k --output data/aesthetic4k --url "$$AESTHETIC4K_URL"; \
		elif [ -n "$$AESTHETIC4K_REPO_ID" ]; then \
			$(PYTHON) -m src.install.fetch_aesthetic4k --output data/aesthetic4k --repo-id "$$AESTHETIC4K_REPO_ID"; \
		else \
			echo "No Aesthetic4K source provided; set AESTHETIC4K_REPO_ID, AESTHETIC4K_URL, AESTHETIC4K_ARCHIVE, or AESTHETIC4K_SOURCE_DIR."; \
			echo "Skipping dataset download."; \
		fi; \
	fi

train: install
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train trainer.max_epochs=1 run.name=local-debug run.tags=[local,debug] wandb.mode=online

benchmark: install data
	$(VENV)/bin/pytest -m benchmark -s