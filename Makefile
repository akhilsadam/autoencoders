SHELL := /bin/bash
VENV ?= .venv
SYS_PYTHON ?= python3
UV ?= uv
PYTHON ?= $(VENV)/bin/python

.PHONY: install data train benchmark

install:
	@# Ensure uv is available, then create venv (if missing) and install deps
	@command -v $(UV) >/dev/null 2>&1 || $(SYS_PYTHON) -m pip install --user uv
	@[ -d "$(VENV)" ] || $(UV) venv $(VENV)
	$(UV) pip install -r src/install/requirements.txt --python $(PYTHON)

install-local:
	@# Ensure uv is available, then create venv (if missing) and install deps
	@command -v $(UV) >/dev/null 2>&1 || $(SYS_PYTHON) -m pip install --user uv
	@[ -d "$(VENV)" ] || $(UV) venv $(VENV)
	$(UV) pip install -r src/install/requirements_client.txt --python $(PYTHON)

slurm: 
	- module load gcc
	- module load miniforge
	- module load cuda/13.0.1

slurm_install: slurm install

train: install
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train
# 	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train run.name=local-debug run.tags=[local,debug]

slurm_train:slurm_install train

push-train: install-local
	$(PYTHON) -m src.deploy.hpc_deploy --dry-run

benchmark: install data
	$(VENV)/bin/pytest -m benchmark -s