.ONESHELL:

SHELL := /bin/bash
VENV ?= .venv
SYS_PYTHON ?= python3.10
UV ?= uv
PYTHON ?= $(VENV)/bin/python
BASE_PYDIR ?= $(shell grep "home =" $(VENV)/pyvenv.cfg | cut -d' ' -f3-)

.PHONY: install data train benchmark

install:
	@# Ensure uv is available, then create venv (if missing) and install deps
	@command -v $(UV) >/dev/null 2>&1 || $(SYS_PYTHON) -m pip install --user uv
	@[ -d "$(VENV)" ] || $(UV) venv $(VENV)
	$(UV) pip install -r src/install/requirements.txt --python $(PYTHON)
# if python3-config is missing, find and link it
	$(MAKE) py3-conf;

install-local:
	@# Ensure uv is available, then create venv (if missing) and install deps
	@command -v $(UV) >/dev/null 2>&1 || $(SYS_PYTHON) -m pip install --user uv
	@[ -d "$(VENV)" ] || $(UV) venv $(VENV)
	$(UV) pip install -r src/install/requirements_client.txt --python $(PYTHON)

py3-conf:
	@# find system python3-config by looking in pyenv
	-ln -s $(BASE_PYDIR)/python3-config $(VENV)/bin/python3-config;
	alias python3-config='$(VENV)/bin/python3-config';


compile:
	source "$(VENV)/bin/activate" && \
	$(PYTHON) -m src.autoencoders.models.cuda.compile ${VENV}

slurm: 
	- module load gcc
	- module load miniforge
	- module load cuda/13.0.1
	- module load nvhpc/24.5

slurm_install: slurm install

train: install compile
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train
# 	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train run.name=local-debug run.tags=[local,debug]

slurm_train:slurm_install train

push-train: install-local
	$(PYTHON) -m src.deploy.hpc_deploy --dry-run

benchmark: install data
	$(VENV)/bin/pytest -m benchmark -s

#### Install specific ####
# ONLY first time
add-tk:
	git subtree add --prefix=lib/ext/tk/ git@github.com:HazyResearch/ThunderKittens.git main --squash

update-tk:
	git subtree pull --prefix=lib/ext/tk/ git@github.com:HazyResearch/ThunderKittens.git main --squash
