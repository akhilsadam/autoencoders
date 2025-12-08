.ONESHELL:

SHELL := /bin/bash
VENV ?= .venv
SYS_PYTHON ?= python3.10
UV ?= uv
PYTHON ?= $(VENV)/bin/python
BASE_PYDIR ?= $(shell grep "home =" $(VENV)/pyvenv.cfg | cut -d' ' -f3-)

INSTALL = src/install

.PHONY: install data train benchmark

install:
	@# Ensure uv is available, then create venv (if missing) and install deps
	@command -v $(UV) >/dev/null 2>&1 || $(SYS_PYTHON) -m pip install --user uv
	@[ -d "$(VENV)" ] || $(UV) venv $(VENV)
	$(UV) pip install -r $(INSTALL)/requirements.txt --python $(PYTHON)
# if python3-config is missing, find and link it
	$(MAKE) py3-conf;

install-local:
	@# Ensure uv is available, then create venv (if missing) and install deps
	@command -v $(UV) >/dev/null 2>&1 || $(SYS_PYTHON) -m pip install --user uv
	@[ -d "$(VENV)" ] || $(UV) venv $(VENV)
	$(UV) pip install -r $(INSTALL)/requirements_client.txt --python $(PYTHON)

py3-conf:
	@# find system python3-config by looking in pyenv
	-ln -s $(BASE_PYDIR)/python3-config $(VENV)/bin/python3-config;
	alias python3-config='$(VENV)/bin/python3-config';

# Need template parameters...
# compile: install
# 	source "$(VENV)/bin/activate" && \
# 	$(PYTHON) -m src.autoencoders.models.cuda.compile ${VENV}

test-cu: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m pytest -s -v src/autoencoders/models/cu
# 	src.autoencoders.models.cu.compile

test-hl: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m pytest -s -v src/autoencoders/models/hl

train: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train
# 	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train run.name=local-debug run.tags=[local,debug]

# scompile: 
# 	source ${INSTALL}/module.sh && $(MAKE) compile

clean-cu:
	rm -rf ~/.cache/torch_extensions/py*

sinstall: 
	source ${INSTALL}/module.sh && $(MAKE) install

stest-hl:
	source ${INSTALL}/module.sh && $(MAKE) test-hl

stest-cu:
	source ${INSTALL}/module.sh && $(MAKE) test-cu

stest-cu-single:
	source ${INSTALL}/module.sh && \
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m pytest -s -v src/autoencoders/models/cu -k "${TEST}"
	
	/tests/${TEST}


stest-cu-single-memcheck:
	source ${INSTALL}/module.sh && \
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 \
	compute-sanitizer --tool memcheck \
	bash -c "$(PYTHON) -m pytest -s -v src/autoencoders/models/cu tests/${TEST}"

strain:
	source ${INSTALL}/module.sh && $(MAKE) train

# push-train: install-local
# 	$(PYTHON) -m src.deploy.hpc_deploy --dry-run

# benchmark: install data
# 	$(VENV)/bin/pytest -m benchmark -s

#### Install specific ####
# ONLY first time
add-tk:
	git subtree add --prefix=lib/ext/tk/ git@github.com:HazyResearch/ThunderKittens.git main --squash

update-tk:
	git subtree pull --prefix=lib/ext/tk/ git@github.com:HazyResearch/ThunderKittens.git main --squash

select-tk:
	git subtree add --prefix=lib/ext/tk/ git@github.com:HazyResearch/ThunderKittens.git b59cee6a3a46ddb046df76d28c18522fd5ce90eb