.ONESHELL:
SHELL := /bin/bash

# ========================================
# Configuration
# ========================================
VENV ?= .venv
SYS_PYTHON ?= python3.12
UV ?= uv
PYTHON ?= $(VENV)/bin/python
BASE_PYDIR ?= $(shell grep "home =" $(VENV)/pyvenv.cfg | cut -d' ' -f3-)
INSTALL = src/install

.PHONY: help install train test clean

# ========================================
# Help
# ========================================
help:
	@echo "Autoencoders Makefile"
	@echo ""
	@echo "Installation:"
	@echo "  make install          - Full install "
	@echo "  make install-packages   - Install (external) packages only"
	@echo "  make install-standalone - Install autoencoders without mura/qg"
	@echo ""
	@echo "Training:"
	@echo "  make train            - Train with default config"
	@echo "  make train-mnist      - Train MNIST model on CPU"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            - Clean caches"

# ========================================
# Installation
# ========================================
install: install-deps install-packages install-autoencoders
	@echo "✅ Installation complete!"
	@echo "Don't forget to run: wandb login && huggingface-cli login"

install-deps:
	@command -v $(UV) >/dev/null 2>&1 || $(SYS_PYTHON) -m pip install --user uv
	@[ -d "$(VENV)" ] || $(UV) venv $(VENV)

install-packages: install-deps
	@echo "📦 Installing external packages..."
	@chmod +x $(INSTALL)/install_packages.sh
	@$(INSTALL)/install_packages.sh $(PYTHON)

install-autoencoders:
	@echo "📦 Installing autoencoders..."
	$(UV) pip install -r $(INSTALL)/requirements.txt --python $(PYTHON)
	$(MAKE) py3-conf
	$(UV) pip install -e . --python $(PYTHON)

py3-conf:
	-ln -sf $(BASE_PYDIR)/python3-config $(VENV)/bin/python3-config

# ========================================
# Training
# ========================================
train: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train

train-mnist: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train \
		trainer.accelerator=cpu \
		trainer.max_epochs=0

train-smallqg: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train \
		data=small_forced_turbulence \
		model=mnist64 \
		trainer.accelerator=cpu \
		trainer.max_epochs=200

train-qgae: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train \
		data=forced_turbulence \
		model=spatial \
		trainer.max_epochs=200

train-rpnae: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train \
		data=rpn_turbulence \
		model=spatial \
		trainer.max_epochs=200


### =========================== MMAI APR26 ============================

train-diffusion: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train \
		exp=mmai_apr26/diffusion \
		trainer.max_epochs=90

train-operator-diffusion: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train \
		exp=mmai_apr26/0_vision \
		trainer.max_epochs=90

train-operator-srdit: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train \
		exp=mmai_apr26/0_vision model=mmai_apr26/operator_srdit \
		trainer.max_epochs=120

train-llm: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train \
		exp=mmai_apr26/1_llm \
		trainer.max_epochs=250

train-llm-wo: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train \
		exp=mmai_apr26/1_llm model=mmai_apr26/llm_wo_sym\
		trainer.max_epochs=250

train-qwen_llm: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train \
		exp=mmai_apr26/1_llm model=mmai_apr26/qwen_llm \
		trainer.max_epochs=120

train-vlm-base: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train \
		exp=mmai_apr26/2_vlm model=mmai_apr26/vlm_baseline\
		trainer.max_epochs=100

train-vlm: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train \
		exp=mmai_apr26/2_vlm \
		trainer.max_epochs=40

train-vlm-wo: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train \
		exp=mmai_apr26/2_vlm model=mmai_apr26/vlm_wo_sym \
		trainer.max_epochs=100

train-operator-basediffusion: install
	source "$(VENV)/bin/activate" && \
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train \
		exp=mmai_apr26/0_vision model=mmai_apr26/operator_pixart\
		trainer.max_epochs=2

# ========================================
# Cleanup
# ========================================
clean:
	rm -rf data/qg_cache/
	rm -rf ~/.cache/torch_extensions/py*
	@echo "✅ All caches cleaned"

# ========================================
# Cluster (Slurm)
# ========================================
sinstall:
	source ${INSTALL}/module.sh && $(MAKE) install

strain:
	source ${INSTALL}/module.sh && $(MAKE) train