SHELL := /bin/bash
PYTHON ?= python

.PHONY: install data train benchmark deploy push-deploy lint format clean

install:
	$(PYTHON) -m pip install -r src/install/requirements.txt
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

train:
	HYDRA_FULL_ERROR=1 $(PYTHON) -m src.autoencoders.train trainer.max_epochs=1 run.name=local-debug run.tags=[local,debug] wandb.mode=online

benchmark: data
	pytest -m benchmark -s

push-deploy:
	$(PYTHON) -m src.deploy.push_to_deploy

deploy:
	$(PYTHON) -m src.deploy.hpc_deploy

lint:
	@echo "No lint tooling configured. Install ruff or similar to enable this target."

format:
	@echo "No formatter configured. Install ruff or similar to enable this target."

clean:
	rm -rf runs/.hydra runs/multirun
