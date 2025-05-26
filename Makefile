DOCKER_COMPOSE = docker compose
DOCKER_SERVICE = hplc-dev
DOCKER_EXEC = $(DOCKER_COMPOSE) exec $(DOCKER_SERVICE)
POETRY_RUN = $(DOCKER_EXEC) poetry run
# Configurable variables (can override via CLI)
CLIENT ?= Pfizer
EXPERIMENT ?= ImpurityTest
TRIAL_ID ?= 0
RT_FILE ?= results/rt.csv
GRADIENT_FILE ?=

docker-build:
	$(DOCKER_COMPOSE) build --no-cache $(DOCKER_SERVICE)

docker-up:
	$(DOCKER_COMPOSE) up -d --force-recreate $(DOCKER_SERVICE)

docker-down:
	$(DOCKER_COMPOSE) down -v

docker-shell:
	$(DOCKER_EXEC) bash

docker-setup-env:
	@echo "Ensuring poetry.lock is up-to-date and installing/syncing dependencies in Docker..."
	$(DOCKER_EXEC) poetry lock --no-update
	$(DOCKER_EXEC) bash -c "PIP_NO_BINARY=cryptography CRYPTOGRAPHY_DONT_BUILD_RUST=0 poetry install --with dev --no-root"
	@echo "Environment setup complete in Docker."

format:
	@echo "Formatting code inside Docker..."
	$(POETRY_RUN) black .
	$(POETRY_RUN) isort .
	$(POETRY_RUN) ruff format .

lint:
	@echo "Linting code inside Docker..."
	$(POETRY_RUN) ruff check .

fix-lint:
	@echo "Fixing lint issues inside Docker..."
	$(POETRY_RUN) ruff check --fix .

pre-commit:
	@echo "Running pre-commit hooks inside Docker..."
	$(POETRY_RUN) pre-commit run --all-files --show-diff-on-failure

test:
	@echo "Running tests inside Docker..."
	$(POETRY_RUN) python -m pytest

clean:
	rm -f hplc_results.csv hplc_convergence.png
	find . -name '*.so' -delete
	find . -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +

suggest:
	$(POETRY_RUN) python hplc_bo/run_trial.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) suggest

report:
	$(POETRY_RUN) python hplc_bo/run_trial.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) report --trial_id $(TRIAL_ID) --rt_file $(RT_FILE) $(if $(GRADIENT_FILE),--gradient_file $(GRADIENT_FILE))

export-results:
	$(POETRY_RUN) python hplc_bo/run_trial.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) export

.PHONY: docker-build docker-up docker-down docker-shell docker-setup-env format lint fix-lint pre-commit test clean suggest report export-results
