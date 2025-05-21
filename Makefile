# Makefile for HPLC BO Optimizer

# Configurable variables (can override via CLI)
CLIENT ?= Pfizer
EXPERIMENT ?= ImpurityTest
TRIAL_ID ?= 0
RT_FILE ?= results/rt.csv
GRADIENT_FILE ?=

# Docker commands
DOCKER_COMPOSE = docker compose
DOCKER_SERVICE = hplc-dev
DOCKER_EXEC = $(DOCKER_COMPOSE) exec $(DOCKER_SERVICE)
POETRY_RUN = $(DOCKER_EXEC) poetry run

# --- Docker Tasks ---

docker-build:
	$(DOCKER_COMPOSE) build

docker-up:
	$(DOCKER_COMPOSE) up -d

docker-down:
	$(DOCKER_COMPOSE) down

docker-shell:
	$(DOCKER_EXEC) bash

# --- Host Dev Setup ---

install:
	poetry install --with dev

format:
	poetry run black .
	poetry run isort .
	poetry run ruff format .

lint:
	poetry run ruff check .

# --- Docker-Based CLI Commands with Configurable Inputs ---

suggest:
	$(POETRY_RUN) python hplc_bo/run_trial.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) suggest

report:
	$(POETRY_RUN) python hplc_bo/run_trial.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) report --trial_id $(TRIAL_ID) --rt_file $(RT_FILE) $(if $(GRADIENT_FILE),--gradient_file $(GRADIENT_FILE))

export-results:
	$(POETRY_RUN) python hplc_bo/run_trial.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) export

run-historical:
	$(POETRY_RUN) python hplc_bo/run_trial.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) run_historical --csv historical_data.csv

# --- Clean ---

clean:
	rm -f hplc_results.csv hplc_convergence.png

.PHONY: docker-build docker-up docker-down docker-shell install format lint suggest report export-results run-historical clean
