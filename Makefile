# Makefile for HPLC BO Optimizer

# Configurable variables (can override via CLI)
CLIENT ?= Pfizer
EXPERIMENT ?= ImpurityTest

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

run-interactive:
	$(POETRY_RUN) python hplc_bo/run_trial.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) --interactive

run-mock:
	$(POETRY_RUN) python hplc_bo/run_trial.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) --mock 10

export-results:
	$(POETRY_RUN) python hplc_bo/run_trial.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) --export

docker-format:
	$(POETRY_RUN) black .
	$(POETRY_RUN) isort .
	$(POETRY_RUN) ruff format .

docker-lint:
	$(POETRY_RUN) ruff check .

# --- Clean ---

clean:
	rm -f hplc_results.csv hplc_convergence.png

.PHONY: docker-build docker-up docker-down docker-shell install format lint run-interactive run-mock export-results clean docker-format docker-lint
