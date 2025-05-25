DOCKER_COMPOSE = docker compose
DOCKER_SERVICE = hplc-dev
DOCKER_EXEC = $(DOCKER_COMPOSE) exec $(DOCKER_SERVICE)
POETRY_RUN = $(DOCKER_EXEC) poetry run

docker-build:
	$(DOCKER_COMPOSE) build --no-cache $(DOCKER_SERVICE)

docker-up:
	$(DOCKER_COMPOSE) up -d --force-recreate $(DOCKER_SERVICE)

docker-down:
	$(DOCKER_COMPOSE) down -v

docker-shell:
	$(DOCKER_EXEC) bash

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

.PHONY: docker-build docker-up docker-down docker-shell format lint fix-lint pre-commit test clean
