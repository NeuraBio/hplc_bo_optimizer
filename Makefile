DOCKER_COMPOSE = docker compose
DOCKER_SERVICE = hplc-dev
DOCKER_EXEC = $(DOCKER_COMPOSE) exec $(DOCKER_SERVICE)
POETRY_RUN = $(DOCKER_EXEC) poetry run
# Configurable variables (can override via CLI)
CLIENT ?= NuraxsDemo
EXPERIMENT ?= HPLC-Optimization
TRIAL_ID ?= 0
RT_FILE ?= results/rt.csv
GRADIENT_FILE ?=
STREAMLIT_PORT ?= 8501

docker-build:
	$(DOCKER_COMPOSE) build --no-cache $(DOCKER_SERVICE)

docker-up:
	$(DOCKER_COMPOSE) up -d --force-recreate $(DOCKER_SERVICE)

docker-down:
	$(DOCKER_COMPOSE) down -v

# Development workflow targets
dev-start:
	@echo "Starting development environment..."
	$(DOCKER_COMPOSE) up -d
	@echo "Container started with Streamlit running automatically"
	@echo "Access the app at http://localhost:8501"
	@echo "View logs with 'make dev-logs'"

dev-restart:
	@echo "Restarting development environment..."
	$(DOCKER_COMPOSE) down
	$(DOCKER_COMPOSE) up -d
	@echo "Container restarted with Streamlit running automatically"
	@echo "Access the app at http://localhost:8501"
	@echo "View logs with 'make dev-logs'"

# Manually start Streamlit in development mode (only needed if the automatic startup fails)
streamlit-dev:
	@echo "Manually starting Streamlit in development mode..."
	docker exec -it hplc-bo-dev bash -c "cd /app && mkdir -p logs && poetry run streamlit run app/main.py --server.port=8501 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --server.runOnSave=true"
	@echo "Streamlit app started in background"
	@echo "Access the app at http://localhost:8501"
	@echo "View logs with 'make dev-logs'"

# View container logs
dev-logs:
	@echo "Viewing container logs..."
	$(DOCKER_COMPOSE) logs -f

# View application logs
app-logs:
	@echo "Viewing application logs..."
	docker exec -it hplc-bo-dev bash -c "cd /app && mkdir -p logs && tail -f logs/streamlit_app.log 2>/dev/null || echo 'No log file found yet'"

# Open a shell in the development container
dev-shell:
	@echo "Opening shell in development container..."
	docker exec -it hplc-bo-dev bash

# Stop the Streamlit app
streamlit-stop:
	@echo "Stopping Streamlit app..."
	docker exec -it hplc-bo-dev bash -c "pkill -f 'streamlit run' || echo 'No Streamlit process found'"
	@echo "Streamlit app stopped"

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
	@echo "Generating suggestion for next trial..."
	$(POETRY_RUN) python hplc_optimize.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) suggest

report:
	@echo "Reporting results for trial $(TRIAL_ID)..."
	$(POETRY_RUN) python hplc_optimize.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) report --trial_id $(TRIAL_ID) --rt_file $(RT_FILE) $(if $(GRADIENT_FILE),--gradient_file $(GRADIENT_FILE))

export-results:
	@echo "Exporting all trial results..."
	$(POETRY_RUN) python hplc_optimize.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) export

# Validation targets
PDF_DIR ?= data/pdfs
VALIDATION_OUTPUT ?= hplc_optimization/validation
BO_OUTPUT ?= hplc_optimization/bo_simulation
CHEMIST_RATINGS ?=
N_SUGGESTIONS ?= 10

validate:
	@echo "Running validation on historical PDF data..."
	$(POETRY_RUN) python hplc_optimize.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) validate --pdf_dir $(PDF_DIR)

simulate:
	@echo "Running Bayesian Optimization simulation..."
	$(POETRY_RUN) python hplc_optimize.py --client_lab $(CLIENT) --experiment $(EXPERIMENT) simulate --n_trials $(N_SUGGESTIONS)

# Streamlit app targets
streamlit-app:
	@echo "Starting Streamlit app..."
	$(POETRY_RUN) streamlit run app/main.py --server.port=$(STREAMLIT_PORT)

# Start Streamlit app in production mode
streamlit-prod:
	@echo "Starting Streamlit app in production mode..."
	$(POETRY_RUN) streamlit run app/main.py --server.port=$(STREAMLIT_PORT) --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false

.PHONY: docker-build docker-up docker-down docker-shell docker-setup-env format lint fix-lint pre-commit test clean suggest report export-results validate simulate docker-start docker-stop docker-clean streamlit-app streamlit-prod dev-start dev-restart dev-logs app-logs dev-shell streamlit-dev streamlit-stop
