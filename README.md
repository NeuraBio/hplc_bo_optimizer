# 🧪 HPLC Bayesian Optimization Toolkit

This repository helps automate **HPLC method development** using **Bayesian Optimization (BO)**. The core idea: given a set of tunable method parameters, use machine learning to propose better combinations over time, guided by chemist feedback or simulated scores.

---

## 📌 Problem Statement

**Goal:** *Given a set of controllable inputs (method parameters), find the combination that maximizes a performance score (e.g., impurity resolution or chromatogram quality).*

| Component       | Example                                                    |
| --------------- | ---------------------------------------------------------- |
| **Input space** | Gradient profile, flow rate, column temp                   |
| **Output**      | Score derived from chromatogram (e.g., resolution, purity) |
| **Constraints** | Run time < 30 min, resolution > 1.5                        |
| **Objective**   | Maximize separation quality, minimize runtime and overlaps |
| **Surrogate**   | TPE / Random Forest / Gaussian Process                     |
| **Acquisition** | Expected Improvement or Thompson Sampling                  |

---

## 🌟 Scoring Functions (BO Objective)

| Score Type                | Captures                | When to Use                           |
| ------------------------- | ----------------------- | ------------------------------------- |
| Avg Resolution Factor     | Peak separation quality | High purity, good peak shapes         |
| # of Resolved Peaks       | Separation quantity     | Detecting more degradants             |
| Composite (Res – Penalty) | Balanced trade-off      | Runtime, overlaps, and purity balance |
| Peak Purity Score         | Spectral purity         | When MS/DAD data is available         |

Score example:

```python
score = rt_score - zigzag_penalty
```

---

## 🔛 Tunable Inputs

| Input Parameter | Type       | Example Range      |
| --------------- | ---------- | ------------------ |
| Flow\_rate      | Continuous | 0.2–1.5 mL/min     |
| Column\_temp    | Continuous | 25–60 °C           |
| pH              | Continuous | 2.0–10             |
| %B\[t0-t4]      | Continuous | 0–100 (at anchors) |

Gradient is modeled using 5 anchor timepoints with corresponding %B values.

---

## 🔍 Validation Workflow

The toolkit includes a validation framework to assess the scoring function and Bayesian Optimization approach using historical HPLC data from PDF reports.

### PDF Data Extraction

The system can extract key data from HPLC PDF reports:

- Retention times (RT)
- Peak widths and tailing factors
- Column temperature and flow rate
- Solvent names (A and B)
- Gradient table data

### Validation Process

The validation workflow has two phases:

#### Phase 1: Scoring Validation

Validates that the scoring function aligns with expert chemist evaluations:

```bash
# Process a directory of PDF reports
make validate-score PDF_DIR=path/to/pdfs VALIDATION_OUTPUT=validation_results
```

This generates:
- Score distribution analysis
- Correlation between scores and method parameters
- Interactive HTML report for browsing results

#### Phase 2: Bayesian Optimization Validation

Assesses how BO would have performed on historical data:

```bash
# Run BO validation on previously scored PDFs
make validate-bo VALIDATION_OUTPUT=validation_results BO_OUTPUT=bo_validation
```

This produces:
- Convergence analysis (trials needed to reach optimal methods)
- Parameter importance plots
- Contour plots showing the optimization landscape
- Suggestions for next trials

#### Full Validation

Run both phases in sequence:

```bash
# Run complete validation workflow
make validate-full PDF_DIR=path/to/pdfs
```

---

## 📦 Current Capabilities

* ✅ Bayesian optimization via Optuna
* ✅ CLI-based interaction (suggest/report/export)
* ✅ Study persistence using SQLite
* ✅ Trial storage via CSV per study
* ✅ Scoring incorporates gradient zigzag penalty
* ✅ File-locking for concurrency protection
* ✅ Study registry for audit/logging

---

## 🔀 Flow Summary

1. `make suggest`: Suggest a new method
2. Chemist runs it (or skips if it looks bad)
3. `make report`: Report RT results and score it
4. BO updates its model accordingly

---

## 🚀 Quickstart

```bash
make docker-build
make docker-up
make suggest CLIENT=Pfizer EXPERIMENT=DegradantA
make report TRIAL_ID=0 RT_FILE=results/rt.csv
make export-results CLIENT=Pfizer EXPERIMENT=DegradantA

This project uses Docker for development to ensure consistent environments:

```bash
# 1. Build the development container
docker compose build hplc-dev

# 2. Start the container
docker compose up -d hplc-dev

# 3. Install dependencies inside the container
docker compose exec hplc-dev bash -c "cd /app && poetry install --with dev"

# Or use the Makefile shortcut
make docker-setup-env

# 4. Configure Git for the mounted repository (one-time setup)
docker compose exec hplc-dev git config --global --add safe.directory /app
```

## 🔄 Unified Workflow

We've integrated validation, simulation, and human-in-the-loop trials into a cohesive workflow using the new `hplc_optimize.py` script:

### 1. Validate Historical Data

Process historical PDF reports to extract data and compute scores:

```bash
python hplc_optimize.py --client_lab YourLab --experiment HPLC-1 validate --pdf_dir validation_pdfs
```

This will:
- Process all PDFs in the directory
- Extract chromatography data (retention times, peak widths, etc.)
- Compute scores using the scoring function
- Generate visualizations and HTML reports
- Save validation results for later use

### 2. Simulate BO Performance

Simulate how Bayesian Optimization would have performed on historical data:

```bash
python hplc_optimize.py --client_lab YourLab --experiment HPLC-1 simulate --n_trials 40
```

This will:
- Simulate BO suggesting parameters sequentially
- Match suggestions to closest historical runs
- Analyze convergence and efficiency compared to manual experimentation
- Generate visualizations and HTML reports

### 3. Run Human-in-the-Loop Trials

Get suggestions for new experiments and report results:

```bash
# Get a suggestion for the next trial
python hplc_optimize.py --client_lab YourLab --experiment HPLC-1 suggest

# Report results after running the experiment
python hplc_optimize.py --client_lab YourLab --experiment HPLC-1 report --trial_id 1 --rt_file results/rt_data.csv

# Export all results to CSV and generate plots
python hplc_optimize.py --client_lab YourLab --experiment HPLC-1 export
```

This workflow allows you to:
- Get BO-suggested parameters for new experiments
- Run the experiments in the lab
- Report results back to the system
- Continuously improve suggestions based on feedback

### Development Workflow

```bash
# Start the container if not running
make docker-up

# Get a shell inside the container
make docker-shell

# Run tests
make test

# Run linting and formatting
make pre-commit
make format
```

### Troubleshooting

If you encounter issues with dependencies, especially native ones like `cryptography`:

```bash
# Manually reinstall problematic packages
docker compose exec hplc-dev bash -c "PIP_NO_BINARY=cryptography CRYPTOGRAPHY_DONT_BUILD_RUST=0 poetry add cryptography==42.0.8 pdfplumber==0.11.6"
```

---

## 📊 Architecture Overview

```
/
├── hplc_optimize.py      # Unified CLI entrypoint
├── run_bo_simulation.py   # Script to run BO simulation
├── hplc_bo/
    ├── workflow.py         # Integrated workflow manager
    ├── validation.py       # Historical data validation
    ├── validation_bo.py    # BO validation on historical data
    ├── bo_simulation.py    # BO simulation with historical matching
    ├── run_trial.py        # Legacy CLI entrypoint
    ├── config.py           # Param search space
    ├── optimizer.py        # Param suggestion logic
    ├── param_types.py      # TypedDict for params
    ├── gradient_utils.py   # Scoring, CSV I/O
    ├── study_access.py     # Safe Optuna interaction
    ├── study_runner.py     # Study lifecycle management
    ├── lock_manager.py     # Locking
    ├── study_registry.py   # Run registry
```

### Key Components

- **workflow.py**: New unified interface for the entire HPLC optimization process
- **validation.py**: Processes historical PDF reports and computes scores
- **validation_bo.py**: Analyzes BO performance on historical data
- **bo_simulation.py**: Simulates BO suggestions and matches to historical runs
- **study_runner.py**: Manages Optuna studies for human-in-the-loop trials
- **optimizer.py**: Core parameter suggestion logic using Optuna
- **gradient_utils.py**: Handles gradient profiles and scoring functions

---

## 📂 Suggested Trial CSV Schema

Written to: `optuna_storage/suggested_trials/<study>.csv`

| Field         | Meaning                          |
| ------------- | -------------------------------- |
| trial\_number | Optuna trial ID                  |
| bo\_gradient  | List of anchor (t, %B) tuples    |
| gradient      | Interpolated %B over 40 min      |
| flow\_rate    | mL/min                           |
| pH            | Buffer pH                        |
| column\_temp  | Column temperature               |
| score         | Assigned score after RT analysis |
| rt\_list      | List of detected RTs             |

---

## 🔒 Locking & Concurrency

* Each study gets a file lock (`optuna_storage/locks/<study>.lock`)
* CLI actions acquire this lock before calling Optuna APIs
* Prevents race conditions in shared environments

---

## 📊 Study Registry

All CLI runs are logged in:

```csv
optuna_storage/study_registry.csv
```

| Field       | Description                |
| ----------- | -------------------------- |
| timestamp   | When the run was triggered |
| client      | Pfizer, Amgen, etc         |
| experiment  | Use case or molecule name  |
| study\_name | Normalized ID              |
| trials      | # of trials added this run |
| run\_mode   | suggest / report / export  |
| status      | always "completed" for now |
| user        | OS-level user              |

---

## 🔸 CLI Commands

```bash
make suggest
make report TRIAL_ID=4 RT_FILE=rt.csv
make export-results
make run-historical
```

Set defaults via:

```makefile
CLIENT=Pfizer EXPERIMENT=DegradantA
```

---

## 🚜 Roadmap

* [x] Replace `--interactive` CLI with suggest/report decoupling
* [x] TrialRecord class to track metadata + CSV
* [x] Add locking + StudyAccess wrapper
* [ ] Add RT parsing from full instrument report
* [ ] Streamlit chemist frontend
* [ ] Integrate resolution + peak shape metrics

---

## 🙋️ Questions / Feedback?

Open an issue or start a PR discussion.
