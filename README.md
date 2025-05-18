Here are your finalized files:

---

## ✅ Updated `README.md`

````markdown
# 🧪 HPLC Bayesian Optimization Toolkit

This repository helps automate **HPLC method development** using **Bayesian Optimization (BO)**. The core idea: given a set of tunable method parameters, use machine learning to propose better combinations over time, guided by chemist feedback or simulated scores.

---

## 📌 Problem Statement

**Goal:** *Given a set of controllable inputs (method parameters), find the combination that maximizes a performance score (e.g., impurity resolution or chromatogram quality).*

| Component        | Example                                                                 |
|------------------|-------------------------------------------------------------------------|
| **Input space**  | Gradient profile, flow rate, column temp, additive                      |
| **Output**       | Score derived from chromatogram (e.g., resolution, purity)              |
| **Constraints**  | Run time < 30 min, resolution > 1.5                                     |
| **Objective**    | Maximize separation quality, minimize runtime and overlaps              |
| **Surrogate**    | TPE / Random Forest / Gaussian Process                                  |
| **Acquisition**  | Expected Improvement or Thompson Sampling                               |

---

## 🎯 Scoring Functions (BO Objective)

| Score Type                  | Captures                      | When to Use                             |
|----------------------------|-------------------------------|-----------------------------------------|
| Avg Resolution Factor      | Peak separation quality       | High purity, good peak shapes           |
| # of Resolved Peaks        | Separation quantity           | Detecting more degradants               |
| Composite (Res – Penalty)  | Balanced trade-off            | Runtime, overlaps, and purity balance   |
| Peak Purity Score          | Spectral purity               | When MS/DAD data is available           |

Example scoring formula:

```python
score = (
    sum(res_factors) / len(res_factors)
    - 0.5 * num_overlaps
    - 0.1 * total_runtime_minutes
)
````

---

## 🎛️ Tunable Inputs

| Input Parameter | Type        | Example Range     |
| --------------- | ----------- | ----------------- |
| %B\_start       | Continuous  | 20–40             |
| %B\_end         | Continuous  | 60–90             |
| Gradient\_time  | Continuous  | 10–30 min         |
| Flow\_rate      | Continuous  | 0.2–1.0 mL/min    |
| Column\_temp    | Continuous  | 25–60 °C          |
| Additive        | Categorical | TFA / Formic Acid |

---

## 📦 Current Capabilities

* ✅ Bayesian optimization via Optuna
* ✅ Chemist-in-the-loop CLI (`--interactive`)
* ✅ Simulated scoring (`--mock`)
* ✅ Persistent study state with SQLite
* ✅ Trial history export to CSV + plot
* ✅ Per-study concurrency control via file-based locks
* ✅ Study registry logging (user, timestamp, trials, mode)
* ⏳ Streamlit UI (coming soon)
* 🧪 Real chromatogram-based scoring (planned)

---

## 🔁 Project Stages

| Stage       | Goal                                                                                               | Status    |
| ----------- | -------------------------------------------------------------------------------------------------- | --------- |
| **Stage 1** | Get a basic parameter-suggestion pipeline working using Optuna with a mock scoring function.       | ✅ Done    |
| **Stage 2** | Add persistence (SQLite), and allow interactive CLI where chemists enter scores based on lab runs. | ✅ Done    |
| **Stage 3** | Build a simple Streamlit UI to allow chemists to run and score experiments through a browser.      | 🔄 Next   |
| **Stage 4** | Integrate real scoring (e.g., parsing resolution from chromatograms or instrument data).           | 🧪 Future |
| **Stage 5** | Extend to richer acquisition/BO strategies and support more method types.                          | 🔮 Future |

---

## 🚀 Quickstart (with Docker)

### Step 1: Start container

```bash
make docker-build
make docker-up
```

### Step 2: Run CLI loop

```bash
make run-interactive CLIENT=Pfizer EXPERIMENT=ImpurityTest
```

You’ll be shown HPLC parameters. Run the method in-lab and enter a numeric score when prompted.

### Step 3: Export results

```bash
make export-results CLIENT=Pfizer EXPERIMENT=ImpurityTest
```

### For simulation:

```bash
make run-mock CLIENT=Pfizer EXPERIMENT=ImpurityTest
```

---

## 🧪 Architecture Overview

* **Optuna**: Trial generation + optimization
* **Poetry**: Dependency + environment management
* **Docker**: Reproducible setup
* **Matplotlib**: Score visualization
* **FileLock**: Per-study mutex to prevent concurrent access
* **Streamlit**: Visual chemist UI (coming soon)

---

## 📁 Project Structure

```
hplc_bo/
├── run_trial.py        # CLI entrypoint
├── optimizer.py        # Suggestion + BO loop
├── scoring.py          # Score calculation
├── config.py           # Search space
├── lock_manager.py     # File-based mutex for each study
├── study_registry.py   # CSV registry logger for all runs
├── param_types.py      # Strong param typing
Makefile                # Dev & Docker commands
```

---

## 🗂️ Study Tracking & Locking

Each study is uniquely identified by a combination of:

* `client_lab` (e.g., `"Pfizer"`)
* `experiment` (e.g., `"Impurity Test"`)

This becomes the internal `study_name` (`"pfizer_impurity_test"`), which is used:

* For SQLite-backed Optuna storage
* For locking (`optuna_storage/locks/`)
* In the registry (`optuna_storage/study_registry.csv`)

### 🔒 Concurrency Control

Only one user/process can interactively score a given study at a time:

```bash
[ERROR] Study 'pfizer_impurity_test' is locked by alice (PID 1234).
```

---

## 📚 Scientific References

* [Bayesian optimization for method development in liquid chromatography](https://www.sciencedirect.com/science/article/pii/S0021967321007524)
* [QSRR + AI in Chromatography (2024)](https://www.sciencedirect.com/science/article/pii/S2095177924002521)
* Tools like **DryLab**, **AutoChrom**, **Fusion QbD** apply similar ideas — but are closed-source.

---

## 👩‍🔬 For Chemists

You can interact via CLI or, soon, a browser-based UI:

* Use `make run-interactive` to receive optimized suggestions
* Enter a numeric score based on your lab run
* Export results with `make export-results`

---

## 🧠 For Developers

* CLI interface via `run_trial.py`
* Optuna-based study tracking
* Trial-level locking
* Study logging via CSV
* Easy to extend for UI or REST APIs

📘 See [CONTRIBUTE.md](CONTRIBUTE.md) for full developer setup and debugging workflow.

---

## 🛣️ Roadmap

* [x] CLI with chemist input loop
* [x] Mocked optimization + scoring
* [x] Export results as CSV/plot
* [x] Study registry and lock protection
* [ ] Streamlit chemist UI
* [ ] Real data scoring from instrument output
* [ ] Auto-parsing chromatograms / peak table
* [ ] Model selection, acquisition strategy control

---

## 📝 License

MIT License

````

---

## ✅ `CONTRIBUTE.md`

```markdown
# 🧠 Contributing to HPLC BO Toolkit

We welcome developers, chemists, and data scientists to help evolve this platform.

---

## 🧰 Local Development Setup

### Poetry-based setup

```bash
poetry install --with dev
````

### Docker-based (recommended)

```bash
make docker-build
make docker-up
make docker-shell
```

---

## 🧪 Developer Commands

```bash
make format       # Run black, isort, and ruff formatter
make lint         # Run ruff checks
make run-mock CLIENT=Pfizer EXPERIMENT=Test
make run-interactive CLIENT=Amgen EXPERIMENT=StabilityStudy
make export-results CLIENT=LabX EXPERIMENT=Batch42
```

All runs are scoped to a client + experiment, logged in the study registry.

---

## 🧭 Debugging Tips

### VS Code / PyCharm

* Mark `hplc_bo/` as source root
* Use Poetry interpreter or Docker container
* Set breakpoints inside `study_runner.py`, `scoring.py`, etc.
* Run/debug `run_trial.py` with CLI args like:

```bash
--client_lab Pfizer --experiment ImpurityTest --interactive
```

---

## 🔐 Locking & Concurrency

* Each study is guarded with a file lock (`optuna_storage/locks/`)
* Only one user/process can run a study interactively at once
* Lock metadata is tracked (`lockmeta.json`)

---

## 📒 Study Registry

All runs are logged to:

```
optuna_storage/study_registry.csv
```

| Column       | Meaning                      |
| ------------ | ---------------------------- |
| `timestamp`  | Time the run was executed    |
| `client`     | Lab name                     |
| `experiment` | Study description            |
| `study_name` | Normalized internal name     |
| `trials`     | Number of trials in this run |
| `status`     | Usually "completed"          |
| `run_mode`   | `"mock"` or `"interactive"`  |
| `user`       | Who triggered the run        |

---

## 🧪 Testing

Planned: `pytest`-based test suite, type-checking, and CI integration

---

## 📫 Questions?

Please open an issue or submit a PR with your ideas!

```
