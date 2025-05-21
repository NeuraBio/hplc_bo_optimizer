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
```

---

## 📊 Architecture Overview

```
hplc_bo/
├── run_trial.py          # CLI entrypoint
├── config.py             # Param search space
├── optimizer.py          # Param suggestion logic
├── param_types.py        # TypedDict for params
├── gradient_utils.py     # Scoring, CSV I/O
├── study_access.py       # Safe Optuna interaction
├── study_runner.py       # CLI orchestrator
├── lock_manager.py       # Locking
├── study_registry.py     # Run registry
```

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
