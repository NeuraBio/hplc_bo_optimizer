# run_trial.py

import argparse

import optuna
import pandas as pd

from hplc_bo.optimizer import suggest_params
from hplc_bo.param_types import OptimizationParams

STORAGE = "sqlite:///hplc_study.db"
STUDY_NAME = "hplc_optimization"


def ask_next():
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    trial = study.ask()
    params = suggest_params(trial)
    print("\n=== Suggested Parameters ===")
    for k, v in params.items():
        print(f"{k}: {v}")
    print("===========================")
    return trial, params


def tell_result(trial, score: float, params: OptimizationParams):
    trial.set_user_attr("params", params)
    trial.set_user_attr("score", score)
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    study.tell(trial, score)


def export_results():
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    records = []
    for t in study.trials:
        if t.value is not None:
            r = t.user_attrs.get("params", t.params)
            r["score"] = t.value
            r["trial_number"] = t.number
            records.append(r)
    df = pd.DataFrame(records)
    df.to_csv("hplc_results.csv", index=False)
    print("Exported hplc_results.csv")

    try:
        import matplotlib.pyplot as plt

        plt.plot(df["trial_number"], df["score"], marker="o")
        plt.xlabel("Trial")
        plt.ylabel("Score")
        plt.title("Optimization Progress")
        plt.grid(True)
        plt.savefig("hplc_convergence.png")
        print("Saved hplc_convergence.png")
    except ImportError:
        print("matplotlib not installed, skipping plot.")


def mock_loop(n_trials: int = 10):
    from hplc_bo.scoring import mock_score

    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    for _ in range(n_trials):
        trial = study.ask()
        params = suggest_params(trial)
        score = mock_score(params)
        trial.set_user_attr("params", params)
        trial.set_user_attr("score", score)
        study.tell(trial, score)
    print(f"Ran {n_trials} mock trials.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Run one ask/tell loop")
    parser.add_argument("--init", action="store_true", help="Initialize new study")
    parser.add_argument("--export", action="store_true", help="Export trial history")
    parser.add_argument("--mock", type=int, help="Run N mock trials")
    args = parser.parse_args()

    if args.init:
        optuna.create_study(direction="maximize", study_name=STUDY_NAME, storage=STORAGE)
        print("Initialized new Optuna study.")

    elif args.interactive:
        trial, params = ask_next()
        score = float(input("Enter score from chemist: "))
        tell_result(trial, score, params)
        print("Trial updated.")

    elif args.export:
        export_results()

    elif args.mock:
        mock_loop(args.mock)

    else:
        print("Specify one of --init, --interactive, --export, or --mock <N>")
