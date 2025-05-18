import matplotlib.pyplot as plt
import optuna
import pandas as pd

from hplc_bo.lock_manager import LockAcquisitionError, LockManager
from hplc_bo.optimizer import suggest_params
from hplc_bo.scoring import mock_score
from hplc_bo.study_registry import log_study_run


class StudyRunner:
    def __init__(
        self,
        client_lab: str,
        experiment: str,
        storage_path: str = "sqlite:///optuna_storage/hplc_study.db",
    ):
        self.study_name = f"{client_lab}_{experiment}".lower().replace(" ", "_")
        self.storage = storage_path
        self.lock = LockManager(self.study_name)
        self.study = self._load_or_create_study()
        self.client_lab = client_lab
        self.experiment = experiment

    def _load_or_create_study(self):
        try:
            return optuna.load_study(study_name=self.study_name, storage=self.storage)
        except KeyError:
            return optuna.create_study(
                direction="maximize",
                study_name=self.study_name,
                storage=self.storage,
            )

    def run_interactive(self):
        try:
            with self.lock.acquire():
                trial = self.study.ask()
                params = suggest_params(trial)
                print("\n=== Suggested Parameters ===")
                for k, v in params.items():
                    print(f"{k}: {v}")
                print("===========================\n")
                score = float(input("Enter score from chemist: "))
                trial.set_user_attr("params", params)
                trial.set_user_attr("score", score)
                self.study.tell(trial, score)
                log_study_run(self.client_lab, self.experiment, self.study_name, 1, "interactive")
                print("Trial updated.")
        except LockAcquisitionError as e:
            print(f"[ERROR] {e}")

    def run_mock_trials(self, n_trials: int = 10):
        try:
            with self.lock.acquire():
                for _ in range(n_trials):
                    trial = self.study.ask()
                    params = suggest_params(trial)
                    score = mock_score(params)
                    trial.set_user_attr("params", params)
                    trial.set_user_attr("score", score)
                    self.study.tell(trial, score)
                log_study_run(self.client_lab, self.experiment, self.study_name, n_trials, "mock")
                print(f"Ran {n_trials} mock trials.")
        except LockAcquisitionError as e:
            print(f"[ERROR] {e}")

    def export_results(self, output_csv="hplc_results.csv", plot_path="hplc_convergence.png"):
        records = []
        for t in self.study.trials:
            if t.value is not None:
                r = t.user_attrs.get("params", t.params)
                r["score"] = t.value
                r["trial_number"] = t.number
                records.append(r)
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        print(f"[✓] Exported {output_csv}")

        if not df.empty:
            plt.plot(df["trial_number"], df["score"], marker="o")
            plt.xlabel("Trial")
            plt.ylabel("Score")
            plt.title("Optimization Progress")
            plt.grid(True)
            plt.savefig(plot_path)
            print(f"[✓] Saved {plot_path}")
