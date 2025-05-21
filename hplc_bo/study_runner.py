import json

import matplotlib.pyplot as plt
import optuna
import pandas as pd

from hplc_bo.gradient_utils import (
    TrialRecord,
    compute_total_score,
    load_rt_list_from_csv,
    penalize_gradient_zigzags,
)
from hplc_bo.lock_manager import LockManager
from hplc_bo.study_access import StudyAccess
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
        self.access = StudyAccess(self.study, self.lock, self.study_name)
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

    def suggest(self):
        record = self.access.ask()
        penalty = penalize_gradient_zigzags(record.bo_gradient)

        if penalty > 50.0:
            print(f"⚠️ High zigzag penalty ({penalty:.1f}).")
            confirm = input("Auto-log this trial with -inf score and skip? [y/N]: ")
            if confirm.strip().lower() == "y":
                record.score = float("-inf")
                self.access.tell(record, extra_attrs={"reason": "zigzag"})
                print("Trial logged as bad (-inf) and skipped.")
                return
        print(f"🧪 Suggested Trial #{record.trial_number}")
        print(json.dumps(record.params, indent=2))
        log_study_run(self.client_lab, self.experiment, self.study_name, 1, "suggest")

    def report_result(self, trial_id: int, rt_csv_path: str):
        record = TrialRecord.load(self.study_name, trial_id)
        gradient = record.bo_gradient
        rt_list = load_rt_list_from_csv(rt_csv_path)
        score = compute_total_score(rt_list, gradient)

        record = TrialRecord.load(self.study_name, trial_id)
        record.score = score
        record.rt_list = rt_list
        self.access.tell(record)

        print(f"✓ Trial {trial_id} updated with score {score:.2f}")
        log_study_run(self.client_lab, self.experiment, self.study_name, 1, "report")

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
