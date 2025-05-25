import json

import matplotlib.pyplot as plt
import optuna
import pandas as pd

from hplc_bo.convergence import plot_convergence
from hplc_bo.gradient_utils import (
    TrialRecord,
    compute_score_usp,
    load_peak_data_from_csv,
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
            print(f"⚠️ Warning: This gradient has zigzags (penalty score: {penalty:.1f}).")
            print(f"🧪 Suggested Trial #{record.trial_number}")
            print(json.dumps(record.params, indent=2))

            choice = input(
                "Do you want to: [1] Continue with this gradient, [2] Mark as invalid and skip? (1/2): "
            )
            if choice.strip() == "2":
                self.mark_invalid(record.trial_number, reason="gradient_zigzag")
                print("Trial marked as invalid. Suggesting a new trial...")
                # Recursively call suggest to get a new trial
                return self.suggest()
        else:
            print(f"🧪 Suggested Trial #{record.trial_number}")
            print(json.dumps(record.params, indent=2))

        log_study_run(self.client_lab, self.experiment, self.study_name, 1, "suggest")
        return record

    def report_result(self, trial_id: int, rt_csv_path: str):
        try:
            # Load all necessary data from the CSV
            rt_list, peak_widths, tailing_factors = load_peak_data_from_csv(rt_csv_path)

            # Calculate score using the new USP-based function
            # We can use default target_run_time and min_resolution from compute_score_usp
            # or pass them as arguments if they need to be configurable per study/run.
            score = compute_score_usp(rt_list, peak_widths, tailing_factors)

        except FileNotFoundError:
            print(
                f"❌ Error: Chromatogram CSV file not found at {rt_csv_path} for trial {trial_id}."
            )
            # Assign a very bad score or mark as invalid
            score = -1e10  # Or consider calling self.mark_invalid(trial_id, reason="csv_not_found")
            rt_list = []  # Ensure rt_list is defined for the record
        except (KeyError, ValueError) as e:
            print(
                f"❌ Error: Failed to parse CSV or invalid data for trial {trial_id} from {rt_csv_path}. Details: {e}"
            )
            # Assign a very bad score
            score = (
                -1e10
            )  # Or consider calling self.mark_invalid(trial_id, reason="csv_parsing_error")
            rt_list = []  # Ensure rt_list is defined for the record
        except Exception as e:  # Catch any other unexpected errors during scoring
            print(
                f"❌ Error: An unexpected error occurred during scoring for trial {trial_id}. Details: {e}"
            )
            score = -1e10
            rt_list = []

        record = TrialRecord.load(self.study_name, trial_id)
        record.score = score
        record.rt_list = rt_list  # rt_list is already part of TrialRecord
        # If we decided to store peak_widths and tailing_factors in TrialRecord,
        # we would set them here:
        # record.peak_widths = peak_widths
        # record.tailing_factors = tailing_factors

        self.access.tell(record)

        if score > -1e9:  # Check if it wasn't an error score
            print(f"✓ Trial {trial_id} updated with score {score:.4f}")
        else:
            print(f"✓ Trial {trial_id} processed with an error score {score:.4f}")

        log_study_run(self.client_lab, self.experiment, self.study_name, 1, "report")

    def mark_invalid(self, trial_id: int, reason: str = "user_rejected"):
        record = TrialRecord.load(self.study_name, trial_id)
        record.score = float("-inf")
        self.access.tell(record, extra_attrs={"reason": reason})
        print(f"✓ Trial {trial_id} marked as invalid and scored as -inf")
        log_study_run(self.client_lab, self.experiment, self.study_name, 1, "invalidate")

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

        plot_convergence(self.study_name, plot_path)
