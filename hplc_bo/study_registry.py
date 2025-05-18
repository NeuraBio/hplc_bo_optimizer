import csv
import getpass
from datetime import datetime
from pathlib import Path

REGISTRY_PATH = Path("optuna_storage/study_registry.csv")
REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_study_run(client, experiment, study_name, trials, run_mode, status="completed"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user = getpass.getuser()
    is_new = not REGISTRY_PATH.exists()

    with open(REGISTRY_PATH, "a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "timestamp",
                "client",
                "experiment",
                "study_name",
                "trials",
                "status",
                "run_mode",
                "user",
            ],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp,
                "client": client,
                "experiment": experiment,
                "study_name": study_name,
                "trials": trials,
                "status": status,
                "run_mode": run_mode,
                "user": user,
            }
        )
