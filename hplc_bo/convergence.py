# hplc_bo/convergence.py
import matplotlib.pyplot as plt
import optuna
import pandas as pd


def plot_convergence(study_name: str, output_path: str = "convergence.png"):
    study = optuna.load_study(
        study_name=study_name, storage="sqlite:///optuna_storage/hplc_study.db"
    )

    # Only use completed trials (with values)
    trials = [t for t in study.trials if t.value is not None]

    if not trials:
        print("⚠️ No completed trials to plot")
        return

    # Prepare data
    df = pd.DataFrame(
        {
            "trial_number": [t.number for t in trials],
            "score": [t.value for t in trials],
            "params": [t.params for t in trials],
        }
    )

    # Plotting code remains the same...
    plt.figure(figsize=(10, 6))
    plt.plot(df["trial_number"], df["score"], "o-", label="Score")
    plt.xlabel("Trial Number")
    plt.ylabel("Separation Score")
    plt.title(f"Optimization Convergence: {study_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"[✓] Saved convergence plot to {output_path}")
