import optuna

from .config import CONTINUOUS_SPACE, GRADIENT_ANCHOR_TIMES
from .param_types import OptimizationParams


def suggest_params(trial: optuna.Trial) -> OptimizationParams:
    # Fixed anchor times (defined in config)
    gradient = []
    for i, t in enumerate(GRADIENT_ANCHOR_TIMES):
        b = trial.suggest_float(f"b{i}", *CONTINUOUS_SPACE[f"b{i}"])
        gradient.append((t, b))

    return {
        "gradient": gradient,
        "flow_rate": trial.suggest_float("flow_rate", *CONTINUOUS_SPACE["flow_rate"]),
        "pH": trial.suggest_float("pH", *CONTINUOUS_SPACE["pH"]),
        "column_temp": trial.suggest_float("column_temp", *CONTINUOUS_SPACE["column_temp"]),
    }
