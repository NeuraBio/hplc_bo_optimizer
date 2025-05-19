import optuna

from .config import CATEGORICAL_SPACE, CONTINUOUS_SPACE
from .param_types import OptimizationParams
from .scoring import mock_score


def suggest_params(trial: optuna.Trial) -> OptimizationParams:
    return {
        "flow_rate": trial.suggest_float("flow_rate", *CONTINUOUS_SPACE["flow_rate"]),
        "pH": trial.suggest_float("pH", *CONTINUOUS_SPACE["pH"]),
        "percent_organic": trial.suggest_float(
            "percent_organic", *CONTINUOUS_SPACE["percent_organic"]
        ),
        "gradient_slope": trial.suggest_float(
            "gradient_slope", *CONTINUOUS_SPACE["gradient_slope"]
        ),
        "gradient_length": trial.suggest_float(
            "gradient_length", *CONTINUOUS_SPACE["gradient_length"]
        ),
        "column_temp": trial.suggest_float("column_temp", *CONTINUOUS_SPACE["column_temp"]),
        "diluent": trial.suggest_categorical("diluent", CATEGORICAL_SPACE["diluent"]),
    }


def run_optimization(n_trials: int = 50) -> optuna.study.Study:
    """
    Execute an optimization process using Optuna to maximize the objective function.

    The optimization finds the best parameters to achieve the highest score by
    iteratively evaluating different parameter sets over a number of trials.

    Args:
        n_trials: The number of optimization trials to run

    Returns:
        A study object containing the optimization results, including the
        best parameters and their corresponding best score
    """

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        score = mock_score(params)
        trial.set_user_attr("params", dict(params))  # Convert TypedDict to dict for serialization
        trial.set_user_attr("score", score)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study
