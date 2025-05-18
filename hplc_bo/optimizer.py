import optuna

from .config import SEARCH_SPACE
from .param_types import OptimizationParams
from .scoring import mock_score


def suggest_params(trial: optuna.Trial) -> OptimizationParams:
    return {
        "b_start": trial.suggest_float("b_start", *SEARCH_SPACE["b_start"]),
        "b_end": trial.suggest_float("b_end", *SEARCH_SPACE["b_end"]),
        "gradient_time": trial.suggest_float("gradient_time", *SEARCH_SPACE["gradient_time"]),
        "flow_rate": trial.suggest_float("flow_rate", *SEARCH_SPACE["flow_rate"]),
        "column_temp": trial.suggest_float("column_temp", *SEARCH_SPACE["column_temp"]),
        "additive": trial.suggest_categorical("additive", SEARCH_SPACE["additive"]),
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
