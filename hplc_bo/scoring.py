import numpy as np

from .param_types import OptimizationParams

# Define target parameters for optimization
TARGET: OptimizationParams = {
    "b_start": 30.0,
    "b_end": 80.0,
    "gradient_time": 20.0,
    "flow_rate": 0.5,
    "column_temp": 40.0,
    "additive": "TFA",
}


def mock_score(params: OptimizationParams) -> float:
    """
    Calculate a mock score for a set of HPLC parameters.

    The score is inversely proportional to the absolute difference
    between the provided parameters and target parameters, with
    different weights applied to different parameters.

    Args:
        params: A dictionary of HPLC optimization parameters

    Returns:
        A float score value with some random noise added
    """
    score = (
        -abs(params["b_start"] - TARGET["b_start"])
        - abs(params["b_end"] - TARGET["b_end"])
        - abs(params["gradient_time"] - TARGET["gradient_time"])
        - abs(params["flow_rate"] - TARGET["flow_rate"]) * 10
        - abs(params["column_temp"] - TARGET["column_temp"]) / 2
    )
    if params["additive"] != TARGET["additive"]:
        score -= 5
    return score + np.random.normal(0, 0.5)
