"""
HPLC Bayesian Optimization package.

This package provides tools for optimizing HPLC parameters
using Bayesian optimization techniques.
"""

from .config import CATEGORICAL_SPACE, CONTINUOUS_SPACE
from .optimizer import run_optimization, suggest_params
from .param_types import OptimizationParams
from .scoring import TARGET, mock_score

__all__ = [
    "OptimizationParams",
    "run_optimization",
    "suggest_params",
    "mock_score",
    "TARGET",
    "CONTINUOUS_SPACE",
    "CATEGORICAL_SPACE",
]
