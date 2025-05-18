"""
Type definitions for HPLC Bayesian Optimization.

This module contains shared type definitions used across the project.
"""

from typing import TypedDict


class OptimizationParams(TypedDict):
    """Type definition for optimization parameters."""

    b_start: float
    b_end: float
    gradient_time: float
    flow_rate: float
    column_temp: float
    additive: str
