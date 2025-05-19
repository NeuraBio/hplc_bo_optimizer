"""
Type definitions for HPLC Bayesian Optimization.

This module contains shared type definitions used across the project.
"""

from typing import TypedDict


class OptimizationParams(TypedDict):
    """Type definition for optimization parameters."""

    flow_rate: float
    pH: float
    percent_organic: float
    gradient_slope: float
    gradient_length: float
    column_temp: float
    diluent: str  # categorical (e.g., FB, FC, etc.)
