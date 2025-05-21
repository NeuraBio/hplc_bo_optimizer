"""
Type definitions for HPLC Bayesian Optimization.

This module contains shared type definitions used across the project.
"""

from typing import List, Tuple, TypedDict


class OptimizationParams(TypedDict):
    gradient: List[Tuple[float, float]]  # (time, %B) anchor points
    flow_rate: float  # mL/min
    pH: float  # buffer pH
    column_temp: float  # °C
