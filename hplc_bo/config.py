"""
Configuration settings for HPLC optimization.

This module defines the search space parameters for HPLC optimization,
including minimum and maximum values for numerical parameters and
possible values for categorical parameters.
"""

from typing import Dict, List, Tuple, Union

# Define the search space for optimization parameters
CONTINUOUS_SPACE: Dict[str, Union[List[str], Tuple[float, float]]] = {
    "flow_rate": (0.2, 1.0),  # From Excel: 0.8 seen
    "pH": (2.0, 9.0),  # Range to be confirmed
    "percent_organic": (5.0, 95.0),  # %Org
    "gradient_slope": (0.5, 20.0),  # GrSlope
    "gradient_length": (5.0, 30.0),  # GrLen
    "column_temp": (20.0, 60.0),  # 'T'
    "diluent": ["MPA", "FB"],  # Dilu – values from Excel
}

CATEGORICAL_SPACE: Dict[str, List[str]] = {
    "diluent": ["MPA", "FB"],
}
