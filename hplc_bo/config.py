"""
Configuration settings for HPLC optimization.

This module defines the search space parameters for HPLC optimization,
including minimum and maximum values for numerical parameters and
possible values for categorical parameters.
"""

from typing import Dict, List, Tuple, Union

# Define the search space for optimization parameters
SEARCH_SPACE: Dict[str, Union[List[str], Tuple[float, float]]] = {
    "b_start": (20.0, 40.0),
    "b_end": (60.0, 90.0),
    "gradient_time": (10.0, 30.0),
    "flow_rate": (0.2, 1.0),
    "column_temp": (25.0, 60.0),
    "additive": ["TFA", "FormicAcid"],
}
