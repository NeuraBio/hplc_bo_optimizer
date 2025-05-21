from typing import Dict, List, Tuple

# Fixed gradient anchor times (in minutes) for 5-point gradient
GRADIENT_ANCHOR_TIMES: List[float] = [0.0, 10.0, 15.0, 25.0, 35.0]

# Scalar input parameter bounds
CONTINUOUS_SPACE: Dict[str, Tuple[float, float]] = {
    "flow_rate": (0.2, 1.5),
    "pH": (2.0, 10),
    "column_temp": (25.0, 60.0),
    # Percent B values at each fixed anchor time
    "b0": (0.0, 100.0),
    "b1": (0.0, 100.0),
    "b2": (0.0, 100.0),
    "b3": (0.0, 100.0),
    "b4": (0.0, 100.0),
}
