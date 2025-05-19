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
    Simulate 4 peaks based on input params and calculate a score:
    - If any pair of peaks has resolution Rs < 1.5, return -inf
    - Else, return -rt_response (minimize last peak time)
    """
    import random

    # Generate fake RTs based on input with spacing
    p1 = 4.0 + random.uniform(-0.1, 0.1)
    p2 = p1 + random.uniform(1.6, 2.0)
    p3 = p2 + random.uniform(1.6, 2.0)
    p4 = p3 + random.uniform(1.6, 2.0)
    rt_list = [p1, p2, p3, p4]
    rt_response = p4  # last peak's RT

    # Resolution calculation
    MIN_RS = 1.5
    EST_PEAK_WIDTH = 0.2
    for i in range(len(rt_list) - 1):
        delta = rt_list[i + 1] - rt_list[i]
        rs = 1.18 * delta / (2 * EST_PEAK_WIDTH)
        if rs < MIN_RS:
            return float("-inf")

    return -rt_response
