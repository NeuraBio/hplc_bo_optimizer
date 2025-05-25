import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from hplc_bo.param_types import OptimizationParams


class TrialRecord:
    def __init__(
        self,
        trial_number: int,
        bo_gradient: List[Tuple[float, float]],
        flow_rate: float,
        pH: float,
        column_temp: float,
    ):
        self.trial_number = trial_number
        self.bo_gradient = bo_gradient
        self.flow_rate = flow_rate
        self.pH = pH
        self.column_temp = column_temp
        self.expanded_gradient = expand_gradient(bo_gradient, resolution=1, max_time=40.0)
        self.score = None
        self.rt_list = None

    def to_dict(self) -> dict:
        return {
            "trial_number": self.trial_number,
            "bo_gradient": json.dumps(self.bo_gradient),
            "gradient": json.dumps(self.expanded_gradient),
            "flow_rate": self.flow_rate,
            "pH": self.pH,
            "column_temp": self.column_temp,
            "score": self.score,
            "rt_list": json.dumps(self.rt_list) if self.rt_list else None,
        }

    @classmethod
    def from_dict(cls, row: dict) -> "TrialRecord":
        obj = cls(
            trial_number=int(row["trial_number"]),
            bo_gradient=json.loads(row["bo_gradient"]),
            flow_rate=float(row["flow_rate"]),
            pH=float(row["pH"]),
            column_temp=float(row["column_temp"]),
        )
        obj.score = float(row.get("score")) if pd.notna(row.get("score")) else None
        obj.rt_list = json.loads(row.get("rt_list")) if pd.notna(row.get("rt_list")) else None
        return obj

    def save(self, study_name: str):
        path = f"optuna_storage/suggested_trials/{study_name}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        row = self.to_dict()
        df = pd.DataFrame([row])
        if os.path.exists(path):
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, mode="w", header=True, index=False)

    def update_result(self, study_name: str):
        path = f"optuna_storage/suggested_trials/{study_name}.csv"
        df = pd.read_csv(path)
        df.loc[df.trial_number == self.trial_number, "score"] = self.score
        df.loc[df.trial_number == self.trial_number, "rt_list"] = json.dumps(self.rt_list)
        df.to_csv(path, index=False)

    @classmethod
    def load(cls, study_name: str, trial_number: int) -> "TrialRecord":
        path = f"optuna_storage/suggested_trials/{study_name}.csv"
        df = pd.read_csv(path)
        row = df[df.trial_number == trial_number].iloc[0]
        return cls.from_dict(row)

    @classmethod
    def from_params(cls, trial_number: int, params: OptimizationParams) -> "TrialRecord":
        return cls(
            trial_number=trial_number,
            bo_gradient=params["gradient"],
            flow_rate=params["flow_rate"],
            pH=params["pH"],
            column_temp=params["column_temp"],
        )

    @property
    def params(self) -> OptimizationParams:
        return OptimizationParams(
            gradient=self.bo_gradient,
            flow_rate=self.flow_rate,
            pH=self.pH,
            column_temp=self.column_temp,
        )


def interpolate_gradient(
    gradient: List[Tuple[float, float]], anchor_times: List[float]
) -> List[Tuple[float, float]]:
    gradient = sorted(gradient)
    times, percents = zip(*gradient, strict=False)
    interpolated = np.interp(anchor_times, times, percents)
    return list(zip(anchor_times, interpolated, strict=False))


def expand_gradient(
    bo_vector: List[Tuple[float, float]], resolution: int = 1, max_time: float = 40.0
) -> List[Tuple[float, float]]:
    bo_vector = sorted(bo_vector)
    times, percents = zip(*bo_vector, strict=False)
    full_times = np.arange(0, max_time + resolution, resolution)
    interpolated = np.interp(full_times, times, percents)
    return list(zip(full_times.tolist(), interpolated.tolist(), strict=False))


def penalize_gradient_zigzags(gradient: List[Tuple[float, float]]) -> float:
    percents = [b for _, b in gradient]
    penalty = 0.0
    for i in range(len(percents) - 1):
        if percents[i + 1] < percents[i]:
            delta = percents[i] - percents[i + 1]
            weight = 2.0 if i < 3 else 0.5
            penalty += delta * weight
    return penalty


def load_peak_data_from_csv(file_path: str) -> Tuple[List[float], List[float], List[float]]:
    """
    Loads peak data (retention times, peak widths, tailing factors) from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A tuple containing:
            - rt_list: Sorted list of retention times.
            - peak_widths_list: List of corresponding peak widths.
            - tailing_factors_list: List of corresponding tailing factors.
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If any of the required columns are missing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file {file_path}: {e}") from e

    # --- ASSUMED COLUMN NAMES ---
    # These may need to be adjusted based on the actual export from Empower
    rt_col = "RT"
    width_col = "Peak_Width"  # Or "Width", "Width_at_50%", etc.
    tailing_col = "Tailing_Factor"  # Or "USP_Tailing", "Asymmetry" etc.
    # --- END ASSUMED COLUMN NAMES ---

    required_columns = [rt_col, width_col, tailing_col]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' not found in CSV file {file_path}. "
                f"Available columns: {df.columns.tolist()}"
            )

    # Ensure data is sorted by retention time and corresponding values are aligned
    df_sorted = df.sort_values(by=rt_col).dropna(subset=required_columns)

    rt_list = df_sorted[rt_col].tolist()
    peak_widths_list = df_sorted[width_col].tolist()
    tailing_factors_list = df_sorted[tailing_col].tolist()

    if not (len(rt_list) == len(peak_widths_list) == len(tailing_factors_list)):
        # This should ideally not happen if dropna(subset=required_columns) works as expected
        raise ValueError(
            "Mismatch in lengths of extracted lists (RT, widths, tailing factors) "
            "after processing. Check CSV data integrity."
        )

    return rt_list, peak_widths_list, tailing_factors_list


def compute_score_usp(
    rt_list: List[float],
    peak_widths: List[float],
    tailing_factors: List[float],
    target_run_time: float = 10.0,  # Target ideal run time in minutes
    min_resolution: float = 1.5,  # Minimum acceptable resolution
) -> float:
    """
    Computes a chromatography score based on USP-like criteria.

    Args:
        rt_list: Sorted list of retention times (minutes).
        peak_widths: Corresponding peak widths (minutes, at the same height level, e.g., 50%).
        tailing_factors: Corresponding USP tailing factors.
        target_run_time: The ideal total run time for penalty calculation.
        min_resolution: The minimum desired resolution between peaks. Peaks below this
                        will be more heavily penalized.

    Returns:
        Combined score (higher = better separation).
    """
    num_peaks = len(rt_list)

    if num_peaks < 2:
        # Not enough peaks to calculate meaningful separation
        return -1e9  # Return a very large negative number

    if not (num_peaks == len(peak_widths) == len(tailing_factors)):
        raise ValueError(
            "Input lists (rt_list, peak_widths, tailing_factors) must have the same length."
        )

    # 1. Resolution between adjacent peaks (weighted e.g., 50%)
    resolutions = []
    for i in range(num_peaks - 1):
        rt1, rt2 = rt_list[i], rt_list[i + 1]
        w1, w2 = peak_widths[i], peak_widths[i + 1]
        if w1 + w2 == 0:  # Avoid division by zero if widths are zero
            res = 0
        else:
            res = 2 * (rt2 - rt1) / (w1 + w2)
        resolutions.append(res)

    # Penalize resolutions below the minimum target
    resolution_score_component = 0
    if resolutions:
        # Simple average, or could be more complex (e.g. geometric mean, penalizing worst)
        # For now, let's average and add a penalty for sub-minimal resolutions
        avg_resolution = sum(resolutions) / len(resolutions)
        resolution_score_component = avg_resolution
        # Add penalty for any resolution less than min_resolution
        for res_val in resolutions:
            if res_val < min_resolution:
                resolution_score_component -= (min_resolution - res_val) * 2  # Heavier penalty

    # 2. Peak symmetry/tailing factor penalty (weighted e.g., 30%)
    # Ideal Tailing Factor (TF) is 1.0. Deviations (e.g., 0.8 < TF < 1.5 or 1.8) are penalized.
    symmetry_penalty_component = 0
    for tf in tailing_factors:
        if not (0.8 <= tf <= 1.8):  # Example acceptable range, can be adjusted
            symmetry_penalty_component -= abs(tf - 1.0) * 1.5  # Penalize deviations from 1.0

    # 3. Run time penalty (weighted e.g., 20%)
    actual_run_time = rt_list[-1] if rt_list else 0
    # Penalize if run time is much longer than target, but don't overly penalize slightly longer
    # No penalty if shorter than target.
    time_penalty_component = 0
    if actual_run_time > target_run_time:
        time_penalty_component = -(actual_run_time - target_run_time) / target_run_time
        # This makes the penalty proportional to how much it exceeds the target.
        # e.g., 12 min run vs 10 min target = -0.2 penalty. 20 min run = -1.0 penalty.

    # Combine scores (adjust weights as needed)
    # Weights should sum to 1 if they are percentages, or just be scaling factors.
    w_res = 0.50
    w_sym = 0.30
    w_time = 0.20

    final_score = (
        w_res * resolution_score_component
        + w_sym * symmetry_penalty_component
        + w_time * time_penalty_component
    )

    # Ensure very bad scenarios (e.g. all peaks co-eluting, giving Rs=0) get a bad score
    if not resolutions or max(resolutions, default=0) < 0.1:  # if max res is very low
        final_score -= 100  # Large penalty for no real separation

    return final_score
