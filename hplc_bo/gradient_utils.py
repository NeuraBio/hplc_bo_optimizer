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
    """Placeholder function for backward compatibility. Always returns 0.

    Zigzag penalties have been removed based on analysis of real-world HPLC methods,
    which showed that non-monotonic gradients are often beneficial for specific separations.

    Args:
        gradient: List of (time, %B) tuples representing the gradient profile

    Returns:
        Always returns 0.0 (no penalty)
    """
    return 0.0


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
    areas: List[float] = None,  # Optional peak areas for additional scoring
    plate_counts: List[float] = None,  # Optional theoretical plate counts
    target_run_time: float = 10.0,  # Target ideal run time in minutes
    min_resolution: float = 1.5,  # Minimum acceptable resolution
) -> float:
    """
    Computes a chromatography score based on USP-like criteria.

    Args:
        rt_list: Sorted list of retention times (minutes).
        peak_widths: Corresponding peak widths (minutes, at the same height level, e.g., 50%).
        tailing_factors: Corresponding USP tailing factors.
        areas: Optional list of peak areas for additional scoring criteria.
        plate_counts: Optional list of theoretical plate counts.
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

    # Check if areas were provided
    if areas is not None and len(areas) != num_peaks:
        # If areas don't match peaks, ignore them
        areas = None

    # Check if plate counts were provided
    if plate_counts is not None and len(plate_counts) != num_peaks:
        # If plate counts don't match peaks, ignore them
        plate_counts = None

    # Initial quality assessment - check for missing data
    missing_data_score = 0

    # Count peaks with missing areas
    if areas:
        missing_areas = sum(1 for a in areas if a is None or a == 0)
        if missing_areas > 0:
            missing_data_score -= missing_areas * 5.0
    else:
        # Severely penalize if no areas at all
        missing_data_score -= 25.0

    # Count peaks with missing plate counts
    if plate_counts:
        missing_plates = sum(1 for p in plate_counts if p is None or p == 0)
        if missing_plates > 0:
            missing_data_score -= missing_plates * 3.0
    else:
        # Penalize if no plate counts at all
        missing_data_score -= 15.0

    # 1. Resolution between adjacent peaks (weighted heavily)
    resolutions = []
    for i in range(num_peaks - 1):
        rt1, rt2 = rt_list[i], rt_list[i + 1]
        w1, w2 = peak_widths[i], peak_widths[i + 1]

        # Check if peak widths are suspiciously small (likely defaults)
        if w1 < rt1 * 0.03:  # Width less than 3% of RT is suspiciously small
            w1 = rt1 * 0.12  # Use a more conservative 12% estimate (increased from 10%)
        if w2 < rt2 * 0.03:
            w2 = rt2 * 0.12

        if w1 + w2 == 0:  # Avoid division by zero if widths are zero
            res = 0
        else:
            res = 2 * (rt2 - rt1) / (w1 + w2)
        resolutions.append(res)

    # Penalize resolutions below the minimum target with stronger penalties
    resolution_score_component = 0
    if resolutions:
        # Calculate average resolution
        avg_resolution = sum(resolutions) / len(resolutions)
        resolution_score_component = avg_resolution

        # Add exponentially increasing penalty for sub-minimal resolutions
        for res_val in resolutions:
            if res_val < min_resolution:
                # Exponential penalty that gets much worse as resolution decreases
                penalty = (min_resolution - res_val) ** 2 * 5.0  # Increased from 3.0
                resolution_score_component -= penalty

        # Extra penalty for very poor resolution (< 1.0)
        poor_resolutions = [r for r in resolutions if r < 1.0]
        if poor_resolutions:
            resolution_score_component -= len(poor_resolutions) * 8.0  # Increased from 5.0

    # 2. Peak symmetry/tailing factor penalty (with stricter acceptable range)
    symmetry_penalty_component = 0
    default_tailing_count = sum(1 for tf in tailing_factors if tf == 1.0)  # Count default values

    # If more than half the tailing factors are exactly 1.0, they're likely defaults
    if default_tailing_count > num_peaks / 2:
        # Apply a penalty for having mostly default values
        symmetry_penalty_component -= 15.0  # Increased from 5.0

    for tf in tailing_factors:
        # Stricter acceptable range (0.9-1.5 instead of 0.8-1.8)
        if not (0.9 <= tf <= 1.5):
            # Stronger penalty (3.0 instead of 1.5)
            symmetry_penalty_component -= abs(tf - 1.0) * 3.0

    # 3. Run time penalty (less weight than before)
    actual_run_time = rt_list[-1] if rt_list else 0
    time_penalty_component = 0
    if actual_run_time > target_run_time:
        time_penalty_component = -(actual_run_time - target_run_time) / target_run_time

    # 4. Early elution penalty (new component with stronger penalties)
    early_elution_penalty = 0
    for rt in rt_list:
        if rt < 2.0:  # Increased threshold from 1.0 to 2.0 minutes
            # Exponential penalty that increases as RT approaches 0
            early_elution_penalty -= (2.0 - rt) ** 2 * 3.0

    # 5. Peak distribution penalty (new component)
    distribution_penalty = 0
    if len(rt_list) >= 3:
        # Calculate standard deviation of retention time differences
        rt_diffs = [rt_list[i + 1] - rt_list[i] for i in range(len(rt_list) - 1)]
        avg_diff = sum(rt_diffs) / len(rt_diffs)
        std_diff = (sum((d - avg_diff) ** 2 for d in rt_diffs) / len(rt_diffs)) ** 0.5

        # Penalize high standard deviation (uneven peak spacing)
        if std_diff > avg_diff * 0.5:  # If std_dev is more than 50% of average
            distribution_penalty -= std_diff / avg_diff * 5.0  # Increased from 3.0

    # 6. Area distribution penalty (if areas are available)
    area_penalty = 0
    if areas and any(a is not None and a > 0 for a in areas):
        # Filter out None or zero areas
        valid_areas = [a for a in areas if a is not None and a > 0]
        if valid_areas:
            # Calculate coefficient of variation of areas
            avg_area = sum(valid_areas) / len(valid_areas)
            if avg_area > 0:
                std_area = (sum((a - avg_area) ** 2 for a in valid_areas) / len(valid_areas)) ** 0.5
                cv_area = std_area / avg_area

                # Penalize high CV (very uneven peak areas)
                if cv_area > 0.8:  # Lowered threshold from 1.0 to 0.8
                    area_penalty -= (cv_area - 0.8) * 5.0  # Increased from 3.0

        # Additional penalty for having too few peaks with valid areas
        if len(valid_areas) < num_peaks / 2:
            area_penalty -= 10.0

    # 7. Plate count assessment (if available)
    plate_count_component = 0
    if plate_counts and any(p is not None and p > 0 for p in plate_counts):
        valid_plates = [p for p in plate_counts if p is not None and p > 0]
        if valid_plates:
            avg_plate = sum(valid_plates) / len(valid_plates)

            # Reward high plate counts (good column efficiency)
            if avg_plate > 5000:
                plate_count_component += min(5.0, avg_plate / 5000)  # Cap at 5.0
            elif avg_plate < 2000:
                # Penalize very low plate counts
                plate_count_component -= (2000 - avg_plate) / 500

    # Combine scores with adjusted weights
    w_res = 0.40  # Resolution (reduced)
    w_sym = 0.15  # Symmetry (reduced)
    w_time = 0.05  # Run time (reduced)
    w_early = 0.15  # Early elution (increased)
    w_dist = 0.10  # Peak distribution
    w_area = 0.10  # Area distribution (increased)
    w_plate = 0.05  # Plate count (new)

    final_score = (
        w_res * resolution_score_component
        + w_sym * symmetry_penalty_component
        + w_time * time_penalty_component
        + w_early * early_elution_penalty
        + w_dist * distribution_penalty
        + w_area * area_penalty
        + w_plate * plate_count_component
    )

    # Additional penalties for specific bad scenarios

    # 1. Too few peaks detected (< 3)
    if num_peaks < 3:
        final_score -= 20.0  # Increased from 10.0

    # 2. No real separation (all peaks co-eluting)
    if not resolutions or max(resolutions, default=0) < 0.5:  # if max res is very low
        final_score -= 75.0  # Increased from 50.0

    # 3. Missing data penalty
    # If we have mostly default values, apply an additional penalty
    default_width_count = sum(1 for i, w in enumerate(peak_widths) if w <= rt_list[i] * 0.05)
    if default_width_count > num_peaks / 2:
        final_score -= 20.0  # Increased from 10.0

    # 4. Apply the missing data score
    final_score += missing_data_score

    # 5. Ghost peaks penalty - if many peaks have no area
    if areas:
        ghost_peaks = sum(1 for a in areas if a is None or a == 0)
        if ghost_peaks > num_peaks / 3:  # If more than 1/3 of peaks have no area
            final_score -= ghost_peaks * 5.0

    # 6. Severe penalty for chromatograms with only early eluting peaks
    if all(rt < 3.0 for rt in rt_list):
        final_score -= 30.0

    return final_score
