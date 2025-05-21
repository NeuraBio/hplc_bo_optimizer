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


def compute_rt_score(rt_list: List[float]) -> float:
    if len(rt_list) < 2:
        return float("-inf")
    deltas = [b - a for a, b in zip(rt_list, rt_list[1:], strict=False)]
    return sum(deltas)


def compute_total_score(rt_list: List[float], gradient: List[Tuple[float, float]]) -> float:
    rt_score = compute_rt_score(rt_list)
    zigzag_penalty = penalize_gradient_zigzags(gradient)
    if rt_score == float("-inf"):
        return rt_score
    return rt_score - zigzag_penalty


def load_rt_list_from_csv(path: str) -> List[float]:
    df = pd.read_csv(path)
    return sorted(df["RT"].dropna().tolist())
