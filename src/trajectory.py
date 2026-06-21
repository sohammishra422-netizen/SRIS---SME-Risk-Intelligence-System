"""Failure trajectory analysis for SME risk histories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FailureTrajectoryEngine:
    """Classify how failure risk evolves over time."""

    rapid_slope_threshold: float = 0.06
    gradual_slope_threshold: float = 0.02
    acceleration_threshold: float = 0.03

    def analyze(self, risk_history: pd.DataFrame) -> pd.DataFrame:
        """Compute trend, acceleration, and trajectory class per SME."""

        required = {"sme_id", "month", "failure_probability"}
        missing = required - set(risk_history.columns)
        if missing:
            raise ValueError(f"Risk history missing columns: {', '.join(sorted(missing))}")

        rows = []
        for sme_id, group in risk_history.sort_values("month").groupby("sme_id"):
            probabilities = group["failure_probability"].astype(float).to_numpy()
            slope = self._slope(probabilities)
            acceleration = self._acceleration(probabilities)
            rows.append(
                {
                    "sme_id": sme_id,
                    "observations": int(len(group)),
                    "latest_failure_probability": float(probabilities[-1]),
                    "risk_trend_slope": slope,
                    "risk_acceleration": acceleration,
                    "trajectory_class": self._classify(slope, acceleration),
                }
            )
        return pd.DataFrame(rows)

    def _classify(self, slope: float, acceleration: float) -> str:
        if slope >= self.rapid_slope_threshold or acceleration >= self.acceleration_threshold:
            return "Rapid deterioration"
        if slope >= self.gradual_slope_threshold:
            return "Gradual decline"
        return "Stable"

    @staticmethod
    def _slope(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        return float(np.polyfit(x, values, 1)[0])

    @staticmethod
    def _acceleration(values: np.ndarray) -> float:
        if len(values) < 3:
            return 0.0
        first_differences = np.diff(values)
        return float(np.mean(np.diff(first_differences)))
