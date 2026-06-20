"""Climate-linked shock detection layer."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ShockDetectionLayer:
    """Detect abrupt risk jumps associated with external disruptions."""

    risk_jump_threshold: float = 0.15
    shock_score_threshold: float = 0.45

    def detect(self, scored_records: pd.DataFrame) -> pd.DataFrame:
        """Tag rows where risk spikes coincide with shock indicators."""

        required = {"sme_id", "month", "failure_probability", "shock_impact_score"}
        missing = required - set(scored_records.columns)
        if missing:
            raise ValueError(f"Scored records missing columns: {', '.join(sorted(missing))}")

        data = scored_records.sort_values(["sme_id", "month"]).copy()
        data["previous_failure_probability"] = data.groupby("sme_id")[
            "failure_probability"
        ].shift(1)
        data["risk_jump"] = (
            data["failure_probability"] - data["previous_failure_probability"]
        ).fillna(0)
        data["shock_induced_failure_risk"] = (
            (data["risk_jump"] >= self.risk_jump_threshold)
            & (data["shock_impact_score"] >= self.shock_score_threshold)
        )
        data["shock_tag"] = data["shock_induced_failure_risk"].map(
            {True: "Shock-induced failure risk", False: "No shock-induced spike"}
        )
        return data
