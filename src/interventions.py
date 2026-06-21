"""Intervention engine for SRIS."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


DEFAULT_INTERVENTIONS = {
    "Payment delays": "Invoice financing",
    "Revenue decline": "Demand stimulation",
    "Supplier dependency": "Supplier diversification",
    "Supply disruption": "Alternate sourcing",
    "Cost escalation": "Cost restructuring",
    "Macro pressure": "Review debt and pricing exposure",
}


@dataclass
class InterventionEngine:
    """Map risk drivers to recommended actions."""

    intervention_map: dict[str, str] = field(default_factory=lambda: DEFAULT_INTERVENTIONS.copy())
    min_contribution_pct: float = 15.0

    def recommend(self, explanations: pd.DataFrame) -> pd.DataFrame:
        """Return recommended interventions for material risk drivers."""

        required = {"risk_driver", "driver_contribution_pct"}
        missing = required - set(explanations.columns)
        if missing:
            raise ValueError(f"Explanations missing columns: {', '.join(sorted(missing))}")

        data = explanations[
            explanations["driver_contribution_pct"] >= self.min_contribution_pct
        ].copy()
        data["recommended_intervention"] = data["risk_driver"].map(self.intervention_map)
        data["recommended_intervention"] = data["recommended_intervention"].fillna(
            "Manual resilience review"
        )
        columns = [
            c
            for c in [
                "sme_id",
                "month",
                "risk_driver",
                "driver_contribution_pct",
                "recommended_intervention",
            ]
            if c in data
        ]
        return data[columns].reset_index(drop=True)
