"""Explainable AI layer for SRIS risk drivers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import FeatureEngineer


DRIVER_FEATURES = {
    "Payment delays": ["payment_delay_days", "ar_turnover_days", "payment_stress_index"],
    "Revenue decline": ["revenue_growth_rate"],
    "Supply disruption": [
        "extreme_weather_event",
        "supply_chain_disruption_index",
        "logistics_delay_factor",
        "demand_shock_indicator",
        "shock_impact_score",
    ],
    "Cost escalation": ["cost_growth_rate", "cost_pressure_index", "operating_margin_gap"],
    "Supplier dependency": ["supplier_dependency", "customer_concentration"],
    "Macro pressure": ["inflation_rate", "interest_rate", "macro_pressure_index"],
}


@dataclass
class RiskExplainer:
    """Produce local, action-oriented risk driver percentages."""

    feature_engineer: FeatureEngineer = field(default_factory=FeatureEngineer)

    def explain(self, records: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
        """Return percentage driver attribution per SME-month record."""

        data = self.feature_engineer.transform(records)
        rows = []
        for idx, row in data.iterrows():
            scores = self._driver_scores(row)
            total = sum(scores.values()) or 1.0
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
            for driver, score in ranked:
                output = {
                    "record_index": idx,
                    "risk_driver": driver,
                    "driver_contribution_pct": round((score / total) * 100, 2),
                }
                if "sme_id" in row:
                    output["sme_id"] = row["sme_id"]
                if "month" in row:
                    output["month"] = row["month"]
                rows.append(output)
        return pd.DataFrame(rows)

    def _driver_scores(self, row: pd.Series) -> dict[str, float]:
        revenue_decline = max(0.0, -float(row["revenue_growth_rate"]))
        return {
            "Payment delays": float(row["payment_stress_index"]),
            "Revenue decline": min(revenue_decline / 0.5, 1.0),
            "Supply disruption": float(row["shock_impact_score"]),
            "Cost escalation": float(row["cost_pressure_index"]),
            "Supplier dependency": float(row["buyer_supplier_concentration"]),
            "Macro pressure": float(row["macro_pressure_index"]),
        }

    @staticmethod
    def summarize_global_importance(feature_importance: pd.DataFrame) -> pd.DataFrame:
        """Map model feature importances into human-readable driver groups."""

        grouped = []
        for driver, features in DRIVER_FEATURES.items():
            score = feature_importance.loc[
                feature_importance["feature"].isin(features), "importance"
            ].sum()
            grouped.append({"risk_driver": driver, "importance": float(score)})
        result = pd.DataFrame(grouped)
        total = result["importance"].sum()
        result["importance_pct"] = np.where(total > 0, result["importance"] / total * 100, 0)
        return result.sort_values("importance", ascending=False).reset_index(drop=True)
