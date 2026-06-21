"""Feature engineering for SRIS model inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "monthly_revenue",
    "previous_month_revenue",
    "profit_margin",
    "cost_growth_rate",
    "payment_delay_days",
    "ar_turnover_days",
    "supplier_dependency",
    "customer_concentration",
    "inflation_rate",
    "interest_rate",
    "extreme_weather_event",
    "supply_chain_disruption_index",
    "logistics_delay_factor",
    "demand_shock_indicator",
]

DERIVED_COLUMNS = [
    "revenue_growth_rate",
    "payment_stress_index",
    "cost_pressure_index",
    "shock_impact_score",
    "buyer_supplier_concentration",
    "macro_pressure_index",
    "operating_margin_gap",
]

SRIS_FEATURES = RAW_COLUMNS + DERIVED_COLUMNS


@dataclass(frozen=True)
class FeatureEngineer:
    """Create model-ready features from monthly SME observations."""

    expected_columns: Iterable[str] = tuple(RAW_COLUMNS)

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of records with SRIS derived features added."""

        data = records.copy()
        self._validate(data)

        current_revenue = data["monthly_revenue"].clip(lower=0)
        previous_revenue = data["previous_month_revenue"].replace(0, np.nan)
        data["revenue_growth_rate"] = (
            (current_revenue - previous_revenue) / previous_revenue
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        delay_component = (data["payment_delay_days"].clip(lower=0) / 90).clip(0, 1)
        turnover_component = (data["ar_turnover_days"].clip(lower=0) / 120).clip(0, 1)
        data["payment_stress_index"] = (0.65 * delay_component) + (
            0.35 * turnover_component
        )

        cost_component = data["cost_growth_rate"].clip(lower=0, upper=1)
        margin_component = (1 - data["profit_margin"].clip(lower=-0.5, upper=0.5) / 0.5)
        margin_component = margin_component.clip(0, 1)
        data["cost_pressure_index"] = (0.6 * cost_component) + (
            0.4 * margin_component
        )

        data["shock_impact_score"] = (
            0.30 * data["extreme_weather_event"].clip(0, 1)
            + 0.30 * data["supply_chain_disruption_index"].clip(0, 1)
            + 0.25 * data["logistics_delay_factor"].clip(0, 1)
            + 0.15 * data["demand_shock_indicator"].clip(0, 1)
        ).clip(0, 1)

        data["buyer_supplier_concentration"] = (
            0.55 * data["customer_concentration"].clip(0, 1)
            + 0.45 * data["supplier_dependency"].clip(0, 1)
        )
        data["macro_pressure_index"] = (
            0.55 * (data["inflation_rate"].clip(lower=0) / 0.15).clip(0, 1)
            + 0.45 * (data["interest_rate"].clip(lower=0) / 0.20).clip(0, 1)
        )
        data["operating_margin_gap"] = (0.15 - data["profit_margin"]).clip(lower=0)

        return data

    def feature_matrix(self, records: pd.DataFrame) -> pd.DataFrame:
        """Return only the numeric feature columns used by the predictor."""

        engineered = self.transform(records)
        return engineered[SRIS_FEATURES].astype(float)

    def _validate(self, records: pd.DataFrame) -> None:
        missing = [column for column in self.expected_columns if column not in records]
        if missing:
            raise ValueError(f"Missing required SRIS columns: {', '.join(missing)}")
