"""Synthetic SME data for SRIS demos and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_sme_data(
    n_smes: int = 80,
    months: int = 12,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create plausible monthly SME data with a failure label."""

    rng = np.random.default_rng(random_state)
    rows = []
    for sme_num in range(1, n_smes + 1):
        sme_id = f"SME-{sme_num:04d}"
        revenue = rng.normal(1_500_000, 350_000)
        supplier_dependency = rng.beta(3, 3)
        customer_concentration = rng.beta(2.5, 3)
        base_margin = rng.normal(0.14, 0.06)
        stress_bias = rng.normal(0, 0.08)

        for month in range(1, months + 1):
            previous_revenue = max(revenue, 50_000)
            cyclone_season = month in {5, 6, 7, 8, 9, 10}
            extreme_weather_event = int(cyclone_season and rng.random() < 0.12)
            supply_disruption = min(
                1.0,
                rng.beta(1.5, 6)
                + 0.35 * extreme_weather_event
                + 0.15 * supplier_dependency,
            )
            logistics_delay = min(
                1.0,
                rng.beta(2, 7) + 0.30 * extreme_weather_event + 0.25 * supply_disruption,
            )
            demand_shock = int(rng.random() < (0.07 + 0.10 * extreme_weather_event))
            cost_growth = max(
                0.0,
                rng.normal(0.06, 0.04)
                + 0.09 * supply_disruption
                + 0.04 * extreme_weather_event,
            )
            payment_delay = max(
                0.0,
                rng.normal(22, 12)
                + 22 * customer_concentration
                + 16 * demand_shock
                + 8 * stress_bias,
            )
            ar_turnover = max(10.0, payment_delay + rng.normal(22, 10))
            inflation = max(0.01, rng.normal(0.055, 0.012))
            interest = max(0.04, rng.normal(0.085, 0.015))

            revenue_growth = rng.normal(0.015, 0.055) - 0.12 * demand_shock - 0.08 * supply_disruption
            revenue = max(50_000, previous_revenue * (1 + revenue_growth))
            profit_margin = base_margin - 0.45 * cost_growth - 0.06 * supply_disruption

            latent_risk = (
                1.5 * max(0, -revenue_growth)
                + 1.2 * min(payment_delay / 90, 1)
                + 1.1 * min(cost_growth / 0.25, 1)
                + 1.0 * supply_disruption
                + 0.8 * logistics_delay
                + 0.7 * customer_concentration
                + 0.6 * supplier_dependency
                + 0.5 * max(0, 0.10 - profit_margin)
                + stress_bias
                - 2.15
            )
            failure_probability = 1 / (1 + np.exp(-latent_risk))
            failed = int(rng.random() < failure_probability)

            rows.append(
                {
                    "sme_id": sme_id,
                    "month": month,
                    "monthly_revenue": round(revenue, 2),
                    "previous_month_revenue": round(previous_revenue, 2),
                    "profit_margin": round(profit_margin, 4),
                    "cost_growth_rate": round(cost_growth, 4),
                    "payment_delay_days": round(payment_delay, 1),
                    "ar_turnover_days": round(ar_turnover, 1),
                    "supplier_dependency": round(float(supplier_dependency), 4),
                    "customer_concentration": round(float(customer_concentration), 4),
                    "inflation_rate": round(inflation, 4),
                    "interest_rate": round(interest, 4),
                    "extreme_weather_event": extreme_weather_event,
                    "supply_chain_disruption_index": round(float(supply_disruption), 4),
                    "logistics_delay_factor": round(float(logistics_delay), 4),
                    "demand_shock_indicator": demand_shock,
                    "failed_within_6m": failed,
                }
            )

    return pd.DataFrame(rows)
