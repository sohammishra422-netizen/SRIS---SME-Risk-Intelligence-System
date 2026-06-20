"""Run a complete SRIS model demo with synthetic data."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sris_models import SRISPipeline
from sris_models.synthetic_data import make_synthetic_sme_data


def main() -> None:
    data = make_synthetic_sme_data(n_smes=120, months=12, random_state=7)
    train = data[data["month"] <= 9]
    score = data[data["month"] >= 7].drop(columns=["failed_within_6m"])

    pipeline = SRISPipeline()
    diagnostics = pipeline.fit(train)
    outputs = pipeline.score(score)

    print("Training diagnostics")
    print(diagnostics)
    print("\nLatest scored records")
    latest = outputs["scored_records"].sort_values(["sme_id", "month"]).groupby("sme_id").tail(1)
    print(
        latest[
            [
                "sme_id",
                "month",
                "failure_probability_pct",
                "shock_impact_score",
                "payment_stress_index",
                "cost_pressure_index",
            ]
        ]
        .sort_values("failure_probability_pct", ascending=False)
        .head(10)
        .to_string(index=False)
    )
    print("\nTrajectory summary")
    print(outputs["trajectory"]["trajectory_class"].value_counts().to_string())
    print("\nShock-induced risk examples")
    print(
        outputs["shock_flags"]
        .loc[outputs["shock_flags"]["shock_induced_failure_risk"]]
        [["sme_id", "month", "failure_probability_pct", "risk_jump", "shock_tag"]]
        .head(10)
        .to_string(index=False)
    )
    print("\nRecommended interventions")
    print(outputs["interventions"].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
