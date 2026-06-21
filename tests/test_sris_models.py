from sris_models import (
    FeatureEngineer,
    FailureTrajectoryEngine,
    InterventionEngine,
    ShockDetectionLayer,
    SRISPipeline,
)
from sris_models.synthetic_data import make_synthetic_sme_data


def test_feature_engineering_adds_required_derived_columns():
    data = make_synthetic_sme_data(n_smes=2, months=3)
    engineered = FeatureEngineer().transform(data)
    assert "revenue_growth_rate" in engineered
    assert "payment_stress_index" in engineered
    assert "cost_pressure_index" in engineered
    assert "shock_impact_score" in engineered


def test_pipeline_scores_all_layers():
    data = make_synthetic_sme_data(n_smes=20, months=8)
    pipeline = SRISPipeline()
    pipeline.fit(data[data["month"] <= 6])
    outputs = pipeline.score(data[data["month"] >= 5].drop(columns=["failed_within_6m"]))
    assert set(outputs) == {
        "scored_records",
        "trajectory",
        "shock_flags",
        "explanations",
        "interventions",
    }
    assert outputs["scored_records"]["failure_probability"].between(0, 1).all()


def test_trajectory_classifies_rapid_deterioration():
    history = make_synthetic_sme_data(n_smes=1, months=4)[["sme_id", "month"]]
    history["failure_probability"] = [0.10, 0.18, 0.31, 0.52]
    result = FailureTrajectoryEngine().analyze(history)
    assert result.loc[0, "trajectory_class"] == "Rapid deterioration"


def test_shock_detection_tags_jump():
    data = make_synthetic_sme_data(n_smes=1, months=3)[["sme_id", "month"]]
    data["failure_probability"] = [0.10, 0.12, 0.35]
    data["shock_impact_score"] = [0.10, 0.20, 0.80]
    result = ShockDetectionLayer().detect(data)
    assert result.iloc[-1]["shock_induced_failure_risk"]


def test_interventions_are_mapped():
    explanations = make_synthetic_sme_data(n_smes=1, months=1)[["sme_id", "month"]]
    explanations["risk_driver"] = "Payment delays"
    explanations["driver_contribution_pct"] = 35.0
    result = InterventionEngine().recommend(explanations)
    assert result.loc[0, "recommended_intervention"] == "Invoice financing"
