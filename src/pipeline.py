"""End-to-end SRIS model pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .explainability import RiskExplainer
from .features import FeatureEngineer
from .interventions import InterventionEngine
from .prediction import FailurePredictionModel
from .shock import ShockDetectionLayer
from .trajectory import FailureTrajectoryEngine


@dataclass
class SRISPipeline:
    """Coordinate prediction, trajectory, shock, XAI, and interventions."""

    feature_engineer: FeatureEngineer = field(default_factory=FeatureEngineer)
    predictor: FailurePredictionModel = field(default_factory=FailurePredictionModel)
    trajectory_engine: FailureTrajectoryEngine = field(default_factory=FailureTrajectoryEngine)
    shock_layer: ShockDetectionLayer = field(default_factory=ShockDetectionLayer)
    explainer: RiskExplainer = field(default_factory=RiskExplainer)
    intervention_engine: InterventionEngine = field(default_factory=InterventionEngine)

    def fit(self, training_records: pd.DataFrame) -> dict:
        """Train the failure prediction model."""

        self.predictor.feature_engineer = self.feature_engineer
        self.explainer.feature_engineer = self.feature_engineer
        return self.predictor.fit(training_records)

    def score(self, records: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Run all SRIS model layers for the supplied records."""

        engineered = self.feature_engineer.transform(records)
        risk_scores = self.predictor.predict_failure_probability(records)
        scored = engineered.merge(
            risk_scores,
            on=[c for c in ["sme_id", "month"] if c in records.columns],
            how="left",
        )
        trajectory = self.trajectory_engine.analyze(
            scored[["sme_id", "month", "failure_probability"]]
        )
        shock_flags = self.shock_layer.detect(scored)
        explanations = self.explainer.explain(records)
        interventions = self.intervention_engine.recommend(explanations)
        return {
            "scored_records": scored,
            "trajectory": trajectory,
            "shock_flags": shock_flags,
            "explanations": explanations,
            "interventions": interventions,
        }
