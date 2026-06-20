"""Failure probability model for SRIS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .features import FeatureEngineer, SRIS_FEATURES

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight runtimes.
    joblib = None

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight runtimes.
    GradientBoostingClassifier = None
    RandomForestClassifier = None
    SimpleImputer = None
    Pipeline = None
    StandardScaler = None
    roc_auc_score = None


Algorithm = Literal["random_forest", "gradient_boosting"]


@dataclass
class FailurePredictionModel:
    """Train and score SME failure probability on a 0-100 scale."""

    algorithm: Algorithm = "gradient_boosting"
    random_state: int = 42
    feature_engineer: FeatureEngineer = field(default_factory=FeatureEngineer)
    model: object | None = None

    def fit(self, records: pd.DataFrame, target_column: str = "failed_within_6m") -> dict:
        """Train the model and return simple training diagnostics."""

        if target_column not in records:
            raise ValueError(f"Training records must include `{target_column}`.")

        X = self.feature_engineer.feature_matrix(records)
        y = records[target_column].astype(int)
        self.model = self._build_model()
        self.model.fit(X, y)
        probabilities = self.model.predict_proba(X)[:, 1]
        diagnostics = {
            "rows": int(len(records)),
            "positive_rate": float(y.mean()),
            "training_auc": self._safe_auc(y, probabilities),
        }
        return diagnostics

    def predict_failure_probability(self, records: pd.DataFrame) -> pd.DataFrame:
        """Return SME failure probability as fractions and percentages."""

        self._require_model()
        X = self.feature_engineer.feature_matrix(records)
        probability = self.model.predict_proba(X)[:, 1]
        result = records[[c for c in ["sme_id", "month"] if c in records]].copy()
        result["failure_probability"] = probability
        result["failure_probability_pct"] = np.round(probability * 100, 2)
        return result

    def feature_importance(self) -> pd.DataFrame:
        """Return global model feature importances when the estimator supports them."""

        self._require_model()
        estimator = (
            self.model.named_steps["estimator"]
            if hasattr(self.model, "named_steps")
            else self.model
        )
        if not hasattr(estimator, "feature_importances_"):
            raise RuntimeError("The selected estimator does not expose feature importances.")
        return (
            pd.DataFrame(
                {
                    "feature": SRIS_FEATURES,
                    "importance": estimator.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> None:
        """Persist the trained model to disk."""

        self._require_model()
        if joblib is not None:
            joblib.dump(self, path)
            return
        import pickle

        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str | Path) -> "FailurePredictionModel":
        """Load a saved model from disk."""

        if joblib is not None:
            return joblib.load(path)
        import pickle

        with open(path, "rb") as handle:
            return pickle.load(handle)

    def _build_model(self):
        if Pipeline is None:
            return NumpyFailureClassifier(random_state=self.random_state)
        estimator = self._build_estimator()
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("estimator", estimator),
            ]
        )

    def _build_estimator(self):
        if self.algorithm == "random_forest":
            return RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=self.random_state,
            )
        if self.algorithm == "gradient_boosting":
            return GradientBoostingClassifier(random_state=self.random_state)
        raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() or load() first.")

    @staticmethod
    def _safe_auc(y_true: pd.Series, probabilities: np.ndarray) -> float | None:
        if y_true.nunique() < 2:
            return None
        if roc_auc_score is not None:
            return float(roc_auc_score(y_true, probabilities))
        return float(_rank_auc(y_true.to_numpy(), probabilities))


class NumpyFailureClassifier:
    """Small logistic classifier used when scikit-learn is unavailable."""

    def __init__(
        self,
        random_state: int = 42,
        learning_rate: float = 0.08,
        iterations: int = 900,
        l2: float = 0.02,
    ) -> None:
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2 = l2
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NumpyFailureClassifier":
        matrix = X.astype(float).to_numpy()
        self.medians_ = np.nanmedian(matrix, axis=0)
        matrix = np.where(np.isnan(matrix), self.medians_, matrix)
        self.mean_ = matrix.mean(axis=0)
        self.std_ = matrix.std(axis=0)
        self.std_ = np.where(self.std_ == 0, 1, self.std_)
        scaled = (matrix - self.mean_) / self.std_

        target = y.astype(float).to_numpy()
        rng = np.random.default_rng(self.random_state)
        self.coef_ = rng.normal(0, 0.01, scaled.shape[1])
        self.intercept_ = 0.0

        for _ in range(self.iterations):
            logits = scaled @ self.coef_ + self.intercept_
            predictions = _sigmoid(logits)
            error = predictions - target
            gradient = (scaled.T @ error) / len(target) + self.l2 * self.coef_
            intercept_gradient = float(error.mean())
            self.coef_ -= self.learning_rate * gradient
            self.intercept_ -= self.learning_rate * intercept_gradient

        importance = np.abs(self.coef_)
        total = importance.sum()
        self.feature_importances_ = importance / total if total else importance
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        matrix = X.astype(float).to_numpy()
        matrix = np.where(np.isnan(matrix), self.medians_, matrix)
        scaled = (matrix - self.mean_) / self.std_
        probability = _sigmoid(scaled @ self.coef_ + self.intercept_)
        return np.column_stack([1 - probability, probability])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -35, 35)))


def _rank_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(probabilities)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(probabilities) + 1)
    positive = y_true == 1
    n_positive = positive.sum()
    n_negative = len(y_true) - n_positive
    if n_positive == 0 or n_negative == 0:
        return 0.5
    rank_sum = ranks[positive].sum()
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
