"""
clinical_risk_model.py
──────────────────────
ClinicalRiskModel: predicts 30-day hospital readmission risk using
XGBoost on the Diabetes-130 dataset.

Why this matters for the JD:
  "Develop ML models and algorithms for predictive data analytics"
  This is the SUPERVISED ML component — complements AnomalyDetector
  (unsupervised). Together they show full ML pipeline capability.

Target variable:
  readmitted → binary: '<30' = high risk (1), else = low risk (0)
  Clinical rationale: 30-day readmission is a standard CMS quality metric
  and directly relevant to clinical data management workflows.

Model choice — XGBoost:
  - Handles mixed types (numeric + categorical after encoding)
  - Built-in feature importance (interpretable to clinical teams)
  - Robust to missing values (common in EHR data)
  - Industry standard for tabular clinical data
  - scale_pos_weight handles class imbalance (~9:1 negative:positive)

Evaluation metrics:
  - ROC-AUC: primary metric (threshold-independent)
  - F1 (macro): accounts for class imbalance
  - Precision / Recall: clinical tradeoff transparency
  - Confusion matrix: for operational review
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s"
)


# ── Evaluation result dataclass ───────────────────────────────────────────────

@dataclass(frozen=True)
class ModelEvalResult:
    """
    Immutable evaluation result for one training run.
    Frozen so it can be safely stored in app.config and audit logs.
    """
    model_name:   str
    roc_auc:      float
    f1_macro:     float
    precision:    float
    recall:       float
    n_train:      int
    n_test:       int
    feature_count: int
    top_features: list

    def to_dict(self) -> dict:
        return {
            "model_name":    self.model_name,
            "roc_auc":       self.roc_auc,
            "f1_macro":      self.f1_macro,
            "precision":     self.precision,
            "recall":        self.recall,
            "n_train":       self.n_train,
            "n_test":        self.n_test,
            "feature_count": self.feature_count,
            "top_features":  self.top_features,
        }

    def summary(self) -> str:
        return (
            f"{self.model_name} | "
            f"AUC={self.roc_auc:.3f} | "
            f"F1={self.f1_macro:.3f} | "
            f"train={self.n_train} | test={self.n_test}"
        )


# ── ClinicalRiskModel class ───────────────────────────────────────────────────

class ClinicalRiskModel:
    """
    XGBoost classifier for 30-day hospital readmission prediction.

    Usage
    -----
    model = ClinicalRiskModel()
    result = model.train("data/raw/diabetic_data.csv")
    print(result.summary())

    # Predict on new records
    risk_scores = model.predict_proba(df_new)

    # Save / load
    model.save("data/models/clinical_risk_model.pkl")
    model = ClinicalRiskModel.load("data/models/clinical_risk_model.pkl")
    """

    MODEL_NAME = "XGBoost-ReadmissionRisk-v1"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators   = n_estimators
        self.max_depth      = max_depth
        self.learning_rate  = learning_rate
        self.random_state   = random_state
        self._model         = None
        self._feature_eng   = None
        self.is_fitted      = False
        self.eval_result_: Optional[ModelEvalResult] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def train(
        self,
        data_path: str,
        test_size: float = 0.2,
        save_path: Optional[str] = "data/models/clinical_risk_model.pkl",
    ) -> ModelEvalResult:
        """
        Full training pipeline: load → engineer → split → train → evaluate.

        Parameters
        ----------
        data_path : path to diabetic_data.csv
        test_size : fraction held out for evaluation
        save_path : if provided, saves model to this path

        Returns
        -------
        ModelEvalResult with AUC, F1, precision, recall, top features
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError(
                "xgboost not installed. Run: pip install xgboost"
            )

        from src.models.feature_engineer import FeatureEngineer

        logger.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path)
        logger.info("Loaded %d records, %d columns", *df.shape)

        # ── Target encoding ───────────────────────────────────────────────────
        # Binary: '<30' = high risk (1), everything else = low risk (0)
        # Clinical rationale: 30-day readmission is the CMS quality metric
        y = (df["readmitted"] == "<30").astype(int)
        X_raw = df.drop(columns=["readmitted"])

        # ── Feature engineering ───────────────────────────────────────────────
        self._feature_eng = FeatureEngineer()
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=test_size,
            random_state=self.random_state, stratify=y
        )

        logger.info("Engineering features...")
        X_train = self._feature_eng.fit_transform(X_train_raw)
        X_test  = self._feature_eng.transform(X_test_raw)
        n_features = X_train.shape[1]
        logger.info("Feature matrix: %d train, %d test, %d features",
                    len(X_train), len(X_test), n_features)

        # ── Class imbalance ───────────────────────────────────────────────────
        # scale_pos_weight = negative / positive count
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = round(neg / max(pos, 1), 2)
        logger.info("Class ratio neg:pos = %d:%d | scale_pos_weight=%.2f",
                    neg, pos, scale_pos_weight)

        # ── Train ─────────────────────────────────────────────────────────────
        logger.info("Training XGBoost | n_estimators=%d | max_depth=%d",
                    self.n_estimators, self.max_depth)
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        self.is_fitted = True

        # ── Evaluate ──────────────────────────────────────────────────────────
        self.eval_result_ = self._evaluate(X_test, y_test, n_features)
        logger.info("Training complete | %s", self.eval_result_.summary())

        # ── Save ──────────────────────────────────────────────────────────────
        if save_path:
            self.save(save_path)

        return self.eval_result_

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return readmission risk scores (probability of class 1).

        Parameters
        ----------
        df : raw DataFrame with same columns as training data
             (without 'readmitted' column)

        Returns
        -------
        np.ndarray of shape (n_samples,) with probabilities in [0, 1]
        """
        self._check_fitted()
        X = self._feature_eng.transform(df)
        return self._model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Return binary predictions (0 = low risk, 1 = high risk).

        Parameters
        ----------
        threshold : decision threshold (default 0.5)
                    lower = more sensitive, higher = more specific
        """
        return (self.predict_proba(df) >= threshold).astype(int)

    def feature_importance(self, top_n: int = 15) -> list[dict]:
        """
        Return top N features by XGBoost importance score.
        Used by the Flask /api/risk-model route.
        """
        self._check_fitted()
        importances = self._model.feature_importances_
        names       = self._feature_eng.feature_names_
        ranked = sorted(
            zip(names, importances),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        return [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in ranked
        ]

    def save(self, path: str) -> None:
        """Persist model + feature engineer to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model,
                         "feature_eng": self._feature_eng,
                         "eval_result": self.eval_result_}, f)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "ClinicalRiskModel":
        """Load a previously saved model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance._model       = data["model"]
        instance._feature_eng = data["feature_eng"]
        instance.eval_result_ = data["eval_result"]
        instance.is_fitted    = True
        logger.info("Model loaded from %s", path)
        return instance

    # ── Private ───────────────────────────────────────────────────────────────

    def _evaluate(
        self,
        X_test: np.ndarray,
        y_test: pd.Series,
        n_features: int,
    ) -> ModelEvalResult:
        """Compute all evaluation metrics on held-out test set."""
        y_pred      = self._model.predict(X_test)
        y_proba     = self._model.predict_proba(X_test)[:, 1]
        report      = classification_report(y_test, y_pred,
                                            output_dict=True, zero_division=0)
        try:
            roc_auc = round(roc_auc_score(y_test, y_proba), 4)
        except ValueError:
            # Only one class in test split (e.g. tiny test fixtures) — undefined
            roc_auc = 0.5
        f1_macro    = round(f1_score(y_test, y_pred, average="macro",
                                     zero_division=0), 4)
        precision   = round(report.get("1", {}).get("precision", 0.0), 4)
        recall      = round(report.get("1", {}).get("recall",    0.0), 4)
        top_feats   = [f["feature"] for f in self.feature_importance(top_n=5)]

        logger.info("Evaluation | AUC=%.3f | F1=%.3f | P=%.3f | R=%.3f",
                    roc_auc, f1_macro, precision, recall)

        return ModelEvalResult(
            model_name    = self.MODEL_NAME,
            roc_auc       = roc_auc,
            f1_macro      = f1_macro,
            precision     = precision,
            recall        = recall,
            n_train       = int(len(y_test) * 4),  # approx
            n_test        = int(len(y_test)),
            feature_count = n_features,
            top_features  = top_feats,
        )

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "Model not fitted. Call train() or load() first."
            )
