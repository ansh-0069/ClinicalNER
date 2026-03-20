"""
feature_engineer.py
───────────────────
FeatureEngineer: domain-specific feature extraction for the
Diabetes-130 readmission prediction dataset.

Design decisions:
  - Single responsibility: all feature logic lives here, not in the model
  - Domain-aware: features reflect clinical reasoning, not just raw columns
  - Reproducible: fit() learns encodings on train set, transform() applies them
  - OOP: clean separation of fit/transform mirrors sklearn API

Clinical feature rationale:
  - n_medications, n_procedures: proxy for patient complexity
  - n_diagnoses: comorbidity burden
  - age_numeric: risk increases with age
  - high_risk_meds: insulin/metformin changes signal glycaemic instability
  - prior_visits: utilisation pattern is strong readmission predictor
  - discharge_to_care: discharge to home vs facility affects readmission
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Age encoding map ──────────────────────────────────────────────────────────

AGE_MAP = {
    "[0-10)": 5,   "[10-20)": 15,  "[20-30)": 25,
    "[30-40)": 35, "[40-50)": 45,  "[50-60)": 55,
    "[60-70)": 65, "[70-80)": 75,  "[80-90)": 85,
    "[90-100)": 95,
}

# High-risk medication columns — changes signal active treatment adjustment
HIGH_RISK_MED_COLS = [
    "insulin", "metformin", "glipizide", "glyburide",
    "pioglitazone", "rosiglitazone",
]

# Columns to drop — IDs, leakage risks, too sparse
DROP_COLS = [
    "encounter_id", "patient_nbr", "weight", "payer_code",
    "medical_specialty", "examide", "citoglipton",
]


@dataclass
class FeatureEngineer:
    """
    Transforms raw Diabetes-130 DataFrame into a feature matrix
    ready for XGBoost.

    Usage
    -----
    fe = FeatureEngineer()
    X_train = fe.fit_transform(df_train)
    X_test  = fe.transform(df_test)
    feature_names = fe.feature_names_
    """

    is_fitted:     bool          = field(default=False, init=False)
    feature_names_: list[str]   = field(default_factory=list, init=False)
    cat_encodings_: dict        = field(default_factory=dict, init=False)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on df and return transformed feature matrix."""
        df = self._engineer(df.copy())
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        for col in cat_cols:
            mapping = {v: i for i, v in enumerate(df[col].unique())}
            self.cat_encodings_[col] = mapping
            df[col] = df[col].map(mapping).fillna(-1).astype(int)
        df = df.fillna(-1)
        self.feature_names_ = df.columns.tolist()
        self.is_fitted = True
        logger.info("FeatureEngineer fitted | %d features", len(self.feature_names_))
        return df.values.astype(float)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform df using encodings learned during fit."""
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform() first.")
        df = self._engineer(df.copy())
        for col, mapping in self.cat_encodings_.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
        df = df.reindex(columns=self.feature_names_, fill_value=-1)
        return df.fillna(-1).values.astype(float)

    # ── Private ───────────────────────────────────────────────────────────────

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all domain-specific feature engineering steps."""

        # 1. Drop irrelevant / leakage columns
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        # 2. Encode age bracket → numeric midpoint
        if "age" in df.columns:
            df["age_numeric"] = df["age"].map(AGE_MAP).fillna(50)
            df = df.drop(columns=["age"])

        # 3. High-risk medication change flag
        # Any change in these meds signals active glycaemic management
        med_cols_present = [c for c in HIGH_RISK_MED_COLS if c in df.columns]
        if med_cols_present:
            df["high_risk_med_change"] = (
                df[med_cols_present]
                .apply(lambda col: col.isin(["Up", "Down"]))
                .any(axis=1)
                .astype(int)
            )

        # 4. Prior utilisation features
        visit_cols = [c for c in ["number_outpatient", "number_emergency",
                                   "number_inpatient"] if c in df.columns]
        if visit_cols:
            df["total_prior_visits"] = df[visit_cols].sum(axis=1)

        # 5. Comorbidity count from diag columns
        diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"]
                     if c in df.columns]
        if diag_cols:
            df["n_diagnoses"] = df[diag_cols].apply(
                lambda row: row.notna().sum(), axis=1
            )

        # 6. Replace '?' with NaN for proper imputation
        df = df.replace({"?": np.nan, "None": np.nan, "none": np.nan})

        # 7. Drop diag columns (too high cardinality for this pass)
        df = df.drop(columns=[c for c in diag_cols if c in df.columns],
                     errors="ignore")

        return df
