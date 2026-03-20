"""
test_clinical_risk_model.py
───────────────────────────
Tests for ClinicalRiskModel and FeatureEngineer.
Drop into: tests/test_clinical_risk_model.py
"""

import pytest
import numpy as np
import pandas as pd
from src.models.feature_engineer import FeatureEngineer
from src.models.clinical_risk_model import ClinicalRiskModel, ModelEvalResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal DataFrame matching Diabetes-130 schema."""
    return pd.DataFrame({
        "encounter_id":            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "patient_nbr":             [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "race":                    ["Caucasian"] * 10,
        "gender":                  ["Male", "Female"] * 5,
        "age":                     ["[50-60)"] * 5 + ["[70-80)"] * 5,
        "weight":                  ["?"] * 10,
        "admission_type_id":       [1] * 10,
        "discharge_disposition_id":[1] * 10,
        "admission_source_id":     [7] * 10,
        "time_in_hospital":        [3, 5, 2, 7, 1, 4, 6, 3, 5, 2],
        "payer_code":              ["MC"] * 10,
        "medical_specialty":       ["Cardiology"] * 10,
        "num_lab_procedures":      [40, 50, 30, 60, 20, 45, 55, 35, 48, 22],
        "num_procedures":          [1, 2, 0, 3, 1, 2, 1, 0, 2, 1],
        "num_medications":         [10, 15, 8, 20, 5, 12, 18, 7, 14, 9],
        "number_outpatient":       [0, 1, 0, 2, 0, 1, 0, 0, 1, 0],
        "number_emergency":        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "number_inpatient":        [1, 2, 0, 3, 0, 1, 2, 0, 1, 0],
        "diag_1":                  ["250.01"] * 10,
        "diag_2":                  ["401.9"] * 10,
        "diag_3":                  ["272.4"] * 10,
        "number_diagnoses":        [7, 9, 5, 9, 3, 8, 9, 4, 7, 5],
        "max_glu_serum":           ["None"] * 10,
        "A1Cresult":               ["None"] * 10,
        "metformin":               ["No", "Steady", "Up", "No", "No",
                                    "Steady", "Down", "No", "Up", "No"],
        "insulin":                 ["Steady"] * 5 + ["No"] * 5,
        "glipizide":               ["No"] * 10,
        "glyburide":               ["No"] * 10,
        "pioglitazone":            ["No"] * 10,
        "rosiglitazone":           ["No"] * 10,
        "diabetesMed":             ["Yes"] * 8 + ["No"] * 2,
        "change":                  ["Ch", "No"] * 5,
        "examide":                 ["No"] * 10,
        "citoglipton":             ["No"] * 10,
    })

@pytest.fixture
def sample_df_with_target(sample_df):
    sample_df["readmitted"] = ["NO", "<30", "NO", ">30", "NO",
                                "<30", "NO", "NO", ">30", "NO"]
    return sample_df


# ── FeatureEngineer tests ─────────────────────────────────────────────────────

class TestFeatureEngineer:

    def test_fit_transform_returns_numpy_array(self, sample_df):
        fe = FeatureEngineer()
        X = fe.fit_transform(sample_df)
        assert isinstance(X, np.ndarray)

    def test_fit_transform_correct_shape(self, sample_df):
        fe = FeatureEngineer()
        X = fe.fit_transform(sample_df)
        assert X.shape[0] == len(sample_df)

    def test_feature_names_populated_after_fit(self, sample_df):
        fe = FeatureEngineer()
        fe.fit_transform(sample_df)
        assert len(fe.feature_names_) > 0

    def test_is_fitted_flag(self, sample_df):
        fe = FeatureEngineer()
        assert fe.is_fitted is False
        fe.fit_transform(sample_df)
        assert fe.is_fitted is True

    def test_transform_requires_fit(self, sample_df):
        fe = FeatureEngineer()
        with pytest.raises(RuntimeError):
            fe.transform(sample_df)

    def test_transform_same_shape_as_fit(self, sample_df):
        fe = FeatureEngineer()
        X_train = fe.fit_transform(sample_df)
        X_test  = fe.transform(sample_df)
        assert X_train.shape[1] == X_test.shape[1]

    def test_age_numeric_feature_created(self, sample_df):
        fe = FeatureEngineer()
        fe.fit_transform(sample_df)
        assert "age_numeric" in fe.feature_names_
        assert "age" not in fe.feature_names_

    def test_high_risk_med_change_feature_created(self, sample_df):
        fe = FeatureEngineer()
        fe.fit_transform(sample_df)
        assert "high_risk_med_change" in fe.feature_names_

    def test_total_prior_visits_feature_created(self, sample_df):
        fe = FeatureEngineer()
        fe.fit_transform(sample_df)
        assert "total_prior_visits" in fe.feature_names_

    def test_drop_cols_not_in_features(self, sample_df):
        fe = FeatureEngineer()
        fe.fit_transform(sample_df)
        for col in ["encounter_id", "patient_nbr", "weight"]:
            assert col not in fe.feature_names_

    def test_no_nan_in_output(self, sample_df):
        fe = FeatureEngineer()
        X = fe.fit_transform(sample_df)
        assert not np.isnan(X).any()

    def test_output_is_float(self, sample_df):
        fe = FeatureEngineer()
        X = fe.fit_transform(sample_df)
        assert X.dtype == float


# ── ModelEvalResult tests ─────────────────────────────────────────────────────

class TestModelEvalResult:

    def test_to_dict_returns_dict(self):
        result = ModelEvalResult(
            model_name="test", roc_auc=0.7, f1_macro=0.5,
            precision=0.4, recall=0.6, n_train=100, n_test=20,
            feature_count=10, top_features=["a", "b"]
        )
        d = result.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_all_keys(self):
        result = ModelEvalResult(
            model_name="test", roc_auc=0.7, f1_macro=0.5,
            precision=0.4, recall=0.6, n_train=100, n_test=20,
            feature_count=10, top_features=["a"]
        )
        for key in ["model_name", "roc_auc", "f1_macro",
                    "precision", "recall", "n_train", "n_test",
                    "feature_count", "top_features"]:
            assert key in result.to_dict()

    def test_summary_returns_string(self):
        result = ModelEvalResult(
            model_name="XGBoost", roc_auc=0.675, f1_macro=0.532,
            precision=0.184, recall=0.575, n_train=81416, n_test=20354,
            feature_count=42, top_features=["number_inpatient"]
        )
        assert isinstance(result.summary(), str)
        assert "0.675" in result.summary()

    def test_frozen_raises_on_mutation(self):
        result = ModelEvalResult(
            model_name="test", roc_auc=0.7, f1_macro=0.5,
            precision=0.4, recall=0.6, n_train=100, n_test=20,
            feature_count=10, top_features=[]
        )
        with pytest.raises(Exception):
            result.roc_auc = 0.9


# ── ClinicalRiskModel tests ───────────────────────────────────────────────────

class TestClinicalRiskModel:

    def test_init_not_fitted(self):
        model = ClinicalRiskModel()
        assert model.is_fitted is False

    def test_predict_requires_fit(self, sample_df):
        model = ClinicalRiskModel()
        with pytest.raises(RuntimeError):
            model.predict(sample_df)

    def test_predict_proba_requires_fit(self, sample_df):
        model = ClinicalRiskModel()
        with pytest.raises(RuntimeError):
            model.predict_proba(sample_df)

    def test_feature_importance_requires_fit(self):
        model = ClinicalRiskModel()
        with pytest.raises(RuntimeError):
            model.feature_importance()

    def test_train_returns_eval_result(self, sample_df_with_target, tmp_path):
        model = ClinicalRiskModel(n_estimators=10)
        data_path = str(tmp_path / "test_data.csv")
        sample_df_with_target.to_csv(data_path, index=False)
        result = model.train(data_path, save_path=None)
        assert isinstance(result, ModelEvalResult)

    def test_train_sets_is_fitted(self, sample_df_with_target, tmp_path):
        model = ClinicalRiskModel(n_estimators=10)
        data_path = str(tmp_path / "test_data.csv")
        sample_df_with_target.to_csv(data_path, index=False)
        model.train(data_path, save_path=None)
        assert model.is_fitted is True

    def test_predict_proba_shape(self, sample_df_with_target, tmp_path):
        model = ClinicalRiskModel(n_estimators=10)
        data_path = str(tmp_path / "test_data.csv")
        sample_df_with_target.to_csv(data_path, index=False)
        model.train(data_path, save_path=None)
        X = sample_df_with_target.drop(columns=["readmitted"])
        proba = model.predict_proba(X)
        assert proba.shape[0] == len(X)

    def test_predict_proba_in_range(self, sample_df_with_target, tmp_path):
        model = ClinicalRiskModel(n_estimators=10)
        data_path = str(tmp_path / "test_data.csv")
        sample_df_with_target.to_csv(data_path, index=False)
        model.train(data_path, save_path=None)
        X = sample_df_with_target.drop(columns=["readmitted"])
        proba = model.predict_proba(X)
        assert all(0.0 <= p <= 1.0 for p in proba)

    def test_predict_binary_output(self, sample_df_with_target, tmp_path):
        model = ClinicalRiskModel(n_estimators=10)
        data_path = str(tmp_path / "test_data.csv")
        sample_df_with_target.to_csv(data_path, index=False)
        model.train(data_path, save_path=None)
        X = sample_df_with_target.drop(columns=["readmitted"])
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_feature_importance_returns_list(self, sample_df_with_target, tmp_path):
        model = ClinicalRiskModel(n_estimators=10)
        data_path = str(tmp_path / "test_data.csv")
        sample_df_with_target.to_csv(data_path, index=False)
        model.train(data_path, save_path=None)
        fi = model.feature_importance(top_n=5)
        assert isinstance(fi, list)
        assert len(fi) <= 5
        assert "feature" in fi[0]
        assert "importance" in fi[0]

    def test_save_and_load(self, sample_df_with_target, tmp_path):
        model = ClinicalRiskModel(n_estimators=10)
        data_path  = str(tmp_path / "test_data.csv")
        model_path = str(tmp_path / "model.pkl")
        sample_df_with_target.to_csv(data_path, index=False)
        model.train(data_path, save_path=model_path)
        loaded = ClinicalRiskModel.load(model_path)
        assert loaded.is_fitted is True
        assert loaded.eval_result_ is not None

    def test_loaded_model_can_predict(self, sample_df_with_target, tmp_path):
        model = ClinicalRiskModel(n_estimators=10)
        data_path  = str(tmp_path / "test_data.csv")
        model_path = str(tmp_path / "model.pkl")
        sample_df_with_target.to_csv(data_path, index=False)
        model.train(data_path, save_path=model_path)
        loaded = ClinicalRiskModel.load(model_path)
        X = sample_df_with_target.drop(columns=["readmitted"])
        proba = loaded.predict_proba(X)
        assert len(proba) == len(X)