"""Flask routes for optional ClinicalRiskModel (tabular XGBoost) artifact."""

import json

import pandas as pd
import pytest

from src.api.app import create_app
from src.models.clinical_risk_model import ClinicalRiskModel


def _one_tabular_record():
    """Single Diabetes-130-style row (matches tests/test_clinical_risk_model sample_df row 0)."""
    return {
        "encounter_id": 1,
        "patient_nbr": 101,
        "race": "Caucasian",
        "gender": "Male",
        "age": "[50-60)",
        "weight": "?",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 3,
        "payer_code": "MC",
        "medical_specialty": "Cardiology",
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 10,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "diag_1": "250.01",
        "diag_2": "401.9",
        "diag_3": "272.4",
        "number_diagnoses": 7,
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "metformin": "No",
        "insulin": "Steady",
        "glipizide": "No",
        "glyburide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "diabetesMed": "Yes",
        "change": "Ch",
        "examide": "No",
        "citoglipton": "No",
    }


@pytest.fixture
def sample_df_with_target():
    df = pd.DataFrame(
        {
            "encounter_id": list(range(1, 11)),
            "patient_nbr": list(range(101, 111)),
            "race": ["Caucasian"] * 10,
            "gender": ["Male", "Female"] * 5,
            "age": ["[50-60)"] * 5 + ["[70-80)"] * 5,
            "weight": ["?"] * 10,
            "admission_type_id": [1] * 10,
            "discharge_disposition_id": [1] * 10,
            "admission_source_id": [7] * 10,
            "time_in_hospital": [3, 5, 2, 7, 1, 4, 6, 3, 5, 2],
            "payer_code": ["MC"] * 10,
            "medical_specialty": ["Cardiology"] * 10,
            "num_lab_procedures": [40, 50, 30, 60, 20, 45, 55, 35, 48, 22],
            "num_procedures": [1, 2, 0, 3, 1, 2, 1, 0, 2, 1],
            "num_medications": [10, 15, 8, 20, 5, 12, 18, 7, 14, 9],
            "number_outpatient": [0, 1, 0, 2, 0, 1, 0, 0, 1, 0],
            "number_emergency": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            "number_inpatient": [1, 2, 0, 3, 0, 1, 2, 0, 1, 0],
            "diag_1": ["250.01"] * 10,
            "diag_2": ["401.9"] * 10,
            "diag_3": ["272.4"] * 10,
            "number_diagnoses": [7, 9, 5, 9, 3, 8, 9, 4, 7, 5],
            "max_glu_serum": ["None"] * 10,
            "A1Cresult": ["None"] * 10,
            "metformin": ["No", "Steady", "Up", "No", "No", "Steady", "Down", "No", "Up", "No"],
            "insulin": ["Steady"] * 5 + ["No"] * 5,
            "glipizide": ["No"] * 10,
            "glyburide": ["No"] * 10,
            "pioglitazone": ["No"] * 10,
            "rosiglitazone": ["No"] * 10,
            "diabetesMed": ["Yes"] * 8 + ["No"] * 2,
            "change": ["Ch", "No"] * 5,
            "examide": ["No"] * 10,
            "citoglipton": ["No"] * 10,
            "readmitted": ["NO", "<30", "NO", ">30", "NO", "<30", "NO", "NO", ">30", "NO"],
        }
    )
    return df


class TestClinicalRiskModelApi:
    def test_status_not_loaded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLINICAL_RISK_MODEL_PATH", str(tmp_path / "missing.pkl"))
        app = create_app(db_path=":memory:")
        with app.test_client() as c:
            r = c.get("/api/clinical-risk-model/status")
        assert r.status_code == 200
        data = r.get_json()
        assert data["loaded"] is False
        assert data["model_name"] == ClinicalRiskModel.MODEL_NAME
        assert "message" in data

    def test_predict_503_when_not_loaded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLINICAL_RISK_MODEL_PATH", str(tmp_path / "missing.pkl"))
        app = create_app(db_path=":memory:")
        with app.test_client() as c:
            r = c.post(
                "/api/clinical-risk-model/predict",
                data=json.dumps({"records": [_one_tabular_record()]}),
                content_type="application/json",
            )
        assert r.status_code == 503

    def test_predict_400_empty_records(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLINICAL_RISK_MODEL_PATH", str(tmp_path / "missing.pkl"))
        app = create_app(db_path=":memory:")
        with app.test_client() as c:
            r = c.post(
                "/api/clinical-risk-model/predict",
                data=json.dumps({"records": []}),
                content_type="application/json",
            )
        assert r.status_code == 400

    def test_status_and_predict_with_artifact(self, monkeypatch, tmp_path, sample_df_with_target):
        model_path = str(tmp_path / "crm.pkl")
        csv_path = str(tmp_path / "train.csv")
        sample_df_with_target.to_csv(csv_path, index=False)
        model = ClinicalRiskModel(n_estimators=10)
        model.train(csv_path, save_path=model_path)

        monkeypatch.setenv("CLINICAL_RISK_MODEL_PATH", model_path)
        app = create_app(db_path=":memory:")

        with app.test_client() as c:
            st = c.get("/api/clinical-risk-model/status")
        assert st.status_code == 200
        stj = st.get_json()
        assert stj["loaded"] is True
        assert stj["eval"] is not None
        assert isinstance(stj.get("feature_importance_top_10"), list)

        with app.test_client() as c:
            pr = c.post(
                "/api/clinical-risk-model/predict",
                data=json.dumps({"records": [_one_tabular_record()]}),
                content_type="application/json",
            )
        assert pr.status_code == 200
        pj = pr.get_json()
        assert pj["n_records"] == 1
        assert len(pj["readmission_prob_30d"]) == 1
        assert 0.0 <= pj["readmission_prob_30d"][0] <= 1.0
