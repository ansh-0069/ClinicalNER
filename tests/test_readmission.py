"""Tests for ReadmissionPredictor and /api/predict-readmission route."""

import sys

import pytest

sys.path.insert(0, ".")

from src.pipeline.readmission_predictor import ReadmissionPredictor, ReadmissionPrediction


def make_note(note_id=1, text_len=300, labels=None):
	labels = labels or ["DATE", "PHONE", "MRN"]
	entities = [{"label": label} for label in labels]
	return {
		"id": note_id,
		"text": "x" * text_len,
		"entities": entities,
	}


def corpus(n=80):
	notes = []
	for i in range(n):
		labels = ["DATE", "PHONE", "MRN"] if i % 3 else ["DATE", "DOB", "MRN", "PHONE"]
		notes.append(make_note(note_id=i + 1, text_len=200 + (i % 20) * 10, labels=labels))
	return notes


def test_fit_sets_fitted_state():
	predictor = ReadmissionPredictor()
	predictor.fit(corpus(60))
	assert predictor.is_fitted is True


def test_predict_requires_fit():
	predictor = ReadmissionPredictor()
	with pytest.raises(RuntimeError, match="fit"):
		predictor.predict_one(make_note())


def test_predict_one_returns_prediction_object():
	predictor = ReadmissionPredictor().fit(corpus(60))
	result = predictor.predict_one(make_note(note_id=99))
	assert isinstance(result, ReadmissionPrediction)
	assert result.note_id == 99
	assert 0.0 <= result.risk_score <= 1.0
	assert result.risk_level in {"low", "medium", "high"}


def test_predict_batch_count_matches_input():
	predictor = ReadmissionPredictor().fit(corpus(60))
	results = predictor.predict_batch([make_note(note_id=1), make_note(note_id=2)])
	assert len(results) == 2


def test_model_stats_shape():
	predictor = ReadmissionPredictor().fit(corpus(60))
	stats = predictor.model_stats()
	assert stats["is_fitted"] is True
	assert stats["training_samples"] == 60
	assert "phi_density" in stats["features"]


@pytest.fixture
def client():
	from src.api.app import create_app

	app = create_app(db_path=":memory:")
	app.config["TESTING"] = True

	# Pre-fit predictor for API tests to avoid DB bootstrap dependency.
	app.config["PREDICTOR"].fit(corpus(80))

	with app.test_client() as c:
		yield c


def test_predict_readmission_single_note(client):
	res = client.post("/api/predict-readmission", json=make_note(note_id=7))
	assert res.status_code == 200
	data = res.get_json()
	assert "risk_score" in data
	assert "model_stats" in data


def test_predict_readmission_batch_notes(client):
	payload = {"notes": [make_note(note_id=1), make_note(note_id=2)]}
	res = client.post("/api/predict-readmission", json=payload)
	assert res.status_code == 200
	data = res.get_json()
	assert data["count"] == 2
	assert len(data["results"]) == 2


def test_predict_readmission_requires_json(client):
	res = client.post("/api/predict-readmission", data="plain-text")
	assert res.status_code == 400
