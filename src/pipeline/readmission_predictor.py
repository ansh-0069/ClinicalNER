"""
readmission_predictor.py
────────────────────────
Lightweight readmission risk predictor for de-identified clinical notes.

The model intentionally uses corpus-derived feature baselines so it can
train itself from processed notes without requiring labeled outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from statistics import mean, pstdev
from typing import Any


@dataclass
class ReadmissionPrediction:
	note_id: int | None
	risk_score: float
	risk_level: str
	confidence: float
	top_factors: list[str] = field(default_factory=list)

	def to_dict(self) -> dict[str, Any]:
		return {
			"note_id": self.note_id,
			"risk_score": self.risk_score,
			"risk_level": self.risk_level,
			"confidence": self.confidence,
			"top_factors": self.top_factors,
		}


class ReadmissionPredictor:
	"""Self-calibrating clinical text risk scorer."""

	def __init__(self) -> None:
		self.is_fitted = False
		self._n_train = 0
		self._feature_means: dict[str, float] = {}
		self._feature_stds: dict[str, float] = {}

	def fit(self, corpus: list[dict[str, Any]]) -> "ReadmissionPredictor":
		if len(corpus) < 20:
			raise ValueError("Need at least 20 notes to fit predictor")

		feats = [self._extract_features(note) for note in corpus]
		keys = feats[0].keys()
		self._feature_means = {k: mean([f[k] for f in feats]) for k in keys}
		self._feature_stds = {k: max(pstdev([f[k] for f in feats]), 1e-6) for k in keys}
		self._n_train = len(corpus)
		self.is_fitted = True
		return self

	def predict_one(self, note: dict[str, Any]) -> ReadmissionPrediction:
		if not self.is_fitted:
			raise RuntimeError("Call fit() before predict_one().")

		feat = self._extract_features(note)
		score, factors = self._score_from_features(feat)
		level = "low" if score < 0.35 else "medium" if score < 0.7 else "high"
		confidence = round(min(0.99, max(0.55, 0.55 + abs(score - 0.5))), 3)

		return ReadmissionPrediction(
			note_id=self._safe_int(note.get("id") if isinstance(note, dict) else None),
			risk_score=round(score, 4),
			risk_level=level,
			confidence=confidence,
			top_factors=factors,
		)

	def predict_batch(self, notes: list[dict[str, Any]]) -> list[ReadmissionPrediction]:
		return [self.predict_one(n) for n in notes]

	def model_stats(self) -> dict[str, Any]:
		return {
			"is_fitted": self.is_fitted,
			"training_samples": self._n_train,
			"features": list(self._feature_means.keys()),
		}

	def _extract_features(self, note: dict[str, Any]) -> dict[str, float]:
		text = str(note.get("text") or "")
		entities = note.get("entities") or []
		phi_count = float(len(entities))
		text_len = float(max(len(text), 1))
		phi_density = phi_count / (text_len / 100.0)

		label_counts: dict[str, int] = {}
		for e in entities:
			if isinstance(e, dict):
				label = str(e.get("label") or "").upper()
			else:
				label = str(getattr(e, "label", "")).upper()
			if label:
				label_counts[label] = label_counts.get(label, 0) + 1

		return {
			"phi_count": phi_count,
			"phi_density": float(phi_density),
			"text_length": text_len,
			"label_diversity": float(len(label_counts)),
			"date_mentions": float(label_counts.get("DATE", 0) + label_counts.get("DOB", 0)),
			"identity_markers": float(label_counts.get("MRN", 0) + label_counts.get("PHONE", 0)),
		}

	def _score_from_features(self, feat: dict[str, float]) -> tuple[float, list[str]]:
		z = {}
		for k, v in feat.items():
			z[k] = (v - self._feature_means[k]) / self._feature_stds[k]

		weighted = (
			0.28 * z["phi_density"]
			+ 0.22 * z["phi_count"]
			+ 0.17 * z["identity_markers"]
			+ 0.15 * z["date_mentions"]
			+ 0.10 * z["label_diversity"]
			+ 0.08 * z["text_length"]
		)

		score = 1.0 / (1.0 + exp(-weighted))

		factor_order = sorted(z.items(), key=lambda kv: abs(kv[1]), reverse=True)
		factors = []
		for key, zval in factor_order[:3]:
			direction = "elevated" if zval >= 0 else "reduced"
			factors.append(f"{key.replace('_', ' ')} {direction}")

		return float(score), factors

	@staticmethod
	def _safe_int(value: Any) -> int | None:
		try:
			return int(value)
		except Exception:
			return None
