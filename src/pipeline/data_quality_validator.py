"""
data_quality_validator.py
──────────────────────────
Data Quality Plan (DQP) compliance validator for clinical trial data.

Validates clinical notes against regulatory standards:
  - HIPAA Safe Harbor compliance
  - Data completeness checks
  - Statistical anomaly detection
  - ICH E6 (GCP) quality standards

Design decision: Separate validator from cleaner — DQP validation is a
distinct regulatory requirement that must be auditable independently.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Result of a single quality check."""
    check_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    issues: List[str]
    severity: str  # 'critical', 'major', 'minor'
    timestamp: str


@dataclass
class DQPReport:
    """Data Quality Plan compliance report."""
    note_id: int
    overall_score: float
    passed: bool
    checks: List[QualityCheckResult]
    recommendations: List[str]
    generated_at: str


class DataQualityValidator:
    """
    Validates clinical notes against DQP standards.
    
    Implements quality checks required for regulatory submissions:
      - Completeness validation
      - Consistency checks
      - Anomaly detection
      - Compliance verification
    
    Parameters
    ----------
    db_path : path to SQLite database
    strict_mode : if True, fail on any quality issue
    """
    
    def __init__(self, db_path: str = "data/clinicalner.db", strict_mode: bool = False):
        self.db_path = Path(db_path)
        self.strict_mode = strict_mode
        self._init_quality_tables()
        logger.info("DataQualityValidator ready | strict_mode=%s", strict_mode)
    
    def _init_quality_tables(self) -> None:
        """Create quality_checks table if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_id INTEGER,
                    check_name TEXT,
                    passed INTEGER,
                    score REAL,
                    severity TEXT,
                    issues TEXT,
                    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    # ── Public API ────────────────────────────────────────────────────────────
    
    def validate_note(self, note_id: int, original_text: str, 
                     processed_text: str, entities: List[Dict]) -> DQPReport:
        """
        Run full DQP validation on a processed note.
        
        Returns
        -------
        DQPReport with overall score and individual check results
        """
        checks = []
        
        # Run all quality checks
        checks.append(self._check_completeness(original_text, processed_text))
        checks.append(self._check_deidentification_quality(entities))
        checks.append(self._check_text_integrity(original_text, processed_text))
        checks.append(self._check_hipaa_compliance(entities))
        checks.append(self._check_consistency(original_text, processed_text, entities))
        
        # Calculate overall score
        overall_score = np.mean([c.score for c in checks])
        passed = all(c.passed for c in checks) if self.strict_mode else overall_score >= 0.8
        
        # Generate recommendations
        recommendations = self._generate_recommendations(checks)
        
        report = DQPReport(
            note_id=note_id,
            overall_score=round(overall_score, 3),
            passed=passed,
            checks=checks,
            recommendations=recommendations,
            generated_at=datetime.now().isoformat()
        )
        
        # Save to database
        self._save_quality_checks(note_id, checks)
        
        return report
    
    def validate_batch(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate a batch of processed notes.
        
        Parameters
        ----------
        notes_df : DataFrame with columns: note_id, original_text, 
                   masked_text, entities_json
        
        Returns
        -------
        DataFrame with quality scores and pass/fail status
        """
        results = []
        
        for _, row in notes_df.iterrows():
            import json
            entities = json.loads(row.get('entities_json', '[]'))
            
            report = self.validate_note(
                note_id=row['note_id'],
                original_text=row.get('original_text', ''),
                processed_text=row.get('masked_text', ''),
                entities=entities
            )
            
            results.append({
                'note_id': report.note_id,
                'quality_score': report.overall_score,
                'passed': report.passed,
                'critical_issues': sum(1 for c in report.checks if c.severity == 'critical' and not c.passed),
                'recommendations': '; '.join(report.recommendations[:3])
            })
        
        return pd.DataFrame(results)
    
    def detect_anomalies(self, notes_df: pd.DataFrame, 
                        contamination: float = 0.1) -> pd.DataFrame:
        """
        Statistical anomaly detection on processed notes.
        
        Identifies notes with unusual characteristics that may indicate
        processing errors or data quality issues.
        
        Parameters
        ----------
        notes_df : DataFrame with processed notes
        contamination : expected proportion of outliers (0.0 to 0.5)
        
        Returns
        -------
        DataFrame with anomaly scores and flags
        """
        from sklearn.ensemble import IsolationForest
        
        # Extract features
        features = []
        for _, row in notes_df.iterrows():
            features.append([
                len(row.get('original_text', '')),
                len(row.get('masked_text', '')),
                row.get('entity_count', 0),
                len(row.get('original_text', '')) - len(row.get('masked_text', '')),
            ])
        
        X = np.array(features)
        
        # Fit isolation forest
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(X)
        scores = clf.score_samples(X)
        
        notes_df['anomaly_score'] = scores
        notes_df['is_anomaly'] = predictions == -1
        
        logger.info("Anomaly detection: %d/%d flagged", 
                   sum(predictions == -1), len(notes_df))
        
        return notes_df
    
    def generate_quality_summary(self) -> Dict:
        """
        Generate summary statistics from quality_checks table.
        
        Returns
        -------
        Dict with overall quality metrics for the study
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT 
                    COUNT(DISTINCT note_id) as total_notes_checked,
                    AVG(score) as avg_quality_score,
                    SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pass_rate,
                    COUNT(CASE WHEN severity = 'critical' AND passed = 0 THEN 1 END) as critical_failures
                FROM quality_checks
            """, conn)
            
            by_check = pd.read_sql_query("""
                SELECT 
                    check_name,
                    AVG(score) as avg_score,
                    SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pass_rate
                FROM quality_checks
                GROUP BY check_name
                ORDER BY pass_rate ASC
            """, conn)
        
        return {
            'summary': df.iloc[0].to_dict() if not df.empty else {},
            'by_check': by_check.to_dict(orient='records')
        }
    
    # ── Quality Checks ────────────────────────────────────────────────────────
    
    def _check_completeness(self, original: str, processed: str) -> QualityCheckResult:
        """Verify no critical data loss during processing."""
        issues = []
        
        # Check for excessive text loss
        orig_len = len(original)
        proc_len = len(processed)
        retention_rate = proc_len / orig_len if orig_len > 0 else 0
        
        if retention_rate < 0.5:
            issues.append(f"Excessive text loss: {(1-retention_rate)*100:.1f}% removed")
        
        # Check for complete deletion
        if proc_len == 0 and orig_len > 0:
            issues.append("Complete text deletion detected")
        
        score = min(retention_rate * 1.2, 1.0)  # Allow up to 20% loss
        passed = retention_rate >= 0.5
        
        return QualityCheckResult(
            check_name="completeness",
            passed=passed,
            score=score,
            issues=issues,
            severity='critical' if not passed else 'minor',
            timestamp=datetime.now().isoformat()
        )
    
    def _check_deidentification_quality(self, entities: List[Dict]) -> QualityCheckResult:
        """Verify de-identification was performed."""
        issues = []
        
        entity_count = len(entities)
        
        if entity_count == 0:
            issues.append("No PHI entities detected - may indicate processing failure")
            score = 0.5
            passed = False
        else:
            # Check entity diversity
            entity_types = set(e.get('label', '') for e in entities)
            diversity = len(entity_types) / 8  # Expect up to 8 types
            
            if diversity < 0.2:
                issues.append(f"Low entity diversity: only {len(entity_types)} types found")
            
            score = min(diversity * 1.5, 1.0)
            passed = entity_count > 0
        
        return QualityCheckResult(
            check_name="deidentification_quality",
            passed=passed,
            score=score,
            issues=issues,
            severity='major' if not passed else 'minor',
            timestamp=datetime.now().isoformat()
        )
    
    def _check_text_integrity(self, original: str, processed: str) -> QualityCheckResult:
        """Verify text structure is maintained."""
        issues = []
        
        # Check sentence structure preserved
        orig_sentences = len(re.findall(r'[.!?]+', original))
        proc_sentences = len(re.findall(r'[.!?]+', processed))
        
        if orig_sentences > 0:
            sentence_retention = proc_sentences / orig_sentences
            if sentence_retention < 0.7:
                issues.append(f"Sentence structure degraded: {sentence_retention*100:.1f}% retained")
        else:
            sentence_retention = 1.0
        
        # Check for malformed masks
        malformed = re.findall(r'\[\[|\]\]|\[(?![A-Z]+\])', processed)
        if malformed:
            issues.append(f"Malformed masks detected: {len(malformed)} instances")
        
        score = sentence_retention * (0.9 if malformed else 1.0)
        passed = sentence_retention >= 0.7 and len(malformed) == 0
        
        return QualityCheckResult(
            check_name="text_integrity",
            passed=passed,
            score=score,
            issues=issues,
            severity='major' if not passed else 'minor',
            timestamp=datetime.now().isoformat()
        )
    
    def _check_hipaa_compliance(self, entities: List[Dict]) -> QualityCheckResult:
        """Verify HIPAA Safe Harbor identifiers are addressed."""
        issues = []
        
        # HIPAA Safe Harbor requires removal of 18 identifier types
        # We check for common ones
        required_types = {'PERSON', 'DATE', 'LOCATION', 'PHONE', 'AGE', 'ID'}
        found_types = set(e.get('label', '') for e in entities)
        
        coverage = len(found_types & required_types) / len(required_types)
        
        if coverage < 0.5:
            issues.append(f"Low HIPAA identifier coverage: {coverage*100:.1f}%")
        
        score = coverage
        passed = coverage >= 0.4  # At least 40% of common types
        
        return QualityCheckResult(
            check_name="hipaa_compliance",
            passed=passed,
            score=score,
            issues=issues,
            severity='critical' if not passed else 'minor',
            timestamp=datetime.now().isoformat()
        )
    
    def _check_consistency(self, original: str, processed: str, 
                          entities: List[Dict]) -> QualityCheckResult:
        """Verify consistency between entities and masked text."""
        issues = []
        
        # Count mask tokens in processed text
        mask_count = len(re.findall(r'\[[A-Z]+\]', processed))
        entity_count = len(entities)
        
        if mask_count != entity_count:
            issues.append(f"Mask/entity mismatch: {mask_count} masks vs {entity_count} entities")
        
        # Check for residual PHI patterns (simple heuristic)
        residual_patterns = [
            (r'\b\d{3}-\d{3}-\d{4}\b', 'phone'),
            (r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', 'date'),
            (r'\bMRN\d+\b', 'MRN'),
        ]
        
        residual_found = []
        for pattern, name in residual_patterns:
            if re.search(pattern, processed):
                residual_found.append(name)
        
        if residual_found:
            issues.append(f"Potential residual PHI: {', '.join(residual_found)}")
        
        consistency_score = 1.0 if mask_count == entity_count else 0.8
        residual_penalty = 0.3 if residual_found else 0.0
        
        score = max(consistency_score - residual_penalty, 0.0)
        passed = mask_count == entity_count and not residual_found
        
        return QualityCheckResult(
            check_name="consistency",
            passed=passed,
            score=score,
            issues=issues,
            severity='critical' if residual_found else 'major',
            timestamp=datetime.now().isoformat()
        )
    
    # ── Helpers ───────────────────────────────────────────────────────────────
    
    def _generate_recommendations(self, checks: List[QualityCheckResult]) -> List[str]:
        """Generate actionable recommendations based on check results."""
        recommendations = []
        
        for check in checks:
            if not check.passed:
                if check.check_name == 'completeness':
                    recommendations.append("Review masking rules to reduce text loss")
                elif check.check_name == 'deidentification_quality':
                    recommendations.append("Verify NER model is loaded and functioning")
                elif check.check_name == 'text_integrity':
                    recommendations.append("Check for malformed entity masks")
                elif check.check_name == 'hipaa_compliance':
                    recommendations.append("Expand entity recognition to cover more HIPAA identifiers")
                elif check.check_name == 'consistency':
                    recommendations.append("Manual review required for residual PHI")
        
        if not recommendations:
            recommendations.append("All quality checks passed - note ready for submission")
        
        return recommendations
    
    def _save_quality_checks(self, note_id: int, checks: List[QualityCheckResult]) -> None:
        """Save quality check results to database."""
        with sqlite3.connect(self.db_path) as conn:
            for check in checks:
                conn.execute("""
                    INSERT INTO quality_checks 
                    (note_id, check_name, passed, score, severity, issues)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    note_id,
                    check.check_name,
                    1 if check.passed else 0,
                    check.score,
                    check.severity,
                    '; '.join(check.issues) if check.issues else None
                ))
