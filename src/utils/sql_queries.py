"""
sql_queries.py
──────────────
SQL queries for clinical trial data analysis.

Demonstrates SQL proficiency for the Associate Clinical Programmer role.
Queries support:
  - Study status monitoring
  - Regulatory reporting
  - Data quality assessment
  - Cross-study analytics
"""

# ── Study Management Queries ──────────────────────────────────────────────────

STUDY_SUMMARY = """
    SELECT 
        cn.medical_specialty as study_arm,
        COUNT(DISTINCT cn.note_id) as total_notes,
        COUNT(DISTINCT pn.note_id) as processed_notes,
        ROUND(COUNT(DISTINCT pn.note_id) * 100.0 / COUNT(DISTINCT cn.note_id), 2) as completion_pct,
        AVG(pn.entity_count) as avg_phi_per_note,
        MAX(pn.entity_count) as max_phi_per_note,
        MIN(pn.processed_at) as first_processed,
        MAX(pn.processed_at) as last_processed
    FROM clinical_notes cn
    LEFT JOIN processed_notes pn ON cn.note_id = pn.note_id
    GROUP BY cn.medical_specialty
    ORDER BY total_notes DESC
"""

PROCESSING_TIMELINE = """
    SELECT 
        DATE(processed_at) as processing_date,
        COUNT(*) as notes_processed,
        AVG(entity_count) as avg_entities,
        SUM(entity_count) as total_entities
    FROM processed_notes
    WHERE processed_at >= DATE('now', '-30 days')
    GROUP BY DATE(processed_at)
    ORDER BY processing_date DESC
"""

# ── Quality Control Queries ───────────────────────────────────────────────────

QUALITY_METRICS = """
    SELECT 
        qc.check_name,
        COUNT(*) as total_checks,
        SUM(CASE WHEN qc.passed = 1 THEN 1 ELSE 0 END) as passed_count,
        ROUND(AVG(qc.score) * 100, 2) as avg_score_pct,
        COUNT(CASE WHEN qc.severity = 'critical' AND qc.passed = 0 THEN 1 END) as critical_failures
    FROM quality_checks qc
    GROUP BY qc.check_name
    ORDER BY avg_score_pct ASC
"""

FAILED_QUALITY_CHECKS = """
    SELECT 
        qc.note_id,
        cn.medical_specialty,
        qc.check_name,
        qc.score,
        qc.severity,
        qc.issues,
        qc.checked_at
    FROM quality_checks qc
    JOIN clinical_notes cn ON qc.note_id = cn.note_id
    WHERE qc.passed = 0 AND qc.severity IN ('critical', 'major')
    ORDER BY 
        CASE qc.severity 
            WHEN 'critical' THEN 1 
            WHEN 'major' THEN 2 
            ELSE 3 
        END,
        qc.checked_at DESC
"""

# ── Regulatory Compliance Queries ─────────────────────────────────────────────

AUDIT_TRAIL_SUMMARY = """
    SELECT 
        event_type,
        COUNT(*) as event_count,
        COUNT(DISTINCT user_id) as unique_users,
        MIN(timestamp) as first_occurrence,
        MAX(timestamp) as last_occurrence
    FROM audit_log
    WHERE timestamp >= DATE('now', '-7 days')
    GROUP BY event_type
    ORDER BY event_count DESC
"""

REGULATORY_SUBMISSION_READINESS = """
    SELECT 
        'Total Notes' as metric,
        COUNT(*) as value
    FROM clinical_notes
    
    UNION ALL
    
    SELECT 
        'Processed Notes' as metric,
        COUNT(*) as value
    FROM processed_notes
    
    UNION ALL
    
    SELECT 
        'Quality Passed' as metric,
        COUNT(DISTINCT note_id) as value
    FROM quality_checks
    WHERE passed = 1
    
    UNION ALL
    
    SELECT 
        'Audit Events' as metric,
        COUNT(*) as value
    FROM audit_log
    
    UNION ALL
    
    SELECT 
        'Critical Issues' as metric,
        COUNT(*) as value
    FROM quality_checks
    WHERE passed = 0 AND severity = 'critical'
"""

# ── PHI Analysis Queries ──────────────────────────────────────────────────────

PHI_BY_SPECIALTY = """
    SELECT 
        cn.medical_specialty,
        COUNT(pn.note_id) as notes_processed,
        AVG(pn.entity_count) as avg_phi_per_note,
        SUM(pn.entity_count) as total_phi_detected,
        MAX(pn.entity_count) as max_phi_in_note
    FROM processed_notes pn
    JOIN clinical_notes cn ON pn.note_id = cn.note_id
    GROUP BY cn.medical_specialty
    ORDER BY avg_phi_per_note DESC
"""

HIGH_RISK_NOTES = """
    SELECT 
        pn.note_id,
        cn.medical_specialty,
        pn.entity_count,
        pn.processed_at,
        CASE 
            WHEN pn.entity_count > 10 THEN 'High Risk'
            WHEN pn.entity_count > 5 THEN 'Medium Risk'
            ELSE 'Low Risk'
        END as risk_level
    FROM processed_notes pn
    JOIN clinical_notes cn ON pn.note_id = cn.note_id
    WHERE pn.entity_count > 5
    ORDER BY pn.entity_count DESC
"""

# ── Cross-Study Analytics ─────────────────────────────────────────────────────

ENTITY_TYPE_DISTRIBUTION = """
    SELECT 
        cn.medical_specialty,
        pn.entity_types_json,
        COUNT(*) as note_count
    FROM processed_notes pn
    JOIN clinical_notes cn ON pn.note_id = cn.note_id
    WHERE pn.entity_types_json IS NOT NULL
    GROUP BY cn.medical_specialty, pn.entity_types_json
"""

PROCESSING_EFFICIENCY = """
    SELECT 
        DATE(pn.processed_at) as date,
        COUNT(*) as notes_processed,
        AVG(LENGTH(cn.transcription)) as avg_note_length,
        AVG(pn.entity_count) as avg_entities,
        ROUND(AVG(pn.entity_count) * 1.0 / AVG(LENGTH(cn.transcription)) * 1000, 2) as phi_density_per_1k_chars
    FROM processed_notes pn
    JOIN clinical_notes cn ON pn.note_id = cn.note_id
    WHERE pn.processed_at >= DATE('now', '-30 days')
    GROUP BY DATE(pn.processed_at)
    ORDER BY date DESC
"""

# ── Data Completeness Queries ─────────────────────────────────────────────────

MISSING_DATA_ANALYSIS = """
    SELECT 
        'Clinical Notes' as table_name,
        COUNT(*) as total_records,
        SUM(CASE WHEN transcription IS NULL OR transcription = '' THEN 1 ELSE 0 END) as missing_transcription,
        SUM(CASE WHEN medical_specialty IS NULL OR medical_specialty = '' THEN 1 ELSE 0 END) as missing_specialty
    FROM clinical_notes
    
    UNION ALL
    
    SELECT 
        'Processed Notes' as table_name,
        COUNT(*) as total_records,
        SUM(CASE WHEN masked_text IS NULL OR masked_text = '' THEN 1 ELSE 0 END) as missing_masked_text,
        SUM(CASE WHEN entities_json IS NULL THEN 1 ELSE 0 END) as missing_entities
    FROM processed_notes
"""

# ── Performance Monitoring ────────────────────────────────────────────────────

SYSTEM_PERFORMANCE = """
    SELECT 
        'Processing Rate' as metric,
        ROUND(COUNT(*) * 1.0 / 
            (JULIANDAY(MAX(processed_at)) - JULIANDAY(MIN(processed_at)) + 1), 2) as value,
        'notes/day' as unit
    FROM processed_notes
    WHERE processed_at >= DATE('now', '-7 days')
    
    UNION ALL
    
    SELECT 
        'Avg Processing Time' as metric,
        ROUND(AVG(
            (JULIANDAY(pn.processed_at) - JULIANDAY(cn.created_at)) * 24 * 60
        ), 2) as value,
        'minutes' as unit
    FROM processed_notes pn
    JOIN clinical_notes cn ON pn.note_id = cn.note_id
    WHERE pn.processed_at >= DATE('now', '-7 days')
"""

# ── Query Templates ───────────────────────────────────────────────────────────

def get_notes_by_date_range(start_date: str, end_date: str) -> str:
    """
    Get all notes processed within a date range.
    
    Parameters
    ----------
    start_date : YYYY-MM-DD
    end_date : YYYY-MM-DD
    """
    return f"""
        SELECT 
            pn.note_id,
            cn.medical_specialty,
            pn.entity_count,
            pn.processed_at
        FROM processed_notes pn
        JOIN clinical_notes cn ON pn.note_id = cn.note_id
        WHERE DATE(pn.processed_at) BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY pn.processed_at DESC
    """

def get_quality_report_for_specialty(specialty: str) -> str:
    """
    Get quality metrics for a specific medical specialty.
    
    Parameters
    ----------
    specialty : medical specialty name
    """
    return f"""
        SELECT 
            qc.check_name,
            COUNT(*) as total_checks,
            SUM(CASE WHEN qc.passed = 1 THEN 1 ELSE 0 END) as passed,
            ROUND(AVG(qc.score) * 100, 2) as avg_score
        FROM quality_checks qc
        JOIN clinical_notes cn ON qc.note_id = cn.note_id
        WHERE cn.medical_specialty = '{specialty}'
        GROUP BY qc.check_name
    """

# ── Query Catalog ─────────────────────────────────────────────────────────────

QUERY_CATALOG = {
    # Study Management
    "study_summary": STUDY_SUMMARY,
    "processing_timeline": PROCESSING_TIMELINE,
    
    # Quality Control
    "quality_metrics": QUALITY_METRICS,
    "failed_quality_checks": FAILED_QUALITY_CHECKS,
    
    # Regulatory
    "audit_trail_summary": AUDIT_TRAIL_SUMMARY,
    "regulatory_readiness": REGULATORY_SUBMISSION_READINESS,
    
    # PHI Analysis
    "phi_by_specialty": PHI_BY_SPECIALTY,
    "high_risk_notes": HIGH_RISK_NOTES,
    
    # Analytics
    "entity_distribution": ENTITY_TYPE_DISTRIBUTION,
    "processing_efficiency": PROCESSING_EFFICIENCY,
    
    # Data Quality
    "missing_data": MISSING_DATA_ANALYSIS,
    "system_performance": SYSTEM_PERFORMANCE,
}
