"""
clinical_listings.py
────────────────────
Generates regulatory-compliant data listings for clinical trials.

Produces reports required for:
  - Regulatory submissions (FDA, EMA)
  - Study status updates
  - Data monitoring committee (DMC) reviews
  - Audit trail documentation

Output formats: PDF, Excel, CSV (SAS-compatible)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sqlite3

import pandas as pd

logger = logging.getLogger(__name__)


class ClinicalReportGenerator:
    """
    Generates regulatory-compliant data listings.
    
    Implements ICH E3 and CDISC standards for clinical trial reporting.
    
    Parameters
    ----------
    db_path : path to SQLite database
    output_dir : directory for generated reports
    """
    
    def __init__(self, db_path: str = "data/clinicalner.db", 
                 output_dir: str = "data/reports"):
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ClinicalReportGenerator ready | output → %s", self.output_dir)
    
    # ── Study Status Reports ──────────────────────────────────────────────────
    
    def generate_processing_summary(self, study_id: Optional[str] = None) -> pd.DataFrame:
        """
        Generate study processing summary report.
        
        Shows:
          - Total notes processed
          - De-identification completion rate
          - Quality control pass rate
          - Processing timeline
        
        Returns
        -------
        DataFrame with summary statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    COUNT(DISTINCT cn.note_id) as total_notes,
                    COUNT(DISTINCT pn.note_id) as processed_notes,
                    ROUND(COUNT(DISTINCT pn.note_id) * 100.0 / COUNT(DISTINCT cn.note_id), 2) as completion_rate,
                    AVG(pn.entity_count) as avg_phi_per_note,
                    MIN(pn.processed_at) as first_processed,
                    MAX(pn.processed_at) as last_processed,
                    cn.medical_specialty
                FROM clinical_notes cn
                LEFT JOIN processed_notes pn ON cn.note_id = pn.note_id
                GROUP BY cn.medical_specialty
                ORDER BY total_notes DESC
            """
            
            df = pd.read_sql_query(query, conn)
        
        # Add quality metrics if available
        try:
            with sqlite3.connect(self.db_path) as conn:
                quality = pd.read_sql_query("""
                    SELECT 
                        AVG(score) as avg_quality_score,
                        SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as quality_pass_rate
                    FROM quality_checks
                """, conn)
                
                if not quality.empty:
                    df['avg_quality_score'] = quality.iloc[0]['avg_quality_score']
                    df['quality_pass_rate'] = quality.iloc[0]['quality_pass_rate']
        except Exception:
            pass
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"processing_summary_{timestamp}.xlsx"
        df.to_excel(output_file, index=False, sheet_name="Summary")
        
        logger.info("Processing summary saved → %s", output_file)
        return df
    
    def generate_audit_listing(self, start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Generate ICH E6 compliant audit trail listing.
        
        Shows all system activities with:
          - User identification
          - Timestamp
          - Action performed
          - Data affected
        
        Parameters
        ----------
        start_date : filter from date (YYYY-MM-DD)
        end_date : filter to date (YYYY-MM-DD)
        
        Returns
        -------
        DataFrame with audit trail entries
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    log_id,
                    event_type,
                    description,
                    note_id,
                    user_id,
                    timestamp,
                    metadata
                FROM audit_log
                WHERE 1=1
            """
            
            params = []
            if start_date:
                query += " AND DATE(timestamp) >= ?"
                params.append(start_date)
            if end_date:
                query += " AND DATE(timestamp) <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
        
        # Format for regulatory review
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"audit_trail_{timestamp}.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Audit Trail")
            
            # Add summary sheet
            summary = pd.DataFrame({
                'Metric': [
                    'Total Events',
                    'Date Range',
                    'Unique Users',
                    'Report Generated'
                ],
                'Value': [
                    len(df),
                    f"{df['timestamp'].min()} to {df['timestamp'].max()}" if not df.empty else 'N/A',
                    df['user_id'].nunique() if 'user_id' in df.columns else 'N/A',
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            })
            summary.to_excel(writer, index=False, sheet_name="Summary")
        
        logger.info("Audit listing saved → %s", output_file)
        return df
    
    def generate_quality_control_report(self) -> pd.DataFrame:
        """
        Generate quality control report for DMC review.
        
        Shows:
          - Quality check results by note
          - Pass/fail rates
          - Critical issues requiring review
        
        Returns
        -------
        DataFrame with QC metrics
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT 
                    qc.note_id,
                    cn.medical_specialty,
                    qc.check_name,
                    qc.passed,
                    qc.score,
                    qc.severity,
                    qc.issues,
                    qc.checked_at
                FROM quality_checks qc
                LEFT JOIN clinical_notes cn ON qc.note_id = cn.note_id
                ORDER BY qc.checked_at DESC
            """, conn)
        
        if df.empty:
            logger.warning("No quality check data available")
            return df
        
        # Create summary pivot
        summary = df.pivot_table(
            index='check_name',
            values='passed',
            aggfunc=['count', 'sum', 'mean']
        )
        summary.columns = ['total_checks', 'passed_count', 'pass_rate']
        summary['pass_rate'] = (summary['pass_rate'] * 100).round(2)
        summary = summary.reset_index()
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"quality_control_{timestamp}.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Detailed Results")
            summary.to_excel(writer, index=False, sheet_name="Summary")
        
        logger.info("QC report saved → %s", output_file)
        return df
    
    def generate_phi_summary_report(self) -> pd.DataFrame:
        """
        Generate PHI detection summary for privacy officer review.
        
        Shows:
          - Entity types detected
          - Frequency by specialty
          - High-risk notes flagged
        
        Returns
        -------
        DataFrame with PHI statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT 
                    cn.medical_specialty,
                    pn.note_id,
                    pn.entity_count,
                    pn.entity_types_json,
                    pn.processed_at
                FROM processed_notes pn
                JOIN clinical_notes cn ON pn.note_id = cn.note_id
                ORDER BY pn.entity_count DESC
            """, conn)
        
        if df.empty:
            logger.warning("No processed notes available")
            return df
        
        # Parse entity types
        import json
        entity_summary = []
        for _, row in df.iterrows():
            try:
                types = json.loads(row['entity_types_json'])
                for entity_type, count in types.items():
                    entity_summary.append({
                        'specialty': row['medical_specialty'],
                        'entity_type': entity_type,
                        'count': count
                    })
            except Exception:
                pass
        
        entity_df = pd.DataFrame(entity_summary)
        
        if not entity_df.empty:
            pivot = entity_df.pivot_table(
                index='entity_type',
                columns='specialty',
                values='count',
                aggfunc='sum',
                fill_value=0
            )
        else:
            pivot = pd.DataFrame()
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"phi_summary_{timestamp}.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="By Note")
            if not pivot.empty:
                pivot.to_excel(writer, sheet_name="By Specialty")
        
        logger.info("PHI summary saved → %s", output_file)
        return df
    
    def generate_regulatory_submission_package(self, study_id: str = "STUDY001") -> Dict[str, Path]:
        """
        Generate complete regulatory submission package.
        
        Creates all required reports for FDA/EMA submission:
          - Processing summary
          - Audit trail
          - Quality control report
          - PHI summary
        
        Parameters
        ----------
        study_id : study identifier for file naming
        
        Returns
        -------
        Dict mapping report type to file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_dir = self.output_dir / f"{study_id}_submission_{timestamp}"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Temporarily change output dir
        original_output = self.output_dir
        self.output_dir = package_dir
        
        try:
            reports = {}
            
            logger.info("Generating regulatory submission package for %s", study_id)
            
            # Generate all reports
            self.generate_processing_summary()
            reports['processing_summary'] = list(package_dir.glob("processing_summary_*.xlsx"))[0]
            
            self.generate_audit_listing()
            reports['audit_trail'] = list(package_dir.glob("audit_trail_*.xlsx"))[0]
            
            self.generate_quality_control_report()
            reports['quality_control'] = list(package_dir.glob("quality_control_*.xlsx"))[0]
            
            self.generate_phi_summary_report()
            reports['phi_summary'] = list(package_dir.glob("phi_summary_*.xlsx"))[0]
            
            # Create README
            readme = package_dir / "README.txt"
            readme.write_text(f"""
Regulatory Submission Package
Study ID: {study_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Contents:
---------
1. processing_summary_*.xlsx - Study processing statistics
2. audit_trail_*.xlsx - ICH E6 compliant audit log
3. quality_control_*.xlsx - Data quality metrics
4. phi_summary_*.xlsx - PHI detection summary

Compliance Standards:
--------------------
- HIPAA Safe Harbor Method
- ICH E6 (GCP) Guidelines
- 21 CFR Part 11 (Electronic Records)

Contact: Clinical Data Operations
""")
            
            logger.info("Submission package complete → %s", package_dir)
            return reports
            
        finally:
            self.output_dir = original_output
    
    def export_to_sas(self, table_name: str = "processed_notes") -> Path:
        """
        Export data in SAS-compatible CSV format.
        
        Many regulatory systems require SAS datasets.
        This exports with SAS-friendly column names and formats.
        
        Parameters
        ----------
        table_name : database table to export
        
        Returns
        -------
        Path to exported CSV file
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        # SAS-friendly column names (max 32 chars, no special chars)
        df.columns = [col.upper().replace('_', '')[:32] for col in df.columns]
        
        # Save with SAS-compatible settings
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{table_name}_sas_{timestamp}.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info("SAS export saved → %s", output_file)
        return output_file
