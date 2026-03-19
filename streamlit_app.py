"""
streamlit_app.py
────────────────
Live demo for ClinicalNER — paste a clinical note, see it de-identified
in real time with entity table, confidence scores, and pipeline metrics.

Run locally:  streamlit run streamlit_app.py
Deploy:       share.streamlit.io → connect GitHub repo → select this file
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from src.pipeline.ner_pipeline import NERPipeline
from src.pipeline.data_cleaner import DataCleaner
from src.pipeline.anomaly_detector import AnomalyDetector

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinicalNER — PHI De-identification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cache pipeline (loads once, reused across reruns) ─────────────────────────
@st.cache_resource
def load_pipeline():
    pipeline = NERPipeline(use_spacy=True)
    cleaner  = DataCleaner(strict_mode=False)
    return pipeline, cleaner

pipeline, cleaner = load_pipeline()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .entity-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.78rem; font-weight: 600; margin: 2px;
  }
  .badge-DATE     { background:#1e3a5f; color:#63b3ed; }
  .badge-DOB      { background:#1e3a5f; color:#90cdf4; }
  .badge-PHONE    { background:#2d3748; color:#68d391; }
  .badge-MRN      { background:#3d2a0e; color:#f6ad55; }
  .badge-HOSPITAL { background:#2a1a3d; color:#b794f4; }
  .badge-AGE      { background:#1a2d2d; color:#76e4f7; }
  .badge-PERSON   { background:#3d1a1a; color:#fc8181; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 ClinicalNER")
st.markdown(
    "**Clinical NLP De-identification Pipeline** · "
    "Built for Associate Clinical Programmer portfolio · "
    "Hybrid regex + spaCy NER"
)
st.divider()

# ── Input / Output columns ────────────────────────────────────────────────────
SAMPLE = """Patient: James Smith, DOB: 04/12/1985
MRN: MRN302145. Admitted to St. Mary's Hospital on 01/15/2024.
Contact: (415) 555-9876. Age: 38. Referred by Dr. Emily Chen.
Diagnosis: Type 2 Diabetes. Follow-up at Memorial Medical Center on 03/01/2024."""

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.subheader("Input — raw clinical note")
    text_input = st.text_area(
        "Paste clinical text here",
        value=SAMPLE,
        height=220,
        help="Any clinical note containing PHI — names, dates, MRNs, phone numbers",
        label_visibility="collapsed",
    )
    run_btn = st.button("🔍  Run de-identification", type="primary", use_container_width=True)

with col2:
    st.subheader("Output — de-identified")
    output_box = st.empty()
    if not run_btn:
        output_box.info("Click **Run de-identification** to process the note.")

# ── Pipeline execution ────────────────────────────────────────────────────────
if run_btn and text_input.strip():
    with st.spinner("Running NER pipeline..."):
        pre    = cleaner.clean_pre_ner(text_input)
        result = pipeline.process_note(pre.cleaned_text, save_to_db=False)
        post   = cleaner.clean_post_ner(result["masked_text"])

    with col2:
        output_box.text_area(
            "Masked text",
            value=post.cleaned_text,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

    st.divider()

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    entities = result.get("entities", [])
    m1.metric("PHI entities found",  result["entity_count"])
    m2.metric("Avg confidence",      f"{result.get('avg_confidence', 0):.2f}")
    m3.metric("Unique labels",       len(result.get("entity_types", {})))
    m4.metric("Pre-clean changes",   len(pre.changes))
    m5.metric("Post-clean valid",    "✅ Yes" if post.is_valid else "⚠️ No")

    # ── Entity table ──────────────────────────────────────────────────────────
    if entities:
        st.subheader("Detected PHI entities")

        df_ents = pd.DataFrame([{
            "Label":      e.get("label",  ""),
            "Text":       e.get("text",   ""),
            "Source":     e.get("source", ""),
            "Confidence": f"{e.get('confidence', 0):.2f}",
            "Span":       f"{e.get('start',0)}–{e.get('end',0)}",
        } for e in entities])

        st.dataframe(df_ents, use_container_width=True, hide_index=True)

        # Entity type badge row
        badge_html = " ".join(
            f'<span class="entity-badge badge-{lbl}">{lbl} × {cnt}</span>'
            for lbl, cnt in sorted(result["entity_types"].items())
        )
        st.markdown(badge_html, unsafe_allow_html=True)

    else:
        st.warning("No PHI entities detected in this text.")

    # ── Cleaning log ──────────────────────────────────────────────────────────
    all_changes = pre.changes + post.changes
    if all_changes:
        with st.expander(f"🔧 Cleaning log ({len(all_changes)} changes applied)"):
            for change_type, desc in all_changes:
                st.markdown(f"- **{change_type}**: {desc}")

    # ── Residual PHI warning ──────────────────────────────────────────────────
    if post.residual_phi:
        st.warning(
            f"⚠️ Possible residual PHI detected: {post.residual_phi}. "
            "Review output before use in production."
        )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About ClinicalNER")
    st.markdown("""
A production-grade PHI de-identification pipeline built as a portfolio
project for clinical data science roles.

**Tech stack**
- Python · spaCy · scikit-learn
- Flask REST API (5 routes)
- SQLite + SQLAlchemy
- Docker · pytest (90% coverage)
- IsolationForest anomaly detection
- GitHub Actions CI

**Supported PHI types**
| Label | Example |
|---|---|
| DATE | 04/12/2022 |
| DOB | DOB: 07/22/1978 |
| PHONE | (415) 555-9876 |
| MRN | MRN302145 |
| HOSPITAL | St. Mary's Hospital |
| AGE | 58-year-old |
| PERSON | James Smith *(spaCy)* |

**Pipeline steps**
1. `DataLoader` — ingest + SQL
2. `ClinicalEDA` — 5 chart types
3. `NERPipeline` — hybrid NER
4. `DataCleaner` — pre + post pass
5. `AuditLogger` — append-only log
6. `AnomalyDetector` — IsolationForest
7. Flask API — 6 REST routes
    """)

    st.divider()
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-ClinicalNER-blue)]"
        "(https://github.com/YOUR_USERNAME/ClinicalNER)"
    )
