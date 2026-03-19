"""
eda.py
──────
Exploratory Data Analysis module for the clinical notes corpus.

All analysis is encapsulated in ClinicalEDA so:
  - Jupyter notebooks call it with one import
  - Flask /dashboard route calls the same methods for live charts
  - pytest can test outputs programmatically

Output: PNG figures saved to output_dir, plus Plotly HTML for Flask.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for Flask)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# ── Colour palette (consistent across all charts) ─────────────────────────────
PALETTE = {
    "primary":   "#1F4E79",
    "secondary": "#2E86AB",
    "accent":    "#E84855",
    "neutral":   "#6C757D",
    "success":   "#2D6A4F",
    "warning":   "#F4A261",
    "bg":        "#F8F9FA",
}
PHI_ENTITY_COLORS = {
    "PERSON":   "#E63946",
    "DATE":     "#457B9D",
    "LOCATION": "#2A9D8F",
    "HOSPITAL": "#E9C46A",
    "AGE":      "#F4A261",
    "PHONE":    "#9B5DE5",
}


class ClinicalEDA:
    """
    Exploratory Data Analysis for a clinical notes DataFrame.

    Parameters
    ----------
    df         : DataFrame with at least 'transcription' and 'medical_specialty'
    output_dir : where to save PNG/HTML outputs
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = "data/eda_outputs") -> None:
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._precompute()
        logger.info("ClinicalEDA ready | %d notes | output → %s", len(self.df), self.output_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def run_full_eda(self) -> dict:
        """
        Run all EDA steps and return a summary dict.
        Useful for the Flask /dashboard route.
        """
        summary = {}
        summary["basic_stats"]      = self.basic_stats()
        summary["specialty_dist"]   = self.plot_specialty_distribution()
        summary["note_length_dist"] = self.plot_note_length_distribution()
        summary["phi_pattern_freq"] = self.plot_phi_pattern_frequency()
        summary["missing_data"]     = self.plot_missing_data()
        summary["top_words"]        = self.plot_top_clinical_words()
        logger.info("Full EDA complete — %d charts saved", 5)
        return summary

    def basic_stats(self) -> dict:
        """Return a dict of key corpus statistics."""
        stats = {
            "total_notes":        len(self.df),
            "unique_specialties": self.df["medical_specialty"].nunique(),
            "mean_note_length":   round(self.df["note_length"].mean(), 1),
            "median_note_length": round(self.df["note_length"].median(), 1),
            "min_note_length":    int(self.df["note_length"].min()),
            "max_note_length":    int(self.df["note_length"].max()),
            "notes_with_phi":     int(self.df["has_phi"].sum()) if "has_phi" in self.df.columns else "N/A",
            "phi_detection_rate": (
                f"{self.df['has_phi'].mean()*100:.1f}%"
                if "has_phi" in self.df.columns else "N/A"
            ),
        }
        logger.info("Basic stats: %s", stats)
        return stats

    def plot_specialty_distribution(self, top_n: int = 12) -> str:
        """Bar chart: top N medical specialties by note count."""
        counts = (
            self.df["medical_specialty"]
            .value_counts()
            .head(top_n)
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(counts.index, counts.values, color=PALETTE["primary"], alpha=0.85)

        # Value labels on bars
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_width() + counts.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9, color=PALETTE["neutral"]
            )

        ax.set_xlabel("Number of clinical notes", fontsize=11)
        ax.set_title("Clinical Notes by Medical Specialty", fontsize=13, fontweight="bold",
                     color=PALETTE["primary"])
        ax.set_facecolor(PALETTE["bg"])
        fig.patch.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        plt.tight_layout()

        out = self.output_dir / "specialty_distribution.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved → %s", out)
        return str(out)

    def plot_note_length_distribution(self) -> str:
        """Histogram + KDE of note character lengths, split by specialty."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Overall distribution
        ax = axes[0]
        ax.hist(self.df["note_length"], bins=40, color=PALETTE["secondary"],
                alpha=0.7, edgecolor="white")
        ax.axvline(self.df["note_length"].median(), color=PALETTE["accent"],
                   linestyle="--", linewidth=1.5, label=f"Median: {self.df['note_length'].median():.0f}")
        ax.axvline(self.df["note_length"].mean(), color=PALETTE["warning"],
                   linestyle="--", linewidth=1.5, label=f"Mean: {self.df['note_length'].mean():.0f}")
        ax.set_xlabel("Note length (characters)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Distribution of Note Lengths", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_facecolor(PALETTE["bg"])
        ax.spines[["top", "right"]].set_visible(False)

        # Box plots per top specialty
        ax2 = axes[1]
        top_specs = self.df["medical_specialty"].value_counts().head(6).index.tolist()
        subset = self.df[self.df["medical_specialty"].isin(top_specs)]
        order = subset.groupby("medical_specialty")["note_length"].median().sort_values().index

        bp_data = [
            subset[subset["medical_specialty"] == s]["note_length"].values
            for s in order
        ]
        bp = ax2.boxplot(bp_data, vert=False, patch_artist=True,
                         medianprops=dict(color=PALETTE["accent"], linewidth=2),
                         flierprops=dict(marker="o", markersize=3, alpha=0.3))
        colors = sns.color_palette("Blues", len(bp_data))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax2.set_yticklabels([s[:20] for s in order], fontsize=9)
        ax2.set_xlabel("Note length (characters)", fontsize=11)
        ax2.set_title("Note Length by Specialty (Top 6)", fontsize=12, fontweight="bold")
        ax2.set_facecolor(PALETTE["bg"])
        ax2.spines[["top", "right"]].set_visible(False)

        fig.suptitle("Clinical Note Length Analysis", fontsize=14, fontweight="bold",
                     color=PALETTE["primary"], y=1.01)
        fig.patch.set_facecolor("white")
        plt.tight_layout()

        out = self.output_dir / "note_length_distribution.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved → %s", out)
        return str(out)

    def plot_phi_pattern_frequency(self) -> str:
        """
        Bar chart of PHI pattern frequency detected via regex.

        Key decision: regex PHI detection here is INTENTIONALLY simple —
        it's EDA to understand the corpus, not the real NER output.
        The spaCy pipeline in Phase 2 is the authoritative extractor.
        """
        phi_counts = Counter()
        patterns = {
            "PERSON (name-like)":   r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
            "DATE":                 r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            "PHONE":                r"\(?\d{3}\)?[\s\-]\d{3}[\-]\d{4}",
            "AGE (N-year-old)":     r"\b\d{1,3}[- ]year[- ]old\b",
            "MRN":                  r"\bMRN\d+\b",
            "HOSPITAL (keyword)":   r"\b(?:Hospital|Medical Center|Clinic|Health System)\b",
        }

        for label, pat in patterns.items():
            total = sum(
                len(re.findall(pat, str(note), re.IGNORECASE))
                for note in self.df["transcription"]
            )
            phi_counts[label] = total

        counts_df = pd.DataFrame(
            list(phi_counts.items()), columns=["PHI Type", "Count"]
        ).sort_values("Count", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = [PHI_ENTITY_COLORS.get(k.split()[0], PALETTE["secondary"])
                  for k in counts_df["PHI Type"]]
        bars = ax.barh(counts_df["PHI Type"], counts_df["Count"],
                       color=colors, alpha=0.85, edgecolor="white")

        for bar, val in zip(bars, counts_df["Count"]):
            ax.text(bar.get_width() + counts_df["Count"].max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=9, color=PALETTE["neutral"])

        ax.set_xlabel("Number of pattern matches", fontsize=11)
        ax.set_title("PHI Pattern Frequency in Corpus\n(regex scan — EDA only)", fontsize=12,
                     fontweight="bold", color=PALETTE["primary"])
        ax.set_facecolor(PALETTE["bg"])
        fig.patch.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        out = self.output_dir / "phi_pattern_frequency.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved → %s", out)
        return str(out)

    def plot_missing_data(self) -> str:
        """Heatmap showing missing value distribution across columns."""
        fig, ax = plt.subplots(figsize=(8, 4))

        missing = (
            self.df.isnull().sum() / len(self.df) * 100
        ).reset_index()
        missing.columns = ["Column", "Missing %"]
        missing = missing[missing["Missing %"] > 0]

        if missing.empty:
            ax.text(0.5, 0.5, "No missing values detected ✓",
                    ha="center", va="center", fontsize=14,
                    color=PALETTE["success"], transform=ax.transAxes)
            ax.axis("off")
        else:
            bars = ax.barh(missing["Column"], missing["Missing %"],
                           color=PALETTE["accent"], alpha=0.75)
            ax.set_xlabel("Missing (%)", fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)

        ax.set_title("Missing Data Profile", fontsize=12, fontweight="bold",
                     color=PALETTE["primary"])
        ax.set_facecolor(PALETTE["bg"])
        fig.patch.set_facecolor("white")
        plt.tight_layout()

        out = self.output_dir / "missing_data.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved → %s", out)
        return str(out)

    def plot_top_clinical_words(self, top_n: int = 30) -> str:
        """
        Horizontal bar of top clinical terms after removing stopwords.

        Key decision: we use a CLINICAL stopword list, not just NLTK's
        English list — words like 'patient', 'history', 'noted' are high
        frequency but clinically uninformative.  Stripping them exposes
        the actual clinical signal.
        """
        CLINICAL_STOPWORDS = {
            # General
            "the", "a", "an", "and", "or", "of", "to", "in", "is", "was",
            "were", "are", "with", "for", "on", "at", "by", "from", "this",
            "that", "be", "it", "as", "has", "have", "had", "not", "he",
            "she", "his", "her", "they", "their", "no", "will", "been",
            "would", "may", "also", "which", "all", "who", "if", "any",
            "but", "than", "so", "up", "do", "did", "its", "our", "we",
            # Clinical boilerplate
            "patient", "history", "noted", "normal", "within", "findings",
            "reported", "exam", "review", "plan", "follow", "mg", "daily",
            "per", "right", "left", "bilaterally", "consistent", "without",
            "given", "po", "prn", "bid", "tid", "qid",
        }

        all_words = " ".join(self.df["transcription"].dropna().str.lower()).split()
        word_counts = Counter(
            w.strip(".,;:()[]") for w in all_words
            if len(w) > 3 and w not in CLINICAL_STOPWORDS and w.isalpha()
        )

        top = pd.DataFrame(
            word_counts.most_common(top_n), columns=["Word", "Count"]
        ).sort_values("Count")

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.get_cmap("Blues")
        colors = [cmap(0.4 + 0.6 * i / len(top)) for i in range(len(top))]

        ax.barh(top["Word"], top["Count"], color=colors, edgecolor="white")
        ax.set_xlabel("Frequency", fontsize=11)
        ax.set_title(f"Top {top_n} Clinical Terms\n(after clinical stopword removal)",
                     fontsize=12, fontweight="bold", color=PALETTE["primary"])
        ax.set_facecolor(PALETTE["bg"])
        fig.patch.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        out = self.output_dir / "top_clinical_words.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved → %s", out)
        return str(out)

    def print_sample_notes(self, n: int = 3) -> None:
        """Print n sample notes to stdout — useful in Jupyter."""
        for i, row in self.df.sample(n, random_state=42).iterrows():
            print(f"\n{'='*70}")
            print(f"[{row['medical_specialty']}]")
            print(row["transcription"][:600])
            print(f"{'─'*70}")

    # ── Private ───────────────────────────────────────────────────────────────

    def _precompute(self) -> None:
        """Add derived columns used by multiple plot methods."""
        self.df["note_length"]  = self.df["transcription"].str.len()
        self.df["word_count"]   = self.df["transcription"].str.split().str.len()
        self.df["sent_count"]   = self.df["transcription"].str.count(r"[.!?]")
        if "has_phi" not in self.df.columns:
            self.df["has_phi"] = 0
