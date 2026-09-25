"""Load and clean raw datasets for each disease model."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config import settings


# ---------------------------------------------------------------------------
# Malaria
# ---------------------------------------------------------------------------

MALARIA_DROP_COLS = [
    "IP_Number",
    "DOA",
    "Discharge_Date",
    "Residence_Area",      # Nigeria deployment — irrelevant
    "Primary_Code",        # leakage
    "Diagnosis_Type",      # leakage
    "Risk_Score",          # leakage
]

MALARIA_BINARY = [
    "Fever", "Headache", "Abdominal_Pain", "General_Body_Malaise",
    "Dizziness", "Vomiting", "Confusion", "Backache",
    "Chest_Pain", "Coughing", "Joint_Pain",
]


def load_malaria_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load raw malaria data and drop leakage/administrative columns."""
    path = Path(path or settings.malaria_data)
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in MALARIA_DROP_COLS if c in df.columns])
    return df


def clean_malaria_data(path: Path | str | None = None) -> pd.DataFrame:
    """Return cleaned malaria data ready for training."""
    df = load_malaria_data(path)

    # Standardize target
    df = df.rename(columns={"Target": "target"})

    # Normalize sex values
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype(str).str.strip().str.title()
        df["Sex"] = df["Sex"].replace({"M": "Male", "F": "Female"})

    # Ensure binary columns are 0/1 integers
    for col in MALARIA_BINARY:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int).clip(0, 1)

    # Ensure Age is numeric
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# COVID-19
# ---------------------------------------------------------------------------
NUMERIC_SENTINEL = -999.0
# ---------------------------------------------------------------------------
# COVID-19
# ---------------------------------------------------------------------------

COVID_SYMPTOMS = [
    "cough", "fever", "sob", "diarrhea", "fatigue", "headache",
    "loss_of_smell", "loss_of_taste", "runny_nose",
    "muscle_sore", "sore_throat",
]

COVID_VITALS = ["temperature", "pulse", "sys", "dia", "rr", "sats"]

COVID_COMORBIDITIES = [
    "diabetes", "chd", "htn", "cancer", "asthma",
    "copd", "autoimmune_dis", "smoker",
]

COVID_OBSERVATIONS = [
    "labored_respiration", "rhonchi", "wheezes", "ctab",
    "high_risk_exposure_occupation", "high_risk_interactions",
]

COVID_SEVERITY = ["cough_severity", "sob_severity"]
COVID_NUMERIC_EXTRA = ["days_since_symptom_onset"]

COVID_BINARY = COVID_SYMPTOMS + COVID_COMORBIDITIES + COVID_OBSERVATIONS

COVID_TARGET_COL = "covid19_test_results"

NUMERIC_SENTINEL = -999.0


def load_covid_data(path=None):
    path = Path(path or settings.covid_data)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _to_binary(series: pd.Series) -> pd.Series:
    """Convert mixed-type column to strict 0/1 integers.

    Handles: bool, int, float (1.0/0.0), strings ("Yes"/"No"/"true"/"false"),
    and NaN. Anything unrecognized becomes 0.
    """
    def convert(value):
        # Missing → 0
        if pd.isna(value):
            return 0

        # Native Python bool
        if isinstance(value, bool):
            return int(value)

        # Numeric (covers int, float, and numpy scalars)
        try:
            numeric = float(value)
            return 1 if numeric >= 0.5 else 0
        except (TypeError, ValueError):
            pass

        # String
        text = str(value).strip().lower()
        if text in {"true", "yes", "y", "positive", "1"}:
            return 1
        if text in {"false", "no", "n", "negative", "0"}:
            return 0
        return 0

    return series.map(convert).astype(int)


def _to_severity(series: pd.Series) -> pd.Series:
    """Convert severity column to ordinal integer (0-3), or -1 for unknown."""
    severity_map = {
        "none": 0,
        "mild": 1,
        "moderate": 2,
        "severe": 3,
    }

    def convert(value):
        if pd.isna(value):
            return -1

        # Numeric already
        try:
            numeric = float(value)
            if numeric in (0, 1, 2, 3):
                return int(numeric)
            return -1
        except (TypeError, ValueError):
            pass

        return severity_map.get(str(value).strip().lower(), -1)

    return series.map(convert).astype(int)


def clean_covid_data(path=None):
    """Clean COVID data for training."""
    df = load_covid_data(path).copy()

    # ----- Target -----
    if COVID_TARGET_COL not in df.columns:
        raise ValueError(f"COVID dataset must contain '{COVID_TARGET_COL}'")

    df["target"] = (
        df[COVID_TARGET_COL]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"positive": 1, "negative": 0})
    )
    df = df.drop(columns=[COVID_TARGET_COL])

    # ----- Drop administrative + leakage columns -----
    drop_cols = [
        "source_file", "batch_date", "test_name", "swab_type",
        "rapid_flu_results", "rapid_strep_results",
        "cxr_findings", "cxr_impression", "cxr_label", "cxr_link",
        "er_referral",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ----- Keep only the confirmed feature set -----
    keep_cols = (
        ["age"]
        + COVID_SYMPTOMS
        + COVID_VITALS
        + COVID_COMORBIDITIES
        + COVID_OBSERVATIONS
        + COVID_SEVERITY
        + COVID_NUMERIC_EXTRA
        + ["target"]
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # ----- Binary conversion -----
    for col in COVID_BINARY:
        if col in df.columns:
            df[col] = _to_binary(df[col])

    # ----- Severity conversion -----
    for col in COVID_SEVERITY:
        if col in df.columns:
            df[col] = _to_severity(df[col])

    # Severity refinement: absent symptom → severity 0
    if "cough" in df.columns and "cough_severity" in df.columns:
        df.loc[df["cough"] == 0, "cough_severity"] = 0
    if "sob" in df.columns and "sob_severity" in df.columns:
        df.loc[df["sob"] == 0, "sob_severity"] = 0

    # ----- Numeric columns -----
    for col in ["age"] + COVID_VITALS + COVID_NUMERIC_EXTRA:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(NUMERIC_SENTINEL)

    return df


# ---------------------------------------------------------------------------
# Dengue
# ---------------------------------------------------------------------------

DENGUE_BINARY = ["Fever", "Headache", "Muscle_Pain", "Rash", "Vomiting"]
DENGUE_NUMERIC = ["Age", "Duration_of_Fever", "Platelet Count", "WBC"]


def load_dengue_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load raw dengue dataset."""
    path = Path(path or settings.dengue_data)
    return pd.read_csv(path)


def clean_dengue_data(path: Path | str | None = None) -> pd.DataFrame:
    """Return cleaned dengue data with only clinician-relevant features."""
    df = load_dengue_data(path).copy()

    # Drop identifier and location
    for col in ["Id", "Location"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # Target
    if "Outcome" in df.columns:
        df["target"] = (
            df["Outcome"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"positive": 1, "negative": 0})
        )
        df = df.drop(columns=["Outcome"])

    # Binary columns
    for col in DENGUE_BINARY:
        if col in df.columns:
            df[col] = _to_binary(df[col])

    # Numeric columns
    for col in DENGUE_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Gender normalization
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
        df["Gender"] = df["Gender"].replace({"M": "Male", "F": "Female"})

    # Column name cleanup: "Platelet Count" → "Platelet_Count"
    df = df.rename(columns={"Platelet Count": "Platelet_Count"})

    return df


# ---------------------------------------------------------------------------
# Persist processed datasets
# ---------------------------------------------------------------------------

def persist_processed() -> dict[str, Path]:
    """Clean all three datasets and save to data_processed/."""
    settings.data_processed_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "malaria": settings.malaria_clean,
        "covid": settings.covid_clean,
        "dengue": settings.dengue_clean,
    }

    clean_malaria_data().to_csv(outputs["malaria"], index=False)
    clean_covid_data().to_csv(outputs["covid"], index=False)
    clean_dengue_data().to_csv(outputs["dengue"], index=False)

    for name, path in outputs.items():
        print(f"✅ {name} → {path}")

    return outputs


if __name__ == "__main__":
    persist_processed()