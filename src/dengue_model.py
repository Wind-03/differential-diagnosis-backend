"""Train and load the Dengue model (XGBoost, with stacking fallback)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from xgboost import XGBClassifier

from app.config import settings
from src.data_loader import clean_dengue_data
from src.reporting import (
    build_report,
    compute_metrics,
    save_disease_report,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DISEASE = "dengue"
TARGET = "target"
RANDOM_STATE = 42

# Feature sets
DENGUE_BINARY = ["Fever", "Headache", "Muscle_Pain", "Rash", "Vomiting"]
DENGUE_NUMERIC = ["Age", "Duration_of_Fever", "Platelet_Count", "WBC"]
DENGUE_CATEGORICAL = ["Gender"]

DENGUE_FEATURES = DENGUE_BINARY + DENGUE_NUMERIC + DENGUE_CATEGORICAL


# ---------------------------------------------------------------------------
# Missingness augmentation
# ---------------------------------------------------------------------------

def _augment_missingness(
    X: pd.DataFrame,
    binary_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    rate_min: float = 0.05,
    rate_max: float = 0.25,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X_aug = X.copy()
    n_rows, n_cols = X_aug.shape

    rates = rng.uniform(rate_min, rate_max, size=n_rows)
    n_masks = np.round(rates * n_cols).astype(int)

    for i in range(n_rows):
        if n_masks[i] == 0:
            continue
        masked_idx = rng.choice(n_cols, size=n_masks[i], replace=False)
        for j in masked_idx:
            col = X_aug.columns[j]
            if col in binary_cols:
                X_aug.iat[i, j] = -1
            elif col in numeric_cols:
                X_aug.iat[i, j] = -999.0
            elif col in categorical_cols:
                X_aug.iat[i, j] = "Unknown"
            else:
                X_aug.iat[i, j] = -999.0

    return X_aug


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _fit_encoders(
    X: pd.DataFrame, categorical_cols: list[str]
) -> dict[str, LabelEncoder]:
    encoders: dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        clean_values = X.loc[X[col].astype(str) != "Unknown", col]
        clean_values = clean_values.fillna("Unknown").astype(str)
        encoder.fit(clean_values)
        encoders[col] = encoder
    return encoders


def _apply_encoders(
    X: pd.DataFrame, encoders: dict[str, LabelEncoder]
) -> pd.DataFrame:
    X = X.copy()
    for col, encoder in encoders.items():
        known = set(encoder.classes_)
        X[col] = X[col].astype(str).map(
            lambda v: encoder.transform([v])[0] if v in known else -1
        )
    return X


def _fit_medians(X: pd.DataFrame, numeric_cols: list[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for col in numeric_cols:
        clean = X.loc[X[col] != -999.0, col]
        medians[col] = float(clean.median()) if not clean.empty else 0.0
    return medians


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dengue_model(
    data_path: Path | str | None = None,
    models_dir: Path | str | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Train dengue model and optionally persist artifacts."""
    t0 = time.time()

    models_dir = Path(models_dir or settings.models_dir) / DISEASE
    models_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Load & clean ----------------
    df = clean_dengue_data(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"Dengue dataset must contain a '{TARGET}' column")

    feature_cols = [c for c in DENGUE_FEATURES if c in df.columns]
    missing = [c for c in DENGUE_FEATURES if c not in df.columns]
    if missing:
        print(f"⚠️  Missing dengue features (will be skipped): {missing}")

    X = df[feature_cols].copy()
    y = df[TARGET].astype(int)

    # ---------------- Train/test split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ---------------- Fit encoders on clean data ----------------
    encoders = _fit_encoders(X_train, DENGUE_CATEGORICAL)
    medians = _fit_medians(X_train, DENGUE_NUMERIC)

    # Apply encoders to train and test
    X_train_enc = _apply_encoders(X_train, encoders)
    X_test_enc = _apply_encoders(X_test, encoders)

    # ---------------- Fit scaler on clean train data ----------------
    scaler = RobustScaler()
    scaler.fit(X_train_enc)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train_enc), columns=feature_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_enc), columns=feature_cols
    )

    # ---------------- Augment training data ----------------
    X_train_aug = _augment_missingness(
        X_train_scaled,
        binary_cols=DENGUE_BINARY,
        numeric_cols=DENGUE_NUMERIC,
        categorical_cols=[],  # already encoded to numeric
        rate_min=settings.missing_rate_min,
        rate_max=settings.missing_rate_max,
    )

    # ---------------- Class weight for XGBoost ----------------
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # ---------------- Base models ----------------
    xgb = CalibratedClassifierCV(
        XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
        ),
        method="sigmoid",
        cv=3,
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    # Third base model: HistGradientBoosting handles NaN natively
    hgb = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )

    stack = StackingClassifier(
        estimators=[
            ("xgb", xgb),
            ("rf", rf),
            ("hgb", hgb),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )

    # ---------------- Train individual models ----------------
    print(f"🔄 Training dengue base models...")
    individual_metrics: dict[str, dict[str, Any]] = {}
    for name, model in [("xgb", xgb), ("rf", rf), ("hgb", hgb)]:
        model.fit(X_train_aug, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        individual_metrics[name] = compute_metrics(y_test, y_pred, y_proba)

    # ---------------- Train ensemble ----------------
    print(f"🔄 Training dengue stacking ensemble...")
    stack.fit(X_train_aug, y_train)
    y_pred_ens = stack.predict(X_test_scaled)
    y_proba_ens = stack.predict_proba(X_test_scaled)[:, 1]
    ensemble_metrics = compute_metrics(y_test, y_pred_ens, y_proba_ens)

    # ---------------- Select winner ----------------
    xgb_f1 = individual_metrics["xgb"]["f1"]
    ensemble_f1 = ensemble_metrics["f1"]
    use_ensemble = ensemble_f1 > xgb_f1

    winner_name = "stack" if use_ensemble else "xgb"
    winner_model = stack if use_ensemble else xgb
    winner_metrics = ensemble_metrics if use_ensemble else individual_metrics["xgb"]

    print(f"\n🏆 Selected model: {winner_name.upper()} (F1={winner_metrics['f1']:.4f})")

    # ---------------- Report ----------------
    dataset_info = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(feature_cols),
        "positive_rate": float(y.mean()),
        "feature_list": feature_cols,
    }
    report = build_report(DISEASE, individual_metrics, ensemble_metrics, dataset_info)

    # ---------------- Persist ----------------
    if save:
        metadata = {
            "disease": DISEASE,
            "selected_model": winner_name,
            "selection_reason": (
                f"{winner_name} achieved higher test-set F1 "
                f"({winner_metrics['f1']:.4f} vs "
                f"{xgb_f1 if use_ensemble else ensemble_f1:.4f})"
            ),
            "feature_cols": feature_cols,
            "binary_cols": DENGUE_BINARY,
            "numeric_cols": DENGUE_NUMERIC,
            "categorical_cols": DENGUE_CATEGORICAL,
            "trained_at": report["timestamp_utc"],
            "n_train": dataset_info["n_train"],
            "n_test": dataset_info["n_test"],
            "random_state": RANDOM_STATE,
            "test_metrics": winner_metrics,
        }

        joblib.dump(winner_model, models_dir / "model.pkl")
        joblib.dump(scaler, models_dir / "scaler.pkl")
        joblib.dump(feature_cols, models_dir / "features.pkl")
        joblib.dump(medians, models_dir / "medians.pkl")
        joblib.dump(encoders, models_dir / "label_encoders.pkl")
        joblib.dump(metadata, models_dir / "metadata.pkl")

        joblib.dump(
            xgb if use_ensemble else stack,
            models_dir / "model_alternative.pkl",
        )

        save_disease_report(DISEASE, report)

        print(f"✅ Dengue artifacts saved → {models_dir}")
        print(f"⏱️  Elapsed: {time.time() - t0:.1f}s")

    return {
        "model": winner_model,
        "model_name": winner_name,
        "scaler": scaler,
        "features": feature_cols,
        "medians": medians,
        "encoders": encoders,
        "metadata": metadata if save else None,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dengue_model(
    models_dir: Path | str | None = None,
) -> tuple[Any, Any, list[str]] | tuple[None, None, None]:
    """Load the saved dengue model, scaler, and feature order."""
    models_dir = Path(models_dir or settings.models_dir) / DISEASE
    try:
        model = joblib.load(models_dir / "model.pkl")
        scaler = joblib.load(models_dir / "scaler.pkl")
        features = joblib.load(models_dir / "features.pkl")
        print("✅ Dengue model loaded successfully!")
        return model, scaler, features
    except FileNotFoundError:
        print(f"❌ Dengue model artifacts not found in {models_dir}")
        return None, None, None


def load_dengue_metadata(models_dir: Path | str | None = None) -> dict[str, Any]:
    models_dir = Path(models_dir or settings.models_dir) / DISEASE
    return joblib.load(models_dir / "metadata.pkl")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = train_dengue_model()
    print("\n" + "=" * 60)
    print("DENGUE REPORT")
    print("=" * 60)
    print(json.dumps(result["report"], indent=2, default=str))