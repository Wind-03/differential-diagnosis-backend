"""Train and load the XGBoost Malaria model."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.config import settings
from src.data_loader import load_malaria_data


def train_malaria_model(data_path=None, models_dir: str | Path | None = None):
    models_dir = Path(models_dir or settings.models_dir)
    df = load_malaria_data(data_path)
    drop_cols = ["IP_Number", "DOA", "Discharge_Date", "Primary_Code", "Diagnosis_Type", "Risk_Score"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X, y = df.drop(columns="Target"), df["Target"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = X[col].fillna("Unknown")
        X[col] = encoder.fit_transform(X[col])
        encoders[col] = encoder

    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    medians = X_train.median(numeric_only=True).to_dict()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = CalibratedClassifierCV(
        xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss",
        ), method="sigmoid", cv=3,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print(f"✅ Malaria accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "malaria_model.pkl")
    joblib.dump(scaler, models_dir / "malaria_scaler.pkl")
    joblib.dump(X.columns.tolist(), models_dir / "malaria_features.pkl")
    joblib.dump(medians, models_dir / "malaria_medians.pkl")
    joblib.dump(encoders, models_dir / "malaria_label_encoders.pkl")
    print(f"✅ Malaria artifacts saved to {models_dir}")
    return model


def load_malaria_model(models_dir: str | Path | None = None):
    models_dir = Path(models_dir or settings.models_dir)
    try:
        model = joblib.load(models_dir / "malaria_model.pkl")
        scaler = joblib.load(models_dir / "malaria_scaler.pkl")
        features = joblib.load(models_dir / "malaria_features.pkl")
        print("✅ Malaria model loaded successfully!")
        return model, scaler, features
    except FileNotFoundError:
        print(f"❌ Malaria model artifacts not found in {models_dir}")
        return None, None, None


if __name__ == "__main__":
    train_malaria_model()
