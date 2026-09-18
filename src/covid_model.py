"""Train and load the XGBoost COVID-19 model."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from app.config import settings
from src.data_loader import load_covid_data


def train_covid_model(data_path=None, models_dir: str | Path | None = None):
    models_dir = Path(models_dir or settings.models_dir)
    df = load_covid_data(data_path).copy()

    if "comorbidity" in df.columns:
        df["comorbidity"] = df["comorbidity"].fillna("None")

    encoders = {}
    if "gender" in df.columns:
        gender_encoder = LabelEncoder()
        df["gender_encoded"] = gender_encoder.fit_transform(df["gender"])
        encoders["gender"] = gender_encoder
        df = df.drop(columns="gender")

    if "comorbidity" in df.columns:
        dummies = pd.get_dummies(df["comorbidity"], prefix="comorbidity", dtype=int)
        df = pd.concat([df.drop(columns="comorbidity"), dummies], axis=1)

    if "patient_id" in df.columns:
        df = df.drop(columns="patient_id")

    X = df.drop(columns="covid_result")
    y = df["covid_result"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    medians = X_train.median(numeric_only=True).to_dict()
    for col in X_train.columns:
        if X_train[col].isna().any():
            X_train[col] = X_train[col].fillna(medians.get(col, 0))
            X_test[col] = X_test[col].fillna(medians.get(col, 0))

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
    print(f"✅ COVID accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Keep the original diagnostic baselines as training-time diagnostics.
    tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train_scaled, y_train)
    logreg = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
    print(f"Diagnostic shallow-tree accuracy: {tree.score(X_test_scaled, y_test):.4f}")
    print(f"Diagnostic logistic-regression accuracy: {logreg.score(X_test_scaled, y_test):.4f}")

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "covid_model.pkl")
    joblib.dump(scaler, models_dir / "covid_scaler.pkl")
    joblib.dump(X.columns.tolist(), models_dir / "covid_features.pkl")
    joblib.dump(medians, models_dir / "covid_medians.pkl")
    joblib.dump(encoders, models_dir / "covid_label_encoders.pkl")
    print(f"✅ COVID artifacts saved to {models_dir}")
    return model


def load_covid_model(models_dir: str | Path | None = None):
    models_dir = Path(models_dir or settings.models_dir)
    try:
        model = joblib.load(models_dir / "covid_model.pkl")
        scaler = joblib.load(models_dir / "covid_scaler.pkl")
        features = joblib.load(models_dir / "covid_features.pkl")
        print("✅ COVID model loaded successfully!")
        return model, scaler, features
    except FileNotFoundError:
        print(f"❌ COVID model artifacts not found in {models_dir}")
        return None, None, None


if __name__ == "__main__":
    train_covid_model()
