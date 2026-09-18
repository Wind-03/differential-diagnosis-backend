"""Dataset inspection utility for identifying suspicious target leakage."""
from __future__ import annotations

import joblib
import pandas as pd

from app.config import settings
from src.data_loader import load_covid_data, load_malaria_data


def check_dataset(df: pd.DataFrame, target_col: str, name: str) -> None:
    print("\n" + "=" * 60)
    print(f"🔍 Checking {name} dataset for leakage")
    print("=" * 60)
    print(f"Columns: {list(df.columns)}")
    numeric = df.select_dtypes(include=["number", "bool"])
    if target_col in numeric.columns:
        print(numeric.corr(numeric_only=True)[target_col].sort_values(key=abs, ascending=False))


def model_importances(model_path, features_path, name):
    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        base = model.calibrated_classifiers_[0].estimator if hasattr(model, "calibrated_classifiers_") else model
        importance = pd.Series(base.feature_importances_, index=features).sort_values(ascending=False)
        print(f"\n{name} top features:\n{importance.head(10)}")
    except Exception as exc:
        print(f"Could not load {name} importances: {exc}")


if __name__ == "__main__":
    covid = load_covid_data()
    malaria = load_malaria_data()
    check_dataset(covid, "covid_result", "COVID")
    check_dataset(malaria, "Target", "Malaria")
    model_importances(settings.covid_model, settings.covid_features, "COVID")
    model_importances(settings.malaria_model, settings.malaria_features, "Malaria")
