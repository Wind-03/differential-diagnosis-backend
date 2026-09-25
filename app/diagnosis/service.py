# """Core differential-diagnosis service."""
# from __future__ import annotations

# from pathlib import Path
# from typing import Any

# import joblib
# import pandas as pd

# from app.config import settings
# from app.diagnosis.explainer import build_tree_explainer, explain_prediction
# from src.covid_model import load_covid_model
# from src.malaria_model import load_malaria_model


# class DifferentialDiagnosis:
#     """Load model artifacts once and expose structured diagnosis operations."""

#     def __init__(self, paths=settings):
#         self.paths = paths
#         self.malaria_model, self.malaria_scaler, self.malaria_features = load_malaria_model(paths.models_dir)
#         self.covid_model, self.covid_scaler, self.covid_features = load_covid_model(paths.models_dir)
#         self.malaria_medians = self._load_optional(paths.malaria_medians, {})
#         self.malaria_label_encoders = self._load_optional(paths.malaria_label_encoders, {})
#         self.covid_medians = self._load_optional(paths.covid_medians, {})
#         self.covid_label_encoders = self._load_optional(paths.covid_label_encoders, {})

#         self.malaria_explainer = build_tree_explainer(self.malaria_model) if self.malaria_model is not None else None
#         self.covid_explainer = build_tree_explainer(self.covid_model) if self.covid_model is not None else None

#     @staticmethod
#     def _load_optional(path: Path, default: Any) -> Any:
#         try:
#             return joblib.load(path)
#         except FileNotFoundError:
#             return default

#     @staticmethod
#     def _encode_categoricals(values: dict[str, Any], encoders: dict[str, Any]) -> dict[str, Any]:
#         values = dict(values)
#         for column, encoder in encoders.items():
#             value = values.get(column)
#             if isinstance(value, str):
#                 try:
#                     values[column] = encoder.transform([value])[0]
#                 except ValueError:
#                     values[column] = 0
#         return values

#     def _prepare_input(self, patient_data: dict[str, Any], features: list[str], medians: dict[str, Any], encoders=None):
#         prepared = {feature: patient_data.get(feature, medians.get(feature, 0)) for feature in features}
#         if encoders:
#             # COVID training stores gender as the numeric ``gender_encoded``
#             # feature. Accept the human-readable ``gender`` input and apply
#             # the exact encoder learned during training.
#             if "gender_encoded" in features and isinstance(patient_data.get("gender"), str) and "gender" in encoders:
#                 try:
#                     prepared["gender_encoded"] = encoders["gender"].transform([patient_data["gender"]])[0]
#                 except ValueError:
#                     prepared["gender_encoded"] = medians.get("gender_encoded", 0)
#             prepared = self._encode_categoricals(prepared, encoders)
#         return prepared

#     @staticmethod
#     def _prediction_probability(model: Any, scaled_row):
#         prediction = int(model.predict(scaled_row)[0])
#         probabilities = model.predict_proba(scaled_row)[0]
#         return prediction, float(probabilities[1])

#     def diagnose(self, patient_data: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
#         if self.malaria_model is None or self.covid_model is None:
#             raise RuntimeError("Both Malaria and COVID-19 model artifacts must be available.")

#         # Backward compatibility for the older NLP output: map raw gender/sex into
#         # the exact feature names expected by the saved COVID model.
#         patient_data = dict(patient_data)
#         if "gender_encoded" not in patient_data and "gender" in patient_data:
#             patient_data["gender_encoded"] = patient_data["gender"]
#         if "gender_encoded" not in patient_data and "gender" in patient_data and not self.covid_label_encoders:
#             patient_data["gender_encoded"] = {"Female": 0, "Male": 1}.get(patient_data["gender"], 0)

#         malaria_input = self._prepare_input(
#             patient_data, self.malaria_features, self.malaria_medians, self.malaria_label_encoders
#         )
#         malaria_df = pd.DataFrame([malaria_input], columns=self.malaria_features)
#         malaria_scaled = self.malaria_scaler.transform(malaria_df)
#         malaria_pred, malaria_prob = self._prediction_probability(self.malaria_model, malaria_scaled)

#         covid_input = self._prepare_input(
#             patient_data, self.covid_features, self.covid_medians, self.covid_label_encoders
#         )
#         covid_df = pd.DataFrame([covid_input], columns=self.covid_features)
#         covid_scaled = self.covid_scaler.transform(covid_df)
#         covid_pred, covid_prob = self._prediction_probability(self.covid_model, covid_scaled)

#         malaria_explanation = explain_prediction(
#             self.malaria_explainer, self.malaria_scaler, self.malaria_features, malaria_input
#         )
#         covid_explanation = explain_prediction(
#             self.covid_explainer, self.covid_scaler, self.covid_features, covid_input
#         )

#         threshold = self.paths.diagnosis_threshold
#         if malaria_prob > threshold and covid_prob > threshold:
#             if malaria_prob > covid_prob:
#                 diagnosis, primary = "Malaria (with possible COVID-19 co-infection)", "Malaria"
#             else:
#                 diagnosis, primary = "COVID-19 (with possible Malaria co-infection)", "COVID-19"
#             confidence = max(malaria_prob, covid_prob)
#         elif malaria_prob > threshold:
#             diagnosis, primary, confidence = "Malaria", "Malaria", malaria_prob
#         elif covid_prob > threshold:
#             diagnosis, primary, confidence = "COVID-19", "COVID-19", covid_prob
#         else:
#             diagnosis, primary, confidence = "Neither Malaria nor COVID-19", "Unclear", max(malaria_prob, covid_prob)

#         result = {
#             "primary_diagnosis": primary,
#             "full_diagnosis": diagnosis,
#             "malaria_probability": malaria_prob,
#             "covid_probability": covid_prob,
#             "confidence": confidence,
#             "malaria_model_prediction": "Positive" if malaria_pred == 1 else "Negative",
#             "covid_model_prediction": "Positive" if covid_pred == 1 else "Negative",
#             "recommendations": self._recommendations(primary),
#             "malaria_explanation": malaria_explanation,
#             "covid_explanation": covid_explanation,
#             "covid_model_status": "provisional — pending validated dataset",
#         }
#         return result

#     @staticmethod
#     def _recommendations(primary: str) -> list[str]:
#         if primary == "Malaria":
#             return [
#                 "🔬 Order: Blood smear microscopy or Rapid Diagnostic Test (RDT)",
#                 "💊 Consider: Antimalarial treatment",
#                 "📊 Monitor: Platelet count, hemoglobin, blood glucose",
#             ]
#         if primary == "COVID-19":
#             return [
#                 "🔬 Order: RT-PCR or Rapid Antigen Test for COVID-19",
#                 "💊 Consider: Supportive care, oxygen therapy if needed",
#                 "📊 Monitor: Oxygen saturation, respiratory rate",
#             ]
#         return [
#             "🔬 Order: Both Malaria RDT AND COVID-19 RT-PCR",
#             "🔄 Re-evaluate: If symptoms persist",
#         ]



# def print_diagnosis_report(result: dict[str, Any]) -> None:
#     print("\n" + "=" * 60)
#     print(f"DIAGNOSIS: {result['full_diagnosis']}")
#     print(f"Confidence: {result['confidence']:.1%}")
#     print("=" * 60)
#     print(f"\nMalaria probability: {result['malaria_probability']:.1%}")
#     print(f"COVID-19 probability: {result['covid_probability']:.1%}")
#     print("\nRecommendations:")
#     for recommendation in result["recommendations"]:
#         print(f"  {recommendation}")

"""Core differential-diagnosis service (Malaria-only)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import settings
from app.diagnosis.explainer import build_tree_explainer, explain_prediction
from src.malaria_model import load_malaria_model

# These match the sentinels src/malaria_model.py's missingness augmentation
# used during training (NUMERIC_SENTINEL / BINARY_SENTINEL in
# src/data_loader.py). The model was specifically trained to recognize these
# values as "unreported", not as genuine low/typical measurements — so a
# missing field must be filled with the sentinel, not a dataset median,
# or the model reads a fabricated value instead of an absence of information.
NUMERIC_MISSING_SENTINEL = -999
BINARY_MISSING_SENTINEL = -1
NUMERIC_FEATURES = {"Age"}  # every other malaria feature is binary/encoded


class DifferentialDiagnosis:
    """Load model artifacts once and expose structured diagnosis operations.

    Only the Malaria pathway is loaded and reported. The COVID-19 model is
    intentionally excluded here because its training data has not been
    validated as clinically reliable (see project documentation) — surfacing
    its output alongside a trustworthy model risks it being read as
    equally trustworthy.
    """

    def __init__(self, paths=settings):
        self.paths = paths
        self.malaria_model, self.malaria_scaler, self.malaria_features = load_malaria_model(paths.models_dir)
        self.malaria_medians = self._load_optional(paths.malaria_medians, {})
        self.malaria_label_encoders = self._load_optional(paths.malaria_label_encoders, {})

        self.malaria_explainer = build_tree_explainer(self.malaria_model) if self.malaria_model is not None else None

    @staticmethod
    def _load_optional(path: Path, default: Any) -> Any:
        try:
            return joblib.load(path)
        except FileNotFoundError:
            return default

    @staticmethod
    def _encode_categoricals(values: dict[str, Any], encoders: dict[str, Any]) -> dict[str, Any]:
        values = dict(values)
        for column, encoder in encoders.items():
            value = values.get(column)
            if isinstance(value, str):
                try:
                    values[column] = encoder.transform([value])[0]
                except ValueError:
                    values[column] = 0
        return values

    def _prepare_input(self, patient_data: dict[str, Any], features: list[str], medians: dict[str, Any], encoders=None):
        prepared = {}
        for feature in features:
            if feature in patient_data:
                prepared[feature] = patient_data[feature]
            elif feature in NUMERIC_FEATURES:
                prepared[feature] = NUMERIC_MISSING_SENTINEL
            else:
                prepared[feature] = BINARY_MISSING_SENTINEL
        if encoders:
            prepared = self._encode_categoricals(prepared, encoders)
        return prepared

    @staticmethod
    def _prediction_probability(model: Any, scaled_row):
        prediction = int(model.predict(scaled_row)[0])
        probabilities = model.predict_proba(scaled_row)[0]
        return prediction, float(probabilities[1])

    def diagnose(self, patient_data: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
        if self.malaria_model is None:
            raise RuntimeError("The Malaria model artifact must be available.")

        # Track which model features were actually supplied by the caller,
        # as opposed to ones that will be silently filled with a training-set
        # median in _prepare_input below. Without this, a defaulted value
        # (e.g. a median Age) can show up in top_factors looking identical
        # to a value the patient actually reported.
        reported_features = {feature for feature in self.malaria_features if feature in patient_data}

        malaria_input = self._prepare_input(
            patient_data, self.malaria_features, self.malaria_medians, self.malaria_label_encoders
        )
        malaria_df = pd.DataFrame([malaria_input], columns=self.malaria_features)
        malaria_scaled = self.malaria_scaler.transform(malaria_df)
        malaria_pred, malaria_prob = self._prediction_probability(self.malaria_model, malaria_scaled)

        malaria_explanation = explain_prediction(
            self.malaria_explainer, self.malaria_scaler, self.malaria_features, malaria_input
        )
        for factor in malaria_explanation["top_factors"]:
            factor["source"] = "reported" if factor["feature"] in reported_features else "assumed_default"

        threshold = self.paths.diagnosis_threshold
        if malaria_prob > threshold:
            diagnosis, primary, confidence = "Malaria", "Malaria", malaria_prob
        else:
            diagnosis, primary, confidence = "No Malaria indicated", "Unclear", 1 - malaria_prob

        result = {
            "primary_diagnosis": primary,
            "full_diagnosis": diagnosis,
            "malaria_probability": malaria_prob,
            "confidence": confidence,
            "malaria_model_prediction": "Positive" if malaria_pred == 1 else "Negative",
            "recommendations": self._recommendations(primary),
            "malaria_explanation": malaria_explanation,
        }
        return result

    @staticmethod
    def _recommendations(primary: str) -> list[str]:
        if primary == "Malaria":
            return [
                "🔬 Order: Blood smear microscopy or Rapid Diagnostic Test (RDT)",
                "💊 Consider: Antimalarial treatment",
                "📊 Monitor: Platelet count, hemoglobin, blood glucose",
            ]
        return [
            "🔬 Order: Malaria RDT if symptoms persist or worsen",
            "🔄 Re-evaluate: If symptoms persist",
        ]


def print_diagnosis_report(result: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"DIAGNOSIS: {result['full_diagnosis']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print("=" * 60)
    print(f"\nMalaria probability: {result['malaria_probability']:.1%}")
    print("\nRecommendations:")
    for recommendation in result["recommendations"]:
        print(f"  {recommendation}")