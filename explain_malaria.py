"""Standalone SHAP explanation for a sample Malaria prediction."""
from __future__ import annotations

from app.diagnosis.service import DifferentialDiagnosis


def explain_patient(patient_data: dict) -> dict:
    """Run the same explanation path used by the production diagnosis service."""
    system = DifferentialDiagnosis()
    result = system.diagnose(patient_data)
    return result["malaria_explanation"]


if __name__ == "__main__":
    system = DifferentialDiagnosis()
    encoders = system.malaria_label_encoders
    sex_encoder = encoders.get("Sex")
    residence_encoder = encoders.get("Residence_Area")
    patient = {
        "Age": 45,
        "Sex": sex_encoder.transform(["Female"])[0] if sex_encoder else "Female",
        "Residence_Area": residence_encoder.transform(["Mangalore"])[0] if residence_encoder else "Mangalore",
        "Fever": 1, "Headache": 1, "Abdominal_Pain": 1,
        "General_Body_Malaise": 1, "Dizziness": 1, "Vomiting": 1,
        "Confusion": 0, "Backache": 0, "Chest_Pain": 0,
        "Coughing": 0, "Joint_Pain": 0,
    }
    print(explain_patient(patient))
