from app.diagnosis.service import DifferentialDiagnosis, print_diagnosis_report

dd = DifferentialDiagnosis()

# Sample patient: fever, headache, joint pain, rash
patient = {
    "text": "45 year old male with fever for 3 days, severe headache, joint pain, and a rash on his torso",
}

from app.nlp.extractor import text_to_features
features = text_to_features(patient["text"])
print("Extracted features:", features)

result = dd.diagnose(features)
print_diagnosis_report(result)