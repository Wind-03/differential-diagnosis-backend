from differential_diagnosis import DifferentialDiagnosis

# Initialize
dd = DifferentialDiagnosis()

# Get features
features = text_to_features(patient_text)

# Show what's being sent to Malaria model
print("\n" + "="*60)
print("🧪 MALARIA MODEL INPUT")
print("="*60)
malaria_input = {}
for feature in dd.malaria_features:
    value = features.get(feature, 0)
    malaria_input[feature] = value
    print(f"  {feature}: {value}")

# Show what's being sent to COVID model
print("\n" + "="*60)
print("🧪 COVID-19 MODEL INPUT")
print("="*60)
covid_input = {}
for feature in dd.covid_features:
    value = features.get(feature, 0)
    covid_input[feature] = value
    print(f"  {feature}: {value}")

# Run diagnosis
result = dd.diagnose(features)
print("\n" + "="*60)
print("🔬 DIAGNOSIS RESULT")
print("="*60)
print(f"Malaria probability: {result['malaria_probability']:.1%}")
print(f"COVID-19 probability: {result['covid_probability']:.1%}")