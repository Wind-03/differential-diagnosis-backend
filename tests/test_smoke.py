from app.nlp.extractor import extract_age, extract_sex, extract_symptoms, text_to_features


def test_age_extraction():
    assert extract_age("45-year-old patient with fever") == 45


def test_sex_extraction():
    assert extract_sex("45-year-old female") == "Female"


def test_symptom_extraction_and_negation():
    symptoms = extract_symptoms("fever and cough, no vomiting")
    assert symptoms["fever"] == 1
    assert symptoms["cough"] == 1
    assert symptoms["vomiting"] == 0


def test_text_feature_contract():
    features = text_to_features("35-year-old male with fever and headache")
    assert features["Age"] == 35
    assert features["Sex"] == "Male"
    assert features["Fever"] == 1
    assert features["Headache"] == 1
