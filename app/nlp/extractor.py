"""Conservative extraction of explicitly stated patient features."""
from __future__ import annotations

import re

SYMPTOM_KEYWORDS = {
    "fever": ["fever", "high temperature", "pyrexia"],
    "headache": ["headache", "head pain", "migraine", "severe headache"],
    "cough": ["cough", "coughing", "dry cough", "wet cough"],
    "fatigue": ["fatigue", "tired", "exhausted", "weak", "lethargy"],
    "malaise": ["general body malaise", "body malaise", "malaise", "generally unwell", "feeling unwell"],
    "shortness_of_breath": ["shortness of breath", "dyspnea", "breathing difficulty", "hard to breathe"],
    "loss_of_smell": ["loss of smell", "anosmia", "can't smell", "smell gone"],
    "loss_of_taste": ["loss of taste", "ageusia", "can't taste", "taste gone"],
    "sore_throat": ["sore throat", "throat pain"],
    "chest_pain": ["chest pain", "chest discomfort"],
    "vomiting": ["vomiting", "throwing up", "nausea", "vomit"],
    "diarrhea": ["diarrhea", "loose stool"],
    "chills": ["chills", "shivering", "rigors", "chill"],
    "myalgia": ["myalgia", "muscle pain", "body pain", "muscle ache"],
    "arthralgia": ["arthralgia", "joint pain", "joint ache"],
    "abdominal_pain": ["abdominal pain", "stomach pain", "belly pain"],
    "confusion": ["confusion", "disoriented", "confused"],
    "dizziness": ["dizziness", "dizzy", "light-headed", "lightheaded", "vertigo"],
}

NEGATION_CUES = [
    "no", "not", "denies", "denied", "without", "absence of", "negative for",
    "ruled out", "no history of", "no signs of",
]


def is_negated(text: str, match_start: int, window_words: int = 5) -> bool:
    preceding_words = text[:match_start].split()[-window_words:]
    snippet = " ".join(preceding_words)
    return any(cue in snippet for cue in NEGATION_CUES)


def extract_symptoms(text: str) -> dict[str, int]:
    text = text.lower()
    symptoms: dict[str, int] = {}
    for symptom, keywords in SYMPTOM_KEYWORDS.items():
        found = False
        for keyword in keywords:
            for match in re.finditer(re.escape(keyword), text):
                if not is_negated(text, match.start()):
                    found = True
                    break
            if found:
                break
        symptoms[symptom] = int(found)
    return symptoms


def extract_age(text: str) -> int | None:
    match = re.search(r"(\d+)\s*-?\s*(?:years?|yrs?|y/o|yo)\b", text.lower())
    if match:
        return min(max(int(match.group(1)), 0), 120)
    return None


def extract_sex(text: str) -> str | None:
    text = f" {text.lower()} "
    if any(term in text for term in ("female", "woman", "girl", " she ", " her ")):
        return "Female"
    if any(term in text for term in ("male", "man", "boy", " he ", " him ")):
        return "Male"
    return None


def text_to_features(text: str) -> dict[str, object]:
    """Return only features explicitly extractable from the supplied text.

    Missing age, sex, vitals, contact history and comorbidities are intentionally
    omitted. The diagnosis service may use training-set medians/encoders for
    model input, but those values are not presented as patient facts.
    """
    symptoms = extract_symptoms(text)
    sex = extract_sex(text)
    age = extract_age(text)

    features: dict[str, object] = {
        "Fever": symptoms.get("fever", 0),
        "Headache": symptoms.get("headache", 0),
        "Joint_Pain": int(bool(symptoms.get("arthralgia", 0) or symptoms.get("myalgia", 0))),
        "Vomiting": symptoms.get("vomiting", 0),
        "General_Body_Malaise": symptoms.get("malaise", 0),
        "Abdominal_Pain": symptoms.get("abdominal_pain", 0),
        "Dizziness": symptoms.get("dizziness", 0),
        "Confusion": symptoms.get("confusion", 0),
        "Coughing": symptoms.get("cough", 0),
        "Chest_Pain": symptoms.get("chest_pain", 0),
        "Backache": 0,
        "fever": symptoms.get("fever", 0),
        "dry_cough": symptoms.get("cough", 0),
        "sore_throat": symptoms.get("sore_throat", 0),
        "fatigue": symptoms.get("fatigue", 0),
        "headache": symptoms.get("headache", 0),
        "shortness_of_breath": symptoms.get("shortness_of_breath", 0),
        "loss_of_smell": symptoms.get("loss_of_smell", 0),
        "loss_of_taste": symptoms.get("loss_of_taste", 0),
        "chest_pain": symptoms.get("chest_pain", 0),
    }
    if age is not None:
        features["Age"] = age
    if sex is not None:
        features["Sex"] = sex
        features["gender"] = sex

    return features
