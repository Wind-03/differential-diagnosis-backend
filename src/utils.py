"""Backward-compatible NLP imports."""
from app.nlp.extractor import (
    NEGATION_CUES,
    SYMPTOM_KEYWORDS,
    extract_age,
    extract_sex,
    extract_symptoms,
    is_negated,
    text_to_features,
)

__all__ = [
    "NEGATION_CUES", "SYMPTOM_KEYWORDS", "extract_age", "extract_sex",
    "extract_symptoms", "is_negated", "text_to_features",
]
