"""Backward-compatible import for the refactored diagnosis service."""
from app.diagnosis.service import DifferentialDiagnosis, print_diagnosis_report

__all__ = ["DifferentialDiagnosis", "print_diagnosis_report"]
