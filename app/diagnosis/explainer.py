"""SHAP explanation helpers for tree-based binary classifiers."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
import shap


def sigmoid(value: float) -> float:
    """Convert log-odds to probability with a numerically stable clamp."""
    value = max(min(float(value), 700.0), -700.0)
    return 1.0 / (1.0 + math.exp(-value))


def unwrap_calibrated_model(model: Any) -> Any:
    """Return the underlying tree estimator when a classifier is calibrated."""
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def build_tree_explainer(model: Any) -> shap.TreeExplainer:
    return shap.TreeExplainer(unwrap_calibrated_model(model))


def explain_prediction(
    explainer: shap.TreeExplainer,
    scaler: Any,
    features: list[str],
    input_dict: dict[str, Any],
    top_n: int = 5,
) -> dict[str, Any]:
    """Return base probability, raw tree probability and top SHAP factors."""
    row = pd.DataFrame([input_dict], columns=features)
    scaled = scaler.transform(row)
    shap_values = explainer.shap_values(scaled)

    if isinstance(shap_values, list):
        contributions = shap_values[1][0]
    else:
        values = getattr(shap_values, "values", shap_values)
        if getattr(values, "ndim", 1) == 3:
            contributions = values[0, :, 1]
        else:
            contributions = values[0]

    expected = explainer.expected_value
    if hasattr(expected, "ndim") and expected.ndim > 0:
        expected = expected[1]
    elif isinstance(expected, (list, tuple)):
        expected = expected[1]
    elif hasattr(expected, "__len__") and not isinstance(expected, (str, bytes)):
        try:
            if len(expected) > 1:
                expected = expected[1]
        except TypeError:
            pass

    base_value = float(expected)
    final_log_odds = base_value + float(sum(contributions))
    base_probability = sigmoid(base_value)
    raw_probability = sigmoid(final_log_odds)

    items = list(zip(features, contributions, row.iloc[0].values))
    items.sort(key=lambda item: abs(float(item[1])), reverse=True)

    factors = [
        {
            "feature": feature,
            "value": _python_scalar(value),
            "direction": "increases" if float(contribution) > 0 else "decreases",
            "magnitude": round(abs(float(contribution)), 3),
        }
        for feature, contribution, value in items[:top_n]
    ]

    return {
        "base_rate_probability": round(base_probability, 4),
        "raw_model_probability": round(raw_probability, 4),
        "top_factors": factors,
    }


def _python_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value
