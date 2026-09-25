"""Metrics collection, JSON reporting, and cross-disease comparison."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.config import settings


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_proba) -> dict[str, Any]:
    """Compute the standard metric suite for a binary classifier."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else 0.0,
        "auc_pr": float(pr_auc),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "n_samples": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }


def compute_confidence_summary(results: list[dict[str, Any]]) -> dict[str, int]:
    """Count how many predictions fell into each confidence bucket."""
    summary = {"High": 0, "Medium": 0, "Low": 0}
    for r in results:
        level = r.get("confidence", "Low")
        if level in summary:
            summary[level] += 1
    return summary


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

def build_report(
    disease: str,
    models: dict[str, dict[str, Any]],
    ensemble: dict[str, Any],
    dataset_info: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a full report for one disease.

    Parameters
    ----------
    disease : str
        Disease name ("malaria", "covid", "dengue")
    models : dict
        Map of model name → metrics dict (e.g., {"xgb": {...}, "rf": {...}, "lr": {...}})
    ensemble : dict
        Metrics for the stacked ensemble.
    dataset_info : dict
        Dataset metadata (n_train, n_test, feature_count, class_balance, etc.)
    """
    return {
        "disease": disease,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_info,
        "individual_models": models,
        "ensemble": ensemble,
        "best_individual_model": max(
            models.items(), key=lambda kv: kv[1]["f1"]
        )[0] if models else None,
        "ensemble_beats_best_individual": (
            ensemble["f1"] > max(m["f1"] for m in models.values())
            if models else False
        ),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy scalars to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_report(report: dict[str, Any], path: Path) -> None:
    """Write one disease report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_sanitize(report), f, indent=2)
    print(f"📊 Report saved → {path}")


def save_comparison(reports: list[dict[str, Any]], path: Path) -> None:
    """Write a cross-disease comparison CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for report in reports:
        disease = report["disease"]
        ens = report["ensemble"]

        # Best individual model by F1
        best_name = report.get("best_individual_model")
        best_metrics = report["individual_models"].get(best_name, {})

        rows.append({
            "disease": disease,
            "n_train": report["dataset"].get("n_train"),
            "n_test": report["dataset"].get("n_test"),
            "n_features": report["dataset"].get("n_features"),
            "positive_rate": report["dataset"].get("positive_rate"),

            "ensemble_accuracy": ens["accuracy"],
            "ensemble_precision": ens["precision"],
            "ensemble_recall": ens["recall"],
            "ensemble_f1": ens["f1"],
            "ensemble_auc_roc": ens["auc_roc"],
            "ensemble_auc_pr": ens["auc_pr"],

            "best_individual_model": best_name,
            "best_individual_f1": best_metrics.get("f1"),

            "ensemble_gain_f1": (
                ens["f1"] - best_metrics.get("f1", 0)
                if best_metrics else None
            ),
        })

    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"📊 Comparison report saved → {path}")


# ---------------------------------------------------------------------------
# Per-disease convenience
# ---------------------------------------------------------------------------

def save_disease_report(disease: str, report: dict[str, Any]) -> None:
    """Route a report to the correct path."""
    path_map = {
        "malaria": settings.malaria_report,
        "covid": settings.covid_report,
        "dengue": settings.dengue_report,
    }
    path = path_map.get(disease)
    if path is None:
        raise ValueError(f"Unknown disease: {disease}")
    save_report(report, path)


def load_all_reports() -> list[dict[str, Any]]:
    """Load all available disease reports (skips missing)."""
    reports = []
    for path in [settings.malaria_report, settings.covid_report, settings.dengue_report]:
        if path.exists():
            with open(path) as f:
                reports.append(json.load(f))
    return reports


def rebuild_comparison() -> None:
    """Regenerate the comparison CSV from all existing reports."""
    reports = load_all_reports()
    if not reports:
        print("⚠️  No reports found — train at least one model first.")
        return
    save_comparison(reports, settings.comparison_report)


if __name__ == "__main__":
    rebuild_comparison()