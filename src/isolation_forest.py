from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "metrics"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

train_normal = pd.read_csv(
    DATA_DIR / "train_normal.csv"
)

validation_normal = pd.read_csv(
    DATA_DIR / "validation_normal.csv"
)

parkinsons = pd.read_csv(
    DATA_DIR / "evaluation_parkinsons.csv"
)


X_train = train_normal.values.astype(np.float64)
X_validation = validation_normal.values.astype(np.float64)
X_parkinsons = parkinsons.values.astype(np.float64)


# --------------------------------------------------
# Train Isolation Forest
# --------------------------------------------------

model = IsolationForest(
    n_estimators=300,
    contamination="auto",
    random_state=42,
)


# IMPORTANT:
# Train ONLY on healthy training data.
model.fit(X_train)


# --------------------------------------------------
# Calculate anomaly scores
# --------------------------------------------------

validation_scores = -model.decision_function(
    X_validation
)

parkinsons_scores = -model.decision_function(
    X_parkinsons
)


# --------------------------------------------------
# Combine evaluation data
# --------------------------------------------------

true_labels = np.concatenate(
    [
        np.zeros(len(validation_scores)),
        np.ones(len(parkinsons_scores)),
    ]
)

anomaly_scores = np.concatenate(
    [
        validation_scores,
        parkinsons_scores,
    ]
)


# --------------------------------------------------
# ROC-AUC
# --------------------------------------------------

roc_auc = roc_auc_score(
    true_labels,
    anomaly_scores,
)


# --------------------------------------------------
# Threshold
# --------------------------------------------------

threshold = np.percentile(
    validation_scores,
    95,
)

predicted_labels = (
    anomaly_scores > threshold
).astype(int)


precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0,
)

recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0,
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0,
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("=" * 60)
print("ISOLATION FOREST BASELINE")
print("=" * 60)

print(
    "Training normal:",
    X_train.shape
)

print(
    "Validation normal:",
    X_validation.shape
)

print(
    "Parkinson's:",
    X_parkinsons.shape
)

print()

print(
    f"ROC-AUC:  {roc_auc:.4f}"
)

print(
    f"Threshold: {threshold:.6f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1-score:  {f1:.4f}"
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

results = pd.DataFrame(
    {
        "metric": [
            "roc_auc",
            "threshold",
            "precision",
            "recall",
            "f1",
        ],
        "value": [
            roc_auc,
            threshold,
            precision,
            recall,
            f1,
        ],
    }
)

output_path = (
    RESULTS_DIR /
    "isolation_forest_results.csv"
)

results.to_csv(
    output_path,
    index=False,
)

print(
    "\nSaved:",
    output_path
)