from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf
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


print("=" * 60)
print("MAHALANOBIS ANOMALY DETECTION")
print("=" * 60)

print("Training normal:", X_train.shape)
print("Validation normal:", X_validation.shape)
print("Parkinson's:", X_parkinsons.shape)


# --------------------------------------------------
# Learn normal distribution
# --------------------------------------------------

# Ledoit-Wolf gives us a regularized covariance estimate.
# This is more stable than directly inverting a sample
# covariance matrix.

covariance_model = LedoitWolf()

covariance_model.fit(X_train)

mean_vector = covariance_model.location_
covariance = covariance_model.covariance_


# --------------------------------------------------
# Calculate Mahalanobis distance
# --------------------------------------------------

def mahalanobis_distance(X):

    differences = X - mean_vector

    precision = covariance_model.precision_

    distances = np.einsum(
        "ij,jk,ik->i",
        differences,
        precision,
        differences,
    )

    return distances


# --------------------------------------------------
# Calculate anomaly scores
# --------------------------------------------------

validation_scores = mahalanobis_distance(
    X_validation
)

parkinsons_scores = mahalanobis_distance(
    X_parkinsons
)


# --------------------------------------------------
# Print score statistics
# --------------------------------------------------

print("\nHealthy validation scores:")

print(
    f"Mean   : {validation_scores.mean():.6f}"
)

print(
    f"Median : {np.median(validation_scores):.6f}"
)

print(
    f"Min    : {validation_scores.min():.6f}"
)

print(
    f"Max    : {validation_scores.max():.6f}"
)


print("\nParkinson's scores:")

print(
    f"Mean   : {parkinsons_scores.mean():.6f}"
)

print(
    f"Median : {np.median(parkinsons_scores):.6f}"
)

print(
    f"Min    : {parkinsons_scores.min():.6f}"
)

print(
    f"Max    : {parkinsons_scores.max():.6f}"
)


# --------------------------------------------------
# Create evaluation set
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


print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)


# --------------------------------------------------
# Threshold from healthy validation data
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
    RESULTS_DIR
    / "mahalanobis_results.csv"
)

results.to_csv(
    output_path,
    index=False,
)


print("\nSaved:")
print(output_path)