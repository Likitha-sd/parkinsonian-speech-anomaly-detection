
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# Locate the existing reconstruction scores
PROJECT_DIR = Path(__file__).resolve().parent.parent

SCORES_PATH = (
    PROJECT_DIR
    / "results"
    / "metrics"
    / "reconstruction_errors.csv"
)

# Load scores produced by score.py
data = pd.read_csv(SCORES_PATH)

healthy = data.loc[
    data["group"] == "healthy",
    "reconstruction_error"
].to_numpy()

pd_errors = data.loc[
    data["group"] == "parkinsons",
    "reconstruction_error"
].to_numpy()

if len(healthy) < 4 or len(pd_errors) < 1:
    raise ValueError(
        "Not enough healthy or Parkinson's samples to evaluate."
    )

# Reserve healthy samples for threshold calibration
rng = np.random.default_rng(42)
healthy = rng.permutation(healthy)

split = len(healthy) // 2
calibration_errors = healthy[:split]
healthy_test_errors = healthy[split:]

# Choose a threshold using healthy calibration data only
threshold = np.quantile(calibration_errors, 0.95)

# Evaluate on held-out healthy samples and PD samples
y_true = np.concatenate([
    np.zeros(len(healthy_test_errors)),
    np.ones(len(pd_errors)),
])

y_scores = np.concatenate([
    healthy_test_errors,
    pd_errors,
])

# Higher reconstruction error = predicted anomaly
y_pred = (y_scores > threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
).ravel()

sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
specificity = tn / (tn + fp) if (tn + fp) else 0.0

roc_auc = roc_auc_score(y_true, y_scores)
avg_precision = average_precision_score(y_true, y_scores)

print("\n=== Anomaly Detection Evaluation ===")
print(f"Threshold:             {threshold:.6f}")
print(f"ROC-AUC:               {roc_auc:.4f}")
print(f"Average Precision:     {avg_precision:.4f}")
print(f"Sensitivity:           {sensitivity:.4f}")
print(f"Specificity:           {specificity:.4f}")
print(f"True positives:        {tp}")
print(f"False negatives:       {fn}")
print(f"True negatives:        {tn}")
print(f"False positives:       {fp}")

# Save the results for reporting
output_path = (
    PROJECT_DIR
    / "results"
    / "metrics"
    / "detection_metrics.csv"
)

pd.DataFrame([{
    "threshold": threshold,
    "roc_auc": roc_auc,
    "average_precision": avg_precision,
    "sensitivity": sensitivity,
    "specificity": specificity,
    "true_positives": tp,
    "false_negatives": fn,
    "true_negatives": tn,
    "false_positives": fp,
}]).to_csv(output_path, index=False)

print(f"\nMetrics saved to: {output_path}")
