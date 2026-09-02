from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from model import SpeechAutoencoder


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_PATH = PROJECT_DIR / "models" / "autoencoder.pt"
SCORES_PATH = (
    PROJECT_DIR
    / "results"
    / "metrics"
    / "reconstruction_errors.csv"
)

RESULTS_DIR = PROJECT_DIR / "results" / "metrics"


# --------------------------------------------------
# Load model
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
)

model = SpeechAutoencoder(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    latent_dim=checkpoint["latent_dim"],
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


# --------------------------------------------------
# Load reconstruction errors
# --------------------------------------------------

scores = pd.read_csv(SCORES_PATH)

healthy_errors = scores[
    scores["group"] == "healthy"
]["reconstruction_error"].values

parkinsons_errors = scores[
    scores["group"] == "parkinsons"
]["reconstruction_error"].values


# --------------------------------------------------
# Determine threshold
# --------------------------------------------------

percentile = 95

threshold = np.percentile(
    healthy_errors,
    percentile,
)

print("=" * 60)
print("ANOMALY THRESHOLD")
print("=" * 60)

print(
    f"Healthy {percentile}th percentile:"
    f" {threshold:.6f}"
)


# --------------------------------------------------
# Create evaluation labels
# --------------------------------------------------

# Healthy = 0
# Parkinson's = 1

true_labels = np.concatenate(
    [
        np.zeros(len(healthy_errors)),
        np.ones(len(parkinsons_errors)),
    ]
)

anomaly_scores = np.concatenate(
    [
        healthy_errors,
        parkinsons_errors,
    ]
)


predicted_labels = (
    anomaly_scores > threshold
).astype(int)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

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

roc_auc = roc_auc_score(
    true_labels,
    anomaly_scores,
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 60)
print("ANOMALY DETECTION RESULTS")
print("=" * 60)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    true_labels,
    predicted_labels,
)

print("\nConfusion Matrix:")
print(cm)


print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "Healthy",
            "Parkinson's",
        ],
        zero_division=0,
    )
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

results = pd.DataFrame(
    {
        "metric": [
            "threshold",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ],
        "value": [
            threshold,
            precision,
            recall,
            f1,
            roc_auc,
        ],
    }
)

results.to_csv(
    RESULTS_DIR / "baseline_results.csv",
    index=False,
)


print(
    "\nSaved:",
    RESULTS_DIR / "baseline_results.csv"
)