from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOAD ORIGINAL PROCESSED DATA
# ---------------------------------------------------------

features = pd.read_csv(
    DATA_DIR / "train_features.csv"
)

labels = pd.read_csv(
    DATA_DIR / "train_labels.csv"
).iloc[:, 0]

subject_ids = pd.read_csv(
    DATA_DIR / "train_subject_ids.csv"
).iloc[:, 0]


data = features.copy()
data["label"] = labels
data["subject_id"] = subject_ids


# ---------------------------------------------------------
# RECREATE SUBJECT SPLIT
# ---------------------------------------------------------

from sklearn.model_selection import train_test_split


healthy = data[data["label"] == 0].copy()
parkinsons = data[data["label"] == 1].copy()

healthy_subjects = healthy["subject_id"].unique()

train_subjects, validation_subjects = train_test_split(
    healthy_subjects,
    test_size=0.20,
    random_state=42,
)


train_normal = healthy[
    healthy["subject_id"].isin(train_subjects)
].copy()

validation_normal = healthy[
    healthy["subject_id"].isin(validation_subjects)
].copy()


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

feature_columns = [
    c for c in data.columns
    if c not in ["label", "subject_id"]
]

X_train = train_normal[feature_columns].values.astype(np.float64)
X_validation = validation_normal[feature_columns].values.astype(np.float64)
X_parkinsons = parkinsons[feature_columns].values.astype(np.float64)


# ---------------------------------------------------------
# ISOLATION FOREST
# ---------------------------------------------------------

model = IsolationForest(
    n_estimators=300,
    contamination="auto",
    random_state=42,
)

model.fit(X_train)


# Higher = more anomalous
validation_scores = -model.decision_function(
    X_validation
)

parkinsons_scores = -model.decision_function(
    X_parkinsons
)


# ---------------------------------------------------------
# RECORDING-LEVEL RESULTS
# ---------------------------------------------------------

recording_labels = np.concatenate([
    np.zeros(len(validation_scores)),
    np.ones(len(parkinsons_scores))
])

recording_scores = np.concatenate([
    validation_scores,
    parkinsons_scores
])

recording_auc = roc_auc_score(
    recording_labels,
    recording_scores
)


# ---------------------------------------------------------
# SUBJECT-LEVEL AGGREGATION
# ---------------------------------------------------------

validation_subject_scores = pd.DataFrame({
    "subject_id": validation_normal["subject_id"].values,
    "score": validation_scores
})

parkinsons_subject_scores = pd.DataFrame({
    "subject_id": parkinsons["subject_id"].values,
    "score": parkinsons_scores
})


# Mean anomaly score per subject
validation_aggregated = (
    validation_subject_scores
    .groupby("subject_id")["score"]
    .mean()
)

parkinsons_aggregated = (
    parkinsons_subject_scores
    .groupby("subject_id")["score"]
    .mean()
)


subject_scores = np.concatenate([
    validation_aggregated.values,
    parkinsons_aggregated.values
])

subject_labels = np.concatenate([
    np.zeros(len(validation_aggregated)),
    np.ones(len(parkinsons_aggregated))
])


subject_auc = roc_auc_score(
    subject_labels,
    subject_scores
)


# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------

print("=" * 60)
print("SUBJECT-LEVEL EVALUATION")
print("=" * 60)

print(
    "Healthy validation subjects:",
    len(validation_aggregated)
)

print(
    "Parkinson's subjects:",
    len(parkinsons_aggregated)
)

print()

print(
    f"Recording-level ROC-AUC: {recording_auc:.4f}"
)

print(
    f"Subject-level ROC-AUC:    {subject_auc:.4f}"
)


# ---------------------------------------------------------
# SAVE SUBJECT SCORES
# ---------------------------------------------------------

subject_results = pd.DataFrame({
    "subject_id": (
        list(validation_aggregated.index)
        + list(parkinsons_aggregated.index)
    ),
    "label": subject_labels.astype(int),
    "mean_anomaly_score": subject_scores
})

subject_results.to_csv(
    RESULTS_DIR / "subject_level_scores.csv",
    index=False
)


comparison = pd.DataFrame({
    "evaluation_level": [
        "recording",
        "subject"
    ],
    "roc_auc": [
        recording_auc,
        subject_auc
    ]
})

comparison.to_csv(
    RESULTS_DIR / "subject_level_comparison.csv",
    index=False
)


print("\nSaved:")
print(
    RESULTS_DIR / "subject_level_scores.csv"
)
print(
    RESULTS_DIR / "subject_level_comparison.csv"
)