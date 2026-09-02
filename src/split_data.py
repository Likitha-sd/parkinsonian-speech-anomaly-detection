from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

FEATURES_PATH = PROCESSED_DIR / "train_features.csv"
LABELS_PATH = PROCESSED_DIR / "train_labels.csv"
SUBJECTS_PATH = PROCESSED_DIR / "train_subject_ids.csv"


# --------------------------------------------------
# Load processed data
# --------------------------------------------------

features = pd.read_csv(FEATURES_PATH)
labels = pd.read_csv(LABELS_PATH).iloc[:, 0]
subject_ids = pd.read_csv(SUBJECTS_PATH).iloc[:, 0]


data = features.copy()

data["label"] = labels
data["subject_id"] = subject_ids


print("=" * 60)
print("ORIGINAL DATA")
print("=" * 60)

print("Samples:", len(data))
print("Subjects:", data["subject_id"].nunique())

print("\nLabels:")
print(data["label"].value_counts())


# --------------------------------------------------
# Separate healthy and Parkinson's subjects
# --------------------------------------------------

healthy = data[data["label"] == 0].copy()
parkinsons = data[data["label"] == 1].copy()


healthy_subjects = healthy["subject_id"].unique()
parkinsons_subjects = parkinsons["subject_id"].unique()


print("\nHealthy subjects:", len(healthy_subjects))
print("Parkinson's subjects:", len(parkinsons_subjects))


# --------------------------------------------------
# Split HEALTHY subjects
# --------------------------------------------------

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


# --------------------------------------------------
# Evaluation data
# --------------------------------------------------

# Parkinson's samples are never used to train
# the autoencoder.

evaluation_parkinsons = parkinsons.copy()


# --------------------------------------------------
# Remove metadata from model inputs
# --------------------------------------------------

feature_columns = [
    column
    for column in data.columns
    if column not in ["label", "subject_id"]
]


train_normal_features = train_normal[
    feature_columns
]

validation_normal_features = validation_normal[
    feature_columns
]

evaluation_parkinsons_features = evaluation_parkinsons[
    feature_columns
]


# --------------------------------------------------
# Save datasets
# --------------------------------------------------

train_normal_features.to_csv(
    PROCESSED_DIR / "train_normal.csv",
    index=False,
)

validation_normal_features.to_csv(
    PROCESSED_DIR / "validation_normal.csv",
    index=False,
)

evaluation_parkinsons_features.to_csv(
    PROCESSED_DIR / "evaluation_parkinsons.csv",
    index=False,
)


# Save labels for evaluation
evaluation_parkinsons["label"].to_csv(
    PROCESSED_DIR / "evaluation_parkinsons_labels.csv",
    index=False,
)


# --------------------------------------------------
# Verify subject separation
# --------------------------------------------------

train_set = set(train_subjects)
validation_set = set(validation_subjects)
parkinsons_set = set(parkinsons_subjects)


print("\n" + "=" * 60)
print("SUBJECT OVERLAP CHECK")
print("=" * 60)

print(
    "Train ∩ Validation:",
    train_set.intersection(validation_set),
)

print(
    "Train ∩ Parkinson's:",
    train_set.intersection(parkinsons_set),
)

print(
    "Validation ∩ Parkinson's:",
    validation_set.intersection(parkinsons_set),
)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("SPLIT SUMMARY")
print("=" * 60)

print(
    "Normal training samples:",
    len(train_normal_features),
)

print(
    "Normal validation samples:",
    len(validation_normal_features),
)

print(
    "Parkinson's evaluation samples:",
    len(evaluation_parkinsons_features),
)

print("\nSaved:")
print("  train_normal.csv")
print("  validation_normal.csv")
print("  evaluation_parkinsons.csv")
print("  evaluation_parkinsons_labels.csv")