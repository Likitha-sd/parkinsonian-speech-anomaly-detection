from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\Likithasri\Downloads\parkinson+speech+dataset+with+multiple+types+of+sound+recordings"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_PATH = DATA_DIR / "train_data.txt"
TEST_PATH = DATA_DIR / "test_data.txt"


# --------------------------------------------------
# Load data
# --------------------------------------------------

train = pd.read_csv(TRAIN_PATH, header=None)
test = pd.read_csv(TEST_PATH, header=None)

print("Original shapes:")
print("Train:", train.shape)
print("Test :", test.shape)


# --------------------------------------------------
# Select columns correctly
# --------------------------------------------------

# Training:
# column 0      = subject ID
# columns 1-26  = 26 acoustic features
# column 28     = class label
#
# Test:
# column 0      = subject ID
# columns 1-26  = 26 acoustic features
# column 27     = class label


train_subject_ids = train.iloc[:, 0]
test_subject_ids = test.iloc[:, 0]

train_features = train.iloc[:, 1:27]
test_features = test.iloc[:, 1:27]

train_labels = train.iloc[:, 28]
test_labels = test.iloc[:, 27]


# --------------------------------------------------
# Validate dimensions
# --------------------------------------------------

print("\nAfter separating data:")

print("Train features:", train_features.shape)
print("Test features :", test_features.shape)

print("\nTrain labels:")
print(train_labels.value_counts())

print("\nTest labels:")
print(test_labels.value_counts())


if train_features.shape[1] != 26:
    raise ValueError("Training data should contain 26 features.")

if test_features.shape[1] != 26:
    raise ValueError("Test data should contain 26 features.")


# --------------------------------------------------
# Standardization
# --------------------------------------------------

scaler = StandardScaler()

# IMPORTANT:
# Fit the scaler ONLY on training data.
train_scaled = scaler.fit_transform(train_features)

# Transform test using the same scaler.
test_scaled = scaler.transform(test_features)


# --------------------------------------------------
# Save processed features
# --------------------------------------------------

train_scaled = pd.DataFrame(
    train_scaled,
    columns=[f"feature_{i}" for i in range(1, 27)],
)

test_scaled = pd.DataFrame(
    test_scaled,
    columns=[f"feature_{i}" for i in range(1, 27)],
)


train_scaled.to_csv(
    PROCESSED_DIR / "train_features.csv",
    index=False,
)

test_scaled.to_csv(
    PROCESSED_DIR / "test_features.csv",
    index=False,
)

train_labels.to_csv(
    PROCESSED_DIR / "train_labels.csv",
    index=False,
)

test_labels.to_csv(
    PROCESSED_DIR / "test_labels.csv",
    index=False,
)

train_subject_ids.to_csv(
    PROCESSED_DIR / "train_subject_ids.csv",
    index=False,
)

test_subject_ids.to_csv(
    PROCESSED_DIR / "test_subject_ids.csv",
    index=False,
)


# --------------------------------------------------
# Save scaler
# --------------------------------------------------

joblib.dump(
    scaler,
    MODELS_DIR / "scaler.pkl",
)


# --------------------------------------------------
# Final confirmation
# --------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("Train features:", train_scaled.shape)
print("Test features :", test_scaled.shape)

print("\nSaved:")
print("  data/processed/train_features.csv")
print("  data/processed/test_features.csv")
print("  data/processed/train_labels.csv")
print("  data/processed/test_labels.csv")
print("  data/processed/train_subject_ids.csv")
print("  data/processed/test_subject_ids.csv")
print("  models/scaler.pkl")