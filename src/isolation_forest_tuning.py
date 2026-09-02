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
# LOAD DATA
# ---------------------------------------------------------

train_normal = pd.read_csv(DATA_DIR / "train_normal.csv")
validation_normal = pd.read_csv(DATA_DIR / "validation_normal.csv")
parkinsons = pd.read_csv(DATA_DIR / "evaluation_parkinsons.csv")

X_train = train_normal.values.astype(np.float64)
X_validation = validation_normal.values.astype(np.float64)
X_parkinsons = parkinsons.values.astype(np.float64)


print("=" * 60)
print("ISOLATION FOREST HYPERPARAMETER TUNING")
print("=" * 60)

print("Training normal:", X_train.shape)
print("Validation normal:", X_validation.shape)
print("Parkinson's:", X_parkinsons.shape)


# ---------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------

configs = [
    {"n_estimators": 100, "max_samples": "auto", "max_features": 1.0},
    {"n_estimators": 300, "max_samples": "auto", "max_features": 1.0},
    {"n_estimators": 500, "max_samples": "auto", "max_features": 1.0},

    {"n_estimators": 300, "max_samples": 0.5, "max_features": 1.0},
    {"n_estimators": 300, "max_samples": 0.8, "max_features": 1.0},

    {"n_estimators": 300, "max_samples": "auto", "max_features": 0.5},
    {"n_estimators": 300, "max_samples": "auto", "max_features": 0.75},
]


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

results = []

for i, config in enumerate(configs, start=1):

    model = IsolationForest(
        n_estimators=config["n_estimators"],
        max_samples=config["max_samples"],
        max_features=config["max_features"],
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    # Train ONLY on healthy subjects
    model.fit(X_train)

    # Higher score = more anomalous
    validation_scores = -model.decision_function(X_validation)
    parkinsons_scores = -model.decision_function(X_parkinsons)

    labels = np.concatenate([
        np.zeros(len(validation_scores)),
        np.ones(len(parkinsons_scores))
    ])

    scores = np.concatenate([
        validation_scores,
        parkinsons_scores
    ])

    auc = roc_auc_score(labels, scores)

    results.append({
        "experiment": i,
        "n_estimators": config["n_estimators"],
        "max_samples": config["max_samples"],
        "max_features": config["max_features"],
        "roc_auc": auc
    })

    print(
        f"\nExperiment {i}"
    )
    print(
        f"n_estimators={config['n_estimators']}, "
        f"max_samples={config['max_samples']}, "
        f"max_features={config['max_features']}"
    )
    print(f"ROC-AUC: {auc:.4f}")


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "roc_auc",
    ascending=False
)

output_path = RESULTS_DIR / "isolation_forest_tuning_results.csv"

results_df.to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nBest configuration:")
print(results_df.iloc[0].to_string())

print("\nSaved:")
print(output_path)