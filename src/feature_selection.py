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
print("FEATURE SELECTION + ISOLATION FOREST")
print("=" * 60)

print("Training normal:", X_train.shape)
print("Validation normal:", X_validation.shape)
print("Parkinson's:", X_parkinsons.shape)


# ---------------------------------------------------------
# FEATURE RANKING
# ---------------------------------------------------------

feature_analysis = pd.read_csv(
    RESULTS_DIR / "feature_analysis.csv"
)

feature_analysis = feature_analysis.sort_values(
    "auc",
    ascending=False
)

ranked_features = feature_analysis["feature"].tolist()


print("\nFeature ranking:")
for i, feature in enumerate(ranked_features, start=1):
    auc = feature_analysis.loc[
        feature_analysis["feature"] == feature, "auc"
    ].iloc[0]

    print(f"{i:2d}. {feature:12s} AUC = {auc:.4f}")


# ---------------------------------------------------------
# TEST DIFFERENT NUMBERS OF FEATURES
# ---------------------------------------------------------

feature_counts = [3, 5, 10, 15, 26]

results = []


for k in feature_counts:

    selected_features = ranked_features[:k]

    indices = [
        int(feature.split("_")[1]) - 1
        for feature in selected_features
    ]

    X_train_selected = X_train[:, indices]
    X_validation_selected = X_validation[:, indices]
    X_parkinsons_selected = X_parkinsons[:, indices]

    # Train only on healthy training data
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42
    )

    model.fit(X_train_selected)

    # Higher = more anomalous
    validation_scores = -model.decision_function(
        X_validation_selected
    )

    parkinsons_scores = -model.decision_function(
        X_parkinsons_selected
    )

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
        "num_features": k,
        "roc_auc": auc,
        "features": ", ".join(selected_features)
    })

    print(
        f"\nTop {k} features"
    )
    print(
        "ROC-AUC:",
        f"{auc:.4f}"
    )


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

output_path = RESULTS_DIR / "feature_selection_results.csv"

results_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nSaved:")
print(output_path)