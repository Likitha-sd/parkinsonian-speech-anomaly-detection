from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed"


# --------------------------------------------------
# Load data
# --------------------------------------------------

healthy = pd.read_csv(
    DATA_DIR / "validation_normal.csv"
)

parkinsons = pd.read_csv(
    DATA_DIR / "evaluation_parkinsons.csv"
)


feature_names = healthy.columns


# --------------------------------------------------
# Analyze every feature
# --------------------------------------------------

results = []

for feature in feature_names:

    healthy_values = healthy[feature].values
    parkinsons_values = parkinsons[feature].values

    # Use the feature itself as a simple score.
    # We calculate both directions because a Parkinsonian
    # pattern could correspond to either a higher or lower value.

    labels = np.concatenate(
        [
            np.zeros(len(healthy_values)),
            np.ones(len(parkinsons_values)),
        ]
    )

    values = np.concatenate(
        [
            healthy_values,
            parkinsons_values,
        ]
    )

    auc_forward = roc_auc_score(
        labels,
        values,
    )

    auc_reverse = roc_auc_score(
        labels,
        -values,
    )

    best_auc = max(
        auc_forward,
        auc_reverse,
    )

    results.append(
        {
            "feature": feature,
            "healthy_mean": np.mean(healthy_values),
            "parkinsons_mean": np.mean(parkinsons_values),
            "healthy_std": np.std(healthy_values),
            "parkinsons_std": np.std(parkinsons_values),
            "auc": best_auc,
        }
    )


# --------------------------------------------------
# Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "auc",
    ascending=False,
)


print("=" * 70)
print("FEATURE-LEVEL ANALYSIS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_path = (
    PROJECT_DIR
    / "results"
    / "metrics"
    / "feature_analysis.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)


print("\nSaved:")
print(output_path)