from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent

SCORES_PATH = (
    PROJECT_DIR
    / "results"
    / "metrics"
    / "reconstruction_errors.csv"
)

FIGURES_DIR = PROJECT_DIR / "results" / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Load scores
scores = pd.read_csv(SCORES_PATH)

healthy = scores[
    scores["group"] == "healthy"
]["reconstruction_error"]

parkinsons = scores[
    scores["group"] == "parkinsons"
]["reconstruction_error"]


# --------------------------------------------------
# Histogram
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    healthy,
    bins=30,
    alpha=0.6,
    label="Healthy",
)

plt.hist(
    parkinsons,
    bins=30,
    alpha=0.6,
    label="Parkinson's",
)

plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Samples")

plt.title(
    "Reconstruction Error Distribution"
)

plt.legend()

plt.tight_layout()

hist_path = (
    FIGURES_DIR
    / "reconstruction_error_distribution.png"
)

plt.savefig(hist_path)

plt.close()


# --------------------------------------------------
# Box plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.boxplot(
    [
        healthy,
        parkinsons,
    ],
    tick_labels=[
        "Healthy",
        "Parkinson's",
    ],
)
plt.ylabel("Reconstruction Error")

plt.title(
    "Reconstruction Error by Group"
)

plt.tight_layout()

box_path = (
    FIGURES_DIR
    / "reconstruction_error_boxplot.png"
)

plt.savefig(box_path)

plt.close()


print("Saved:")
print(hist_path)
print(box_path)