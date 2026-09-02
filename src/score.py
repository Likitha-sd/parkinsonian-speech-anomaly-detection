from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import SpeechAutoencoder


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_PATH = PROJECT_DIR / "models" / "autoencoder.pt"
RESULTS_DIR = PROJECT_DIR / "results" / "metrics"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# Load model
# --------------------------------------------------

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
# Reconstruction error function
# --------------------------------------------------

def calculate_reconstruction_error(data):

    X = data.values.astype(np.float32)

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():

        reconstruction = model(X_tensor)

        errors = torch.mean(
            (X_tensor - reconstruction) ** 2,
            dim=1,
        )

    return errors.cpu().numpy()


# --------------------------------------------------
# Load data
# --------------------------------------------------

normal_validation = pd.read_csv(
    DATA_DIR / "validation_normal.csv"
)

parkinsons = pd.read_csv(
    DATA_DIR / "evaluation_parkinsons.csv"
)


print("\nData:")
print(
    "Healthy validation:",
    normal_validation.shape
)

print(
    "Parkinson's evaluation:",
    parkinsons.shape
)


# --------------------------------------------------
# Calculate anomaly scores
# --------------------------------------------------

normal_errors = calculate_reconstruction_error(
    normal_validation
)

parkinsons_errors = calculate_reconstruction_error(
    parkinsons
)


# --------------------------------------------------
# Print statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("RECONSTRUCTION ERROR")
print("=" * 60)

print("\nHealthy validation:")
print(
    f"Mean   : {normal_errors.mean():.6f}"
)

print(
    f"Median : {np.median(normal_errors):.6f}"
)

print(
    f"Std    : {normal_errors.std():.6f}"
)

print(
    f"Min    : {normal_errors.min():.6f}"
)

print(
    f"Max    : {normal_errors.max():.6f}"
)


print("\nParkinson's:")
print(
    f"Mean   : {parkinsons_errors.mean():.6f}"
)

print(
    f"Median : {np.median(parkinsons_errors):.6f}"
)

print(
    f"Std    : {parkinsons_errors.std():.6f}"
)

print(
    f"Min    : {parkinsons_errors.min():.6f}"
)

print(
    f"Max    : {parkinsons_errors.max():.6f}"
)


# --------------------------------------------------
# Save scores
# --------------------------------------------------

normal_scores = pd.DataFrame(
    {
        "group": "healthy",
        "reconstruction_error": normal_errors,
    }
)

parkinsons_scores = pd.DataFrame(
    {
        "group": "parkinsons",
        "reconstruction_error": parkinsons_errors,
    }
)


scores = pd.concat(
    [
        normal_scores,
        parkinsons_scores,
    ],
    ignore_index=True,
)


scores.to_csv(
    RESULTS_DIR / "reconstruction_errors.csv",
    index=False,
)


print("\nSaved:")
print(
    RESULTS_DIR
    / "reconstruction_errors.csv"
)