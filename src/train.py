from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import SpeechAutoencoder


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "train_normal.csv"
)

MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = MODEL_DIR / "autoencoder.pt"


EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# Load normal training data
# --------------------------------------------------

data = pd.read_csv(DATA_PATH)

X = data.values.astype(np.float32)

print("Training data shape:", X.shape)


# --------------------------------------------------
# Convert to PyTorch
# --------------------------------------------------

X_tensor = torch.tensor(
    X,
    dtype=torch.float32,
)

dataset = TensorDataset(X_tensor)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = SpeechAutoencoder(
    input_dim=X.shape[1],
    hidden_dim=16,
    latent_dim=8,
).to(device)


# --------------------------------------------------
# Loss + optimizer
# --------------------------------------------------

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# Training
# --------------------------------------------------

print("\nStarting training...\n")

training_losses = []


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch in loader:

        x = batch[0].to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        reconstruction = model(x)

        # Reconstruction loss
        loss = criterion(
            reconstruction,
            x,
        )

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    training_losses.append(average_loss)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1:3d}/{EPOCHS}] "
            f"Loss: {average_loss:.6f}"
        )


# --------------------------------------------------
# Save model
# --------------------------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "hidden_dim": 16,
        "latent_dim": 8,
    },
    MODEL_PATH,
)


# --------------------------------------------------
# Save training losses
# --------------------------------------------------

loss_path = (
    PROJECT_DIR
    / "results"
    / "metrics"
    / "training_loss.csv"
)

pd.DataFrame(
    {
        "epoch": range(1, EPOCHS + 1),
        "loss": training_losses,
    }
).to_csv(
    loss_path,
    index=False,
)


print("\nTraining completed.")
print("Model saved to:", MODEL_PATH)
print("Loss history saved to:", loss_path)