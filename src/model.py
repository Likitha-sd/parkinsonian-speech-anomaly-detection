import torch
import torch.nn as nn


class SpeechAutoencoder(nn.Module):
    """
    Autoencoder for learning a compact representation
    of normal speech features.
    """

    def __init__(
        self,
        input_dim=26,
        hidden_dim=16,
        latent_dim=8,
    ):
        super().__init__()

        # Encoder:
        # 26 features -> 16 -> 8 latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder:
        # 8 latent features -> 16 -> 26 reconstructed features
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)

        return reconstruction