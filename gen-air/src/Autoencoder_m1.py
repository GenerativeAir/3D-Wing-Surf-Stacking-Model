import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """VAE geometrico: profilo alare -> profilo alare ricostruito.

    Input/output:
        100 ordinate y = 50 dorso + 50 pancia.
    """

    def __init__(self, input_dim=100, latent_dim=16):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.act = nn.LeakyReLU(0.1)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 32)
        self.fc4 = nn.Linear(32, 64)
        self.fc5 = nn.Linear(64, input_dim)

    def encode(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.act(self.fc3(z))
        h = self.act(self.fc4(h))
        return self.fc5(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


class AeroToAirfoil(nn.Module):
    """Modello inverso supervisionato: condizioni aerodinamiche -> profilo alare.

    Input normalizzato:
        [log10_re, ncrit, alpha, cl, cd, cdp, cm, top_xtr, bot_xtr]

    Output normalizzato:
        100 ordinate y = 50 dorso + 50 pancia.
    """

    def __init__(self, input_dim=9, output_dim=100, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.1),

            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, aero_features):
        return self.net(aero_features)