import torch
import torch.optim as optim
import torch.nn as nn

from src.Autoencoder_m1 import Autoencoder as VAE
from src.main_data import dataloader
from src.config import WEIGHTS_FILE, WEIGHTS_FILE_PARSEC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.MSELoss()

beta = 0.0001

for epoch in range(1000):

    model.train()

    epoch_loss = 0

    for batch in dataloader:

        b = batch.to(device)

        reconstructed, mu, logvar = model(b)

        recon_loss = criterion(reconstructed, b)

        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        )

        loss = recon_loss + beta * kl_loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/200, Loss={epoch_loss/len(dataloader):.6f}\n")

    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), WEIGHTS_FILE)
        # torch.save(model.state_dict(), WEIGHTS_FILE_PARSEC)

    print(f"Epoch {epoch+1}: Loss={loss.item():.6f}\n")