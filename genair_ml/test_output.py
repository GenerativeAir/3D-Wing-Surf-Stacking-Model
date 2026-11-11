from src.Autoencoder_m1 import Autoencoder as VAE

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.main_data import dataloader_v as dt
from src.point_extract import min_val, max_val
from src.config import WEIGHTS_FILE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE()

model.load_state_dict(torch.load(WEIGHTS_FILE))

model.eval()

with torch.no_grad():

    batch = next(iter(dt))

    sample = batch[0].unsqueeze(0).to(device)

    mu, logvar = model.encode(sample)
    reconstructed = model.decode(mu)

    sample_denorm = sample * (max_val - min_val) + min_val

    reconstructed_denorm = reconstructed * (max_val - min_val) + min_val

    original_np = sample_denorm.squeeze().detach().numpy()

    reconstructed_np = reconstructed_denorm.squeeze().detach().numpy()

    dorso_orig = original_np[:50]
    pancia_orig = original_np[50:]

    dorso_rec = reconstructed_np[:50]
    pancia_rec = reconstructed_np[50:]


def parsec_curve(coeffs, X):

    powers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

    Z = np.sum(coeffs * X[:, None]**powers, axis=1)

    return Z


theta = np.linspace(0, np.pi, 50)
X = (1 - np.cos(theta)) / 2
X[0] = 1e-4

plt.figure(figsize=(8, 4))

plt.plot(X, dorso_rec, label='Dorso ricostruito')
plt.plot(X, pancia_rec, label='Pancia ricostruita')

plt.plot(X, dorso_orig, '--', label='Dorso originale')
plt.plot(X, pancia_orig, '--', label='Pancia originale')

plt.title("Confronto Airfoil: originale vs ricostruito")

plt.legend()

plt.grid(True)

plt.show()