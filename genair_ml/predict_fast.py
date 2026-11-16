"""
predict_fast.py — come predict.py, ma con cache del dataset.
Prima esecuzione: lenta (costruisce la cache). Dalla seconda: istantanea.
"""
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.Autoencoder_m1 import AeroToAirfoil
from src.config import RESULTS_DIR, WEIGHTS_FILE_INVERSE, ROOT_DIR, POLARS_DIR
from src.run_xfoil import run_generated_profile_with_xfoil

PREDICT_RANDOM_SEED = 42
RUN_XFOIL = True
TARGET_ALPHA = 4.0
ALPHA_TOL = 1.0e-6

CACHE_FILE = Path(__file__).resolve().parent / "data" / "cache" / "inverse_test.pkl"
FEATURE_NAMES = ["log10_re", "ncrit", "alpha", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr"]


class CachedInverseDataset:
    """Copia minimale di AirfoilInverseDataset, senza import pesanti."""

    def __init__(self, samples, scalers):
        self.samples = list(samples)
        self.scalers = scalers
        self.target_dim = int(self.samples[0]["coords"].shape[0])
        theta = np.linspace(0.0, np.pi, 50)
        x = (1.0 - np.cos(theta)) / 2.0
        x[0] = 1.0e-4
        self.x_fixed = x.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = (s["features"] - self.scalers["x_mean"]) / self.scalers["x_std"]
        y = (s["coords"] - self.scalers["y_mean"]) / self.scalers["y_std"]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), s["meta"]

    def denormalize_coords(self, y_norm):
        if isinstance(y_norm, torch.Tensor):
            y_norm = y_norm.detach().cpu().numpy()
        return y_norm * self.scalers["y_std"] + self.scalers["y_mean"]

    def denormalize_features(self, x_norm):
        if isinstance(x_norm, torch.Tensor):
            x_norm = x_norm.detach().cpu().numpy()
        x_raw = x_norm * self.scalers["x_std"] + self.scalers["x_mean"]
        values = {name: float(v) for name, v in zip(FEATURE_NAMES, x_raw)}
        values["re"] = float(10.0 ** values["log10_re"])
        values["is_n5"] = 1.0 if abs(values["ncrit"] - 5.0) < abs(values["ncrit"] - 9.0) else 0.0
        return values


def _dataset_signature():
    """Firma leggera (solo nomi cartelle): se i dati cambiano, cache ricostruita."""
    geom = {p.name for p in ROOT_DIR.iterdir() if p.is_dir()}
    pol = {p.name for p in POLARS_DIR.iterdir() if p.is_dir()}
    return sorted(geom & pol)


def load_or_build_test_dataset():
    signature = _dataset_signature()
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            payload = pickle.load(f)
        if payload["signature"] == signature:
            print(f"[cache] test set caricato da {CACHE_FILE} ({len(payload['samples'])} righe)")
            return CachedInverseDataset(payload["samples"], payload["scalers"])
        print("[cache] dataset cambiato: ricostruisco...")

    # Percorso lento, SOLO la prima volta: pipeline originale.
    from src.main_data import get_inverse_data  # import pigro
    _, _, _, _, test_dataset = get_inverse_data()
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"signature": signature,
                     "samples": test_dataset.samples,
                     "scalers": test_dataset.scalers}, f)
    print(f"[cache] salvata in {CACHE_FILE}")
    return CachedInverseDataset(test_dataset.samples, test_dataset.scalers)


def choose_random_sample_with_alpha(test_dataset, target_alpha,
                                    seed=PREDICT_RANDOM_SEED, alpha_tol=ALPHA_TOL):
    candidates = [i for i, s in enumerate(test_dataset.samples)
                  if np.isclose(float(s["meta"]["alpha"]), target_alpha, atol=alpha_tol)]
    if not candidates:
        available = sorted({round(float(s["meta"]["alpha"]), 6) for s in test_dataset.samples})
        raise ValueError(f"Nessuna riga con alpha={target_alpha}. Alpha disponibili: {available}")
    return int(np.random.default_rng(seed).choice(candidates))


def main():
    test_dataset = load_or_build_test_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(WEIGHTS_FILE_INVERSE, map_location=device, weights_only=False)
    model = AeroToAirfoil(
        input_dim=checkpoint.get("input_dim", len(FEATURE_NAMES)),
        output_dim=checkpoint.get("output_dim", test_dataset.target_dim),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample_index = choose_random_sample_with_alpha(test_dataset, TARGET_ALPHA)
    aero_x_norm, true_y_norm, meta = test_dataset[sample_index]
    with torch.no_grad():
        pred_y_norm = model(aero_x_norm.unsqueeze(0).to(device)).squeeze(0).cpu()
    pred_y = test_dataset.denormalize_coords(pred_y_norm)
    true_y = test_dataset.denormalize_coords(true_y_norm)
    cond = test_dataset.denormalize_features(aero_x_norm)

    print(f"\nSample con alpha={TARGET_ALPHA} | sample_index: {sample_index}")
    print("\nCondizioni aerodinamiche in input alla rete:")
    for key in ["re", "ncrit", "alpha", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr"]:
        print(f"  {key}: {cond[key]}")
    print(f"\nairfoil_name: {meta['airfoil_name']}\npolar_file: {meta['polar_file']}")

    x = test_dataset.x_fixed
    plt.figure(figsize=(8, 4))
    plt.plot(x, pred_y[:50], label="Dorso generato")
    plt.plot(x, pred_y[50:], label="Pancia generata")
    plt.plot(x, true_y[:50], "--", label="Dorso target")
    plt.plot(x, true_y[50:], "--", label="Pancia target")
    plt.title("Inverse model: profilo target vs profilo generato")
    plt.legend()
    plt.grid(True)
    plt.show()

    if RUN_XFOIL:
        print("\nEseguo XFOIL sul profilo generato...")
        polar = run_generated_profile_with_xfoil(
            y_profile=pred_y,
            reynolds=round(cond["re"]),
            ncrit=round(cond["ncrit"]),
            alpha=cond["alpha"],
            work_dir=RESULTS_DIR / "xfoil_inverse_test",
            name=f"generated_{meta['airfoil_name']}",
        )
        print("\nRisultato XFOIL sul profilo generato:")
        print(polar.to_string(index=False))


if __name__ == "__main__":
    main()