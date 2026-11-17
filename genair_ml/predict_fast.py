"""
predict_fast.py — come predict.py, ma con cache del dataset.
Prima esecuzione: lenta (costruisce la cache). Dalla seconda: istantanea.

Salva inoltre il profilo generato come .dat (formato Lednicer) in
genair_geom/play/airfoils/, pronto per essere caricato da geom.airfoil.Airfoil
(es. Airfoil('airfoils/generated_<nome>.dat') negli script di play/).
"""
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.Autoencoder_m1 import AeroToAirfoil
from src.config import RESULTS_DIR, WEIGHTS_FILE_INVERSE, ROOT_DIR, POLARS_DIR, PROJECT_ROOT

PREDICT_RANDOM_SEED = 42
RUN_XFOIL = True
TARGET_ALPHA = 4.0
ALPHA_TOL = 1.0e-6

CACHE_FILE = Path(__file__).resolve().parent / "data" / "cache" / "inverse_test.pkl"
FEATURE_NAMES = ["log10_re", "ncrit", "alpha", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr"]

# Cartella .dat di genair_geom (genair_ml/ e genair_geom/ sono cartelle sorelle nella repo).
GEOM_AIRFOILS_DIR = PROJECT_ROOT.parent / "genair_geom" / "play" / "airfoils"


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


def save_generated_dat(y_profile, output_dat, x_grid, name="generated_airfoil"):
    """Scrive il profilo generato come .dat in formato Lednicer.

    È il formato letto da genair_geom (geom.airfoil.Airfoil._load): riga 1 nome,
    riga 2 numero di punti (dorso, pancia), poi il blocco dorso LE->TE e il
    blocco pancia LE->TE. Le 100 ordinate del modello (50 dorso + 50 pancia)
    vengono scritte sulla griglia coseno x_grid, in ordine di x crescente.
    """

    y = np.asarray(y_profile, dtype=float).reshape(-1)
    if y.shape[0] != 100:
        raise ValueError(f"Mi aspettavo 100 ordinate y, trovate {y.shape[0]}")

    x = np.asarray(x_grid, dtype=float)
    if x.shape[0] != 50:
        raise ValueError(f"Mi aspettavo una griglia x da 50 punti, trovati {x.shape[0]}")
    y_upper = y[:50]
    y_lower = y[50:]

    output_dat.parent.mkdir(parents=True, exist_ok=True)
    with output_dat.open("w", encoding="utf-8") as f:
        f.write(f"{name}\n")
        f.write(f"{len(x):>7.0f}. {len(x):>8.0f}.\n\n")
        for xi, yi in zip(x, y_upper):
            f.write(f"{xi: .7f} {yi: .7f}\n")
        f.write("\n")
        for xi, yi in zip(x, y_lower):
            f.write(f"{xi: .7f} {yi: .7f}\n")



def run_generated_profile_with_neuralfoil(y_profile, cond, x_grid):
    """Valuta il profilo generato con NeuralFoil (sostituto rapido di XFOIL).

    Serve solo come fallback quando l'eseguibile XFOIL non è installato:
    NeuralFoil è una rete addestrata proprio su polari XFOIL, quindi i numeri
    sono confrontabili (errore medio dichiarato su CD: ~0.4-2% vs XFOIL).
    Restituisce un dict con le stesse chiavi usate per stampare la polar XFOIL.
    """
    import neuralfoil  # già nelle dipendenze (requirements.txt)

    y = np.asarray(y_profile, dtype=float).reshape(-1)
    x = np.asarray(x_grid, dtype=float)
    # Coordinate in ordine Selig: TE dorso -> LE -> TE pancia.
    coords = np.concatenate(
        [np.column_stack([x[::-1], y[:50][::-1]]), np.column_stack([x[1:], y[50:][1:]])],
        axis=0,
    )
    out = neuralfoil.get_aero_from_coordinates(
        coordinates=coords,
        alpha=cond["alpha"],
        Re=cond["re"],
        n_crit=cond["ncrit"],
        xtr_upper=cond["top_xtr"],
        xtr_lower=cond["bot_xtr"],
        model_size="xxlarge",
    )
    keys = ["CL", "CD", "CM", "Top_Xtr", "Bot_Xtr", "analysis_confidence"]
    return {k: float(np.asarray(out[k]).reshape(-1)[0]) for k in keys}



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

    # Plot: profilo generato vs target
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

    # Analisi a valle: XFOIL se installato, altrimenti NeuralFoil.
    # Il .dat viene salvato SOLO dopo che l'analisi è andata a buon fine.
    analysis_ok = False
    if RUN_XFOIL:
        from src.run_xfoil import find_xfoil_executable, run_generated_profile_with_xfoil
        try:
            find_xfoil_executable()
        except FileNotFoundError:
            print("\nXFOIL non trovato: lo salto e valuto il profilo con NeuralFoil "
                  "(rete addestrata su polari XFOIL).")
            nf = run_generated_profile_with_neuralfoil(pred_y, cond, test_dataset.x_fixed)
            print("\nRisultato NeuralFoil sul profilo generato:")
            print(f"  CL: {nf['CL']:.6f}   CD: {nf['CD']:.6f}   CM: {nf['CM']:.6f}")
            print(f"  Top_Xtr: {nf['Top_Xtr']:.6f}   Bot_Xtr: {nf['Bot_Xtr']:.6f}")
            print(f"  analysis_confidence: {nf['analysis_confidence']:.3f}")
            print("  (per la polar XFOIL esatta: installa XFOIL e rilancia)")
            analysis_ok = True
        else:
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
            analysis_ok = True

    # Salvataggio del .dat a valle dell'analisi: il file esiste solo se il
    # profilo è stato effettivamente valutato (o se l'analisi è disattivata
    # con RUN_XFOIL = False, nel qual caso si salva direttamente).
    if analysis_ok or not RUN_XFOIL:
        dat_path = GEOM_AIRFOILS_DIR / f"generated_{meta['airfoil_name']}.dat"
        save_generated_dat(pred_y, dat_path, x_grid=test_dataset.x_fixed, name=dat_path.stem)
        print(f"\nProfilo generato salvato in: {dat_path}")
    else:
        print("\nAnalisi non eseguita: .dat non salvato.")


if __name__ == "__main__":
    main()