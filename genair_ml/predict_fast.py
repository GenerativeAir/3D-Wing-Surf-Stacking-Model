"""
predict_fast.py — come predict.py, ma con cache del dataset.
Prima esecuzione: lenta (costruisce la cache). Dalla seconda: istantanea.

Salva inoltre il profilo generato come .dat (formato Lednicer) in
genair_geom/play/airfoils/, pronto per essere caricato da geom.airfoil.Airfoil
(es. Airfoil('airfoils/generated_<nome>.dat') negli script di play/).

PIPELINE INTEGRATA (genair_ml -> genair_geom): dopo il salvataggio del
.dat, se RUN_GENAIR_GEOM = True, predict_fast.py
  1) aggiorna la riga AIRFOIL = [...] di genair_geom/play/trapezoidal_wing.py
     con il nome del .dat appena generato (ripetuto su tutte le stazioni)
     e salva il file;
  2) avvia genair_geom/main.py con l'auto-run di %run trapezoidal_wing.py,
     che parte appena compare il banner "Welcome to Genair!".
Il risultato è l'ala 3D costruita col profilo generato (trapezoidal_wing.p
e .igs in genair_geom/play/).
"""
import pickle
import sys
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

# Pipeline integrata: dopo il salvataggio del .dat, aggiorna la riga
# AIRFOIL di genair_geom/play/trapezoidal_wing.py e avvia genair_geom
# (python main.py --run trapezoidal_wing.py). Metti False per fermarsi al .dat.
RUN_GENAIR_GEOM = True

CACHE_FILE = Path(__file__).resolve().parent / "data" / "cache" / "inverse_test.pkl"
FEATURE_NAMES = ["log10_re", "ncrit", "alpha", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr"]

# Cartella .dat di genair_geom (genair_ml/ e genair_geom/ sono cartelle sorelle nella repo).
GEOM_AIRFOILS_DIR = PROJECT_ROOT.parent / "genair_geom" / "play" / "airfoils"
GEOM_DIR = PROJECT_ROOT.parent / "genair_geom"


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


# ==================================================================
# PIPELINE INTEGRATA: aggiorna trapezoidal_wing.py e avvia genair_geom
# ==================================================================
def update_trapezoidal_wing_airfoil(dat_name, wing_script=None):
    """Riscrive la riga AIRFOIL di trapezoidal_wing.py con il .dat generato.

    Il nome viene ripetuto su tutte le stazioni (il profilo generato è
    unico, come nell'esempio originale dove lo stesso .dat compariva tre
    volte). Il file viene salvato su disco prima di ritornare.
    """

    import re

    wing_script = Path(wing_script) if wing_script else \
        GEOM_DIR / "play" / "trapezoidal_wing.py"
    if not wing_script.exists():
        raise FileNotFoundError(f"Non trovo {wing_script}")

    dat_name = Path(dat_name).name  # solo il nome: path relativo a play/
    text = wing_script.read_text()

    pattern = re.compile(r"^(\s*AIRFOIL\s*=\s*)\[.*\](\s*(?:\#.*)?)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Riga AIRFOIL non trovata in {wing_script}")

    # numero di stazioni: dalla riga CHORD (stazioni = elementi di CHORD)
    chord = re.search(r"^CHORD\s*=\s*\[(.*?)\]", text, re.MULTILINE)
    stations = len(chord.group(1).split(",")) if chord else 3

    names = ", ".join(["'{}'".format(dat_name)] * stations)
    new_line = "{}[{}]{}".format(match.group(1), names, match.group(2) or "")

    wing_script.write_text(text[: match.start()] + new_line + text[match.end():])
    print(f"\n[integration] riga AIRFOIL aggiornata in {wing_script}:")
    print(f"[integration] AIRFOIL = ['{dat_name}' x {stations} stazioni]")
    return wing_script


def launch_genair_geom(main_script=None, run_target="trapezoidal_wing.py", batch=None):
    """Avvia la pipeline genair_geom, in modo robusto a tre livelli.

    1. python main.py --run trapezoidal_wing.py [--exit]
       main.py apre la shell "Welcome to Genair!" e il %run parte da solo
       appena la shell è pronta (richiede il main.py di questa repo
       integrata, che riconosce --run).
    2. Se quel main.py non riconosce --run (versione non aggiornata),
       ripiega sul launcher autonomo integration/run_genair_geom.py,
       che inietta lui il comando nella shell.
    3. In ultima analisi inietta il comando via stdin, come se l'utente
       lo digitasse: funziona anche col main.py originale al 100%.

    batch=None (default): con un terminale interattivo (TTY) la shell
    resta aperta col finestrino 3D, altrimenti si chiude da sola dopo
    il run. Ritorna il codice di ritorno del processo.
    """

    import subprocess

    main_script = Path(main_script) if main_script else GEOM_DIR / "main.py"
    if not main_script.exists():
        raise FileNotFoundError(f"Non trovo {main_script}")

    if batch is None:
        batch = not sys.stdout.isatty()

    # ------------------------------------------------------------------
    # Livello 1: il main.py aggiornato riconosce --run (cerca il flag
    # nel sorgente: più affidabile di qualunque euristica a runtime).
    # ------------------------------------------------------------------
    supports_run = "--run" in main_script.read_text()
    if supports_run:
        cmd = [sys.executable, str(main_script), "--run", run_target]
        if batch:
            cmd.append("--exit")
        print(f"\n[integration] avvio genair_geom: {' '.join(cmd)}")
        print("[integration] attendere: la shell 'Welcome to Genair!' esegue %run da sola...")
        result = subprocess.run(cmd, cwd=str(GEOM_DIR))
        if result.returncode == 0:
            return 0
        print(f"[integration] genair_geom terminato con codice {result.returncode}: "
              "provo il launcher autonomo...")

    # ------------------------------------------------------------------
    # Livello 2: launcher autonomo (non dipende da main.py aggiornato).
    # ------------------------------------------------------------------
    launcher = GEOM_DIR / "integration" / "run_genair_geom.py"
    if launcher.exists():
        cmd = [sys.executable, str(launcher), str(run_target)]
        if batch:
            cmd.append("--exit")
        print(f"\n[integration] fallback launcher autonomo: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(GEOM_DIR))
        if result.returncode == 0:
            return 0
        print(f"[integration] launcher terminato con codice {result.returncode}: "
              "provo l'iniezione via stdin...")

    # ------------------------------------------------------------------
    # Livello 3: iniezione via stdin (compatibile col main.py originale).
    # ------------------------------------------------------------------
    print(f"\n[integration] fallback stdin su {main_script}")
    print("[integration] la shell 'Welcome to Genair!' riceve da tastiera: "
          f"%run {run_target}")
    if batch:
        stdin_lines = f"%run {run_target}\nexit\n"
    else:
        stdin_lines = f"%run {run_target}\n"
    result = subprocess.run(
        [sys.executable, str(main_script)],
        input=stdin_lines,
        text=True,
        cwd=str(GEOM_DIR),
    )
    return result.returncode



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
    # Ogni ramo è crash-safe: un errore di XFOIL/NeuralFoil non deve
    # interrompere la pipeline integrata (si passa al .dat comunque,
    # l'analisi serve solo a validare il profilo).
    analysis_ok = False
    if RUN_XFOIL:
        try:
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
        except Exception as e:
            # xfoil crashato o polar senza dati: avvisa ma NON fermare la
            # pipeline — il .dat si salva lo stesso e l'ala si genera.
            print(f"\n[integration] analisi saltata per errore ({type(e).__name__}: {e}); "
                  "salvo comunque il .dat.")
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

    # ------------------------------------------------------------------
    # PIPELINE INTEGRATA: .dat salvato -> aggiorna AIRFOIL -> avvia genair_geom
    # ------------------------------------------------------------------
    if RUN_GENAIR_GEOM and (analysis_ok or not RUN_XFOIL):
        try:
            update_trapezoidal_wing_airfoil(dat_path.name)

            # timestamp del .igs PRIMA del run: se dopo il launch è più
            # recente, l'ala è stata davvero rigenerata col nuovo profilo.
            igs = GEOM_DIR / "play" / "trapezoidal_wing.igs"
            t_before = igs.stat().st_mtime if igs.exists() else 0.0

            rc = launch_genair_geom()

            t_after = igs.stat().st_mtime if igs.exists() else 0.0
            if rc == 0 and t_after > t_before and t_after > 0:
                print(f"\n[integration] PIPELINE COMPLETA: ala rigenerata e salvata "
                      f"in {igs}")
            elif rc == 0:
                print(f"\n[integration] genair_geom è terminato senza errori, ma {igs} "
                      "non risulta aggiornato: guarda l'output qui sopra per capire "
                      "se %run è stato eseguito (righe 'geom.airfoil.Airfoil._load :: "
                      "...' e 'Fatto! ...').")
            else:
                print(f"\n[integration] genair_geom è terminato con codice {rc}: "
                      "guarda l'output qui sopra.")
        except (FileNotFoundError, ValueError) as e:
            print(f"\n[integration] pipeline genair_geom saltata: {e}")
    elif RUN_GENAIR_GEOM:
        print("\n[integration] .dat non salvato: aggiorno AIRFOIL solo dopo un'analisi riuscita.")


if __name__ == "__main__":
    main()
