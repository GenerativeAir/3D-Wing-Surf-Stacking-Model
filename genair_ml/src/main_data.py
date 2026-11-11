from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import ROOT_DIR, POLARS_DIR
from src.point_extract import dataset_training as old_dataset_training
from src.point_extract import dataset_validation as old_dataset_validation


# ============================================================
# CONFIG COMUNE
# ============================================================

OLD_BATCH_SIZE = 32

INVERSE_BATCH_SIZE = 128
INVERSE_TRAIN_FRACTION = 0.80
INVERSE_RANDOM_SEED = 42
EPS = 1.0e-8

FEATURE_NAMES = [
    "log10_re",
    "ncrit",
    "alpha",
    "cl",
    "cd",
    "cdp",
    "cm",
    "top_xtr",
    "bot_xtr",
]

POLAR_VECTOR_NAMES = ["alpha", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr"]
FLOAT_RE = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"

class AirfoilDataset(Dataset):
    """Dataset geometrico vecchio: restituisce solo il profilo alare normalizzato."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


tensor_t = torch.tensor(old_dataset_training, dtype=torch.float32)
tensor_v = torch.tensor(old_dataset_validation, dtype=torch.float32)

print("Old training dataset:", tensor_t.shape)
print("Old validation dataset:", tensor_v.shape)

airfoil_dataset = AirfoilDataset(tensor_t)
validation_dataset = AirfoilDataset(tensor_v)

# Nomi storici: NON cambiarli, servono ai vecchi main_trainer.py e test_output.py.
dataloader = DataLoader(airfoil_dataset, batch_size=OLD_BATCH_SIZE, shuffle=True)
dataloader_v = DataLoader(validation_dataset, batch_size=OLD_BATCH_SIZE, shuffle=False)


# ============================================================
# FUNZIONI GEOMETRICHE RIUSABILI
# ============================================================

def fixed_x_grid(n_points_per_side: int = 50) -> np.ndarray:
    theta = np.linspace(0.0, np.pi, n_points_per_side)
    x = (1.0 - np.cos(theta)) / 2.0
    x[0] = 1.0e-4
    return x.astype(np.float32)


X_FIXED = fixed_x_grid(50)


def parsec_to_points(coeffs: Sequence[float], x: np.ndarray) -> np.ndarray:
    powers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
    coeffs_np = np.asarray(coeffs, dtype=np.float32)
    return np.sum(coeffs_np * x[:, None] ** powers, axis=1).astype(np.float32)


def _parse_float_list(text: str) -> List[float]:
    return [float(v) for v in re.findall(FLOAT_RE, text)]


def read_parsec_airfoil_y(airfoil_dir: Path, x_grid: np.ndarray = X_FIXED) -> np.ndarray:
    """Legge *_PARSEC_equations.txt e restituisce 100 ordinate y: 50 dorso + 50 pancia."""

    candidates = sorted(airfoil_dir.glob("*_PARSEC_equations.txt"))
    if not candidates:
        candidates = sorted(p for p in airfoil_dir.glob("*.txt") if "PARSEC" in p.name.upper())
    if not candidates:
        raise FileNotFoundError(f"Nessun file PARSEC trovato in {airfoil_dir}")

    text = candidates[0].read_text(encoding="utf-8", errors="ignore")

    dorso = re.search(r"Dorso\s*\[c1\.\.c6\]:\s*\[(.*?)\]", text, re.IGNORECASE | re.DOTALL)
    pancia = re.search(r"Pancia\s*\[c1\.\.c6\]:\s*\[(.*?)\]", text, re.IGNORECASE | re.DOTALL)
    if not dorso or not pancia:
        raise ValueError(f"Dorso/Pancia PARSEC non trovati in {candidates[0]}")

    dorso_vals = _parse_float_list(dorso.group(1))
    pancia_vals = _parse_float_list(pancia.group(1))
    if len(dorso_vals) != 6 or len(pancia_vals) != 6:
        raise ValueError(f"Mi aspettavo 6 coefficienti per dorso/pancia in {candidates[0]}")

    y_dorso = parsec_to_points(dorso_vals, x_grid)
    y_pancia = parsec_to_points(pancia_vals, x_grid)
    return np.concatenate([y_dorso, y_pancia]).astype(np.float32)

def _extract_vector(text: str, name: str) -> np.ndarray:
    match = re.search(rf"\b{re.escape(name)}\b\s*=\s*\[(.*?)\]", text, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Vettore {name} non trovato")
    values = _parse_float_list(match.group(1))
    if not values:
        raise ValueError(f"Vettore {name} vuoto")
    return np.asarray(values, dtype=np.float32)


def _extract_reynolds(text: str, polar_file: Path) -> float:
    match = re.search(r"\bRE\b\s*=\s*(" + FLOAT_RE + r")", text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Fallback: xf-airfoil-name-500000.txt oppure xf-airfoil-name-500000-n5.txt.
    match = re.search(r"-(\d+)(?:-n5)?$", polar_file.stem, re.IGNORECASE)
    if match:
        return float(match.group(1))

    raise ValueError(f"RE non trovato in {polar_file}")


def _extract_ncrit_from_name(polar_file: Path) -> float:
    # Suffisso -n5 = Ncrit 5. Senza suffisso, di solito Ncrit 9.
    if re.search(r"-n5$", polar_file.stem, re.IGNORECASE):
        return 5.0
    return 9.0


def read_polar_rows(polar_file: Path) -> List[Dict[str, float]]:
    """Legge un file polar vettoriale e lo espande in una lista di righe, una per alpha."""

    text = polar_file.read_text(encoding="utf-8", errors="ignore")
    reynolds = _extract_reynolds(text, polar_file)
    ncrit = _extract_ncrit_from_name(polar_file)

    vectors = {name: _extract_vector(text, name) for name in POLAR_VECTOR_NAMES}
    lengths = {name: len(values) for name, values in vectors.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Lunghezze vettori polar incoerenti in {polar_file}: {lengths}")

    rows: List[Dict[str, float]] = []
    n = next(iter(lengths.values()))

    for i in range(n):
        row = {name: float(vectors[name][i]) for name in POLAR_VECTOR_NAMES}
        row["re"] = float(reynolds)
        row["ncrit"] = float(ncrit)
        row["is_n5"] = 1.0 if ncrit == 5.0 else 0.0
        rows.append(row)

    return rows


def row_to_features(row: Dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            np.log10(row["re"]),
            row["ncrit"],
            row["alpha"],
            row["cl"],
            row["cd"],
            row["cdp"],
            row["cm"],
            row["top_xtr"],
            row["bot_xtr"],
        ],
        dtype=np.float32,
    )


def common_airfoil_names(root_dir: Path = ROOT_DIR, polars_dir: Path = POLARS_DIR) -> List[str]:
    geom_names = {p.name for p in root_dir.iterdir() if p.is_dir()}
    polar_names = {p.name for p in polars_dir.iterdir() if p.is_dir()}
    return sorted(geom_names & polar_names)


def split_names(
    names: Sequence[str],
    train_fraction: float = INVERSE_TRAIN_FRACTION,
    seed: int = INVERSE_RANDOM_SEED,
) -> Tuple[List[str], List[str], List[str]]:
    names_np = np.asarray(list(names), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(names_np)

    if len(names_np) <= 1:
        return list(names_np), [], []

    test_airfoil_names = [names_np[-1]]
    remaining_names = names_np[:-1]

    if len(remaining_names) <= 1:
        return list(remaining_names), [], test_airfoil_names

    n_train = int(round(len(remaining_names) * train_fraction))
    n_train = min(max(n_train, 1), len(remaining_names) - 1)

    train_airfoil_names = list(remaining_names[:n_train])
    validation_airfoil_names = list(remaining_names[n_train:])

    return train_airfoil_names, validation_airfoil_names, test_airfoil_names


def build_samples(
    names: Iterable[str],
    root_dir: Path = ROOT_DIR,
    polars_dir: Path = POLARS_DIR,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for airfoil_name in names:
        geom_dir = root_dir / airfoil_name
        polar_dir = polars_dir / airfoil_name

        try:
            y_coords = read_parsec_airfoil_y(geom_dir)
        except Exception as exc:
            skipped.append(f"{airfoil_name}: geometry error: {exc}")
            continue

        polar_files = sorted(polar_dir.glob("*.txt"))
        if not polar_files:
            skipped.append(f"{airfoil_name}: nessun polar txt")
            continue

        for polar_file in polar_files:
            try:
                polar_rows = read_polar_rows(polar_file)
            except Exception as exc:
                skipped.append(f"{airfoil_name}/{polar_file.name}: polar error: {exc}")
                continue

            for row in polar_rows:
                samples.append(
                    {
                        "features": row_to_features(row),
                        "coords": y_coords.copy(),
                        "meta": {
                            "airfoil_name": airfoil_name,
                            "polar_file": str(polar_file),
                            **row,
                        },
                    }
                )

    if skipped:
        print("Warning: alcuni elementi sono stati saltati:")
        for msg in skipped[:20]:
            print("  -", msg)
        if len(skipped) > 20:
            print(f"  ... altri {len(skipped) - 20} elementi saltati")

    return samples


def fit_scalers(train_samples: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not train_samples:
        raise ValueError("train_samples vuoto: impossibile calcolare gli scaler")

    x = np.stack([sample["features"] for sample in train_samples]).astype(np.float32)
    y = np.stack([sample["coords"] for sample in train_samples]).astype(np.float32)

    return {
        "x_mean": x.mean(axis=0).astype(np.float32),
        "x_std": (x.std(axis=0) + EPS).astype(np.float32),
        "y_mean": y.mean(axis=0).astype(np.float32),
        "y_std": (y.std(axis=0) + EPS).astype(np.float32),
    }


class AirfoilInverseDataset(Dataset):
    """Dataset inverso: restituisce input aerodinamico, profilo target e metadata."""

    def __init__(self, samples: Sequence[Dict[str, Any]], scalers: Dict[str, np.ndarray]):
        self.samples = list(samples)
        self.scalers = scalers
        if not self.samples:
            raise ValueError("Dataset inverso vuoto")
        self.target_dim = int(self.samples[0]["coords"].shape[0])
        self.x_fixed = X_FIXED.copy()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        x = (sample["features"] - self.scalers["x_mean"]) / self.scalers["x_std"]
        y = (sample["coords"] - self.scalers["y_mean"]) / self.scalers["y_std"]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            sample["meta"],
        )

    def denormalize_coords(self, y_norm: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(y_norm, torch.Tensor):
            y_norm = y_norm.detach().cpu().numpy()
        return y_norm * self.scalers["y_std"] + self.scalers["y_mean"]

    def denormalize_features(self, x_norm: np.ndarray | torch.Tensor) -> Dict[str, float]:
        if isinstance(x_norm, torch.Tensor):
            x_norm = x_norm.detach().cpu().numpy()

        x_raw = x_norm * self.scalers["x_std"] + self.scalers["x_mean"]
        values = {name: float(value) for name, value in zip(FEATURE_NAMES, x_raw)}
        values["re"] = float(10.0 ** values["log10_re"])
        values["is_n5"] = 1.0 if abs(values["ncrit"] - 5.0) < abs(values["ncrit"] - 9.0) else 0.0
        return values


def collate_keep_meta(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), list(metas)


# ============================================================
# COSTRUZIONE DATASET INVERSO A IMPORT
# ============================================================
def random_subset_samples(
    samples: Sequence[Dict[str, Any]],
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    samples = list(samples)

    if len(samples) <= max_samples:
        return samples

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), size=max_samples, replace=False)

    return [samples[i] for i in idx]

def get_inverse_data():
    all_airfoil_names = common_airfoil_names()
    train_airfoil_names, validation_airfoil_names, test_airfoil_names = split_names(all_airfoil_names)

    inverse_train_samples = build_samples(train_airfoil_names)
    inverse_validation_samples = build_samples(validation_airfoil_names)
    inverse_test_samples = build_samples(test_airfoil_names)

    inverse_train_samples = random_subset_samples(
        inverse_train_samples,
        max_samples=20000,
        seed=INVERSE_RANDOM_SEED,
    )

    inverse_validation_samples = random_subset_samples(
        inverse_validation_samples,
        max_samples=5000,
        seed=INVERSE_RANDOM_SEED + 1,
    )

    inverse_scalers = fit_scalers(inverse_train_samples)

    inverse_train_dataset = AirfoilInverseDataset(inverse_train_samples, inverse_scalers)
    inverse_validation_dataset = AirfoilInverseDataset(inverse_validation_samples, inverse_scalers)
    inverse_test_dataset = AirfoilInverseDataset(inverse_test_samples, inverse_scalers)

    inverse_dataloader = DataLoader(
        inverse_train_dataset,
        batch_size=INVERSE_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_keep_meta,
    )

    inverse_dataloader_v = DataLoader(
        inverse_validation_dataset,
        batch_size=INVERSE_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_keep_meta,
    )

    print(f"Airfoil comuni ROOT/POLARS: {len(all_airfoil_names)}")
    print(f"Profili inverse train: {len(train_airfoil_names)} | righe train: {len(inverse_train_dataset)}")
    print(f"Profili inverse validation: {len(validation_airfoil_names)} | righe validation: {len(inverse_validation_dataset)}")
    print(f"Input aerodinamico inverse: {len(FEATURE_NAMES)} feature -> {FEATURE_NAMES}")
    print(f"Target geometrico inverse: {inverse_train_dataset.target_dim} ordinate y")

    return (
        inverse_dataloader,
        inverse_dataloader_v,
        inverse_train_dataset,
        inverse_validation_dataset,
        inverse_test_dataset,
    )