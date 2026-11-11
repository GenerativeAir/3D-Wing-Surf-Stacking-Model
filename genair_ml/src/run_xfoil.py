from pathlib import Path
import re
import shutil
import subprocess
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

try:
    from src.main_data import X_FIXED
except Exception:
    theta = np.linspace(0.0, np.pi, 50)
    X_FIXED = (1.0 - np.cos(theta)) / 2.0
    X_FIXED[0] = 1.0e-4


MACH_DEFAULT = 0.0
ITERATIONS_DEFAULT = 1000


def find_xfoil_executable() -> str:
    candidates = [
        shutil.which("xfoil"),
        "/opt/homebrew/bin/xfoil",
        "/usr/local/bin/xfoil",
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)

    raise FileNotFoundError("Non trovo XFOIL. Da terminale prova: which xfoil")


def y_profile_to_xfoil_dat(y_profile: np.ndarray, output_dat: Path, name: str = "generated_airfoil") -> None:
    """Scrive un .dat XFOIL partendo da 100 y: 50 dorso + 50 pancia.

    L'ordine XFOIL è: trailing edge lato superiore -> leading edge -> trailing edge lato inferiore.
    """

    y = np.asarray(y_profile, dtype=float).reshape(-1)
    if y.shape[0] != 100:
        raise ValueError(f"Mi aspettavo 100 ordinate y, trovate {y.shape[0]}")

    x = np.asarray(X_FIXED, dtype=float)
    y_upper = y[:50]
    y_lower = y[50:]

    upper = list(zip(x[::-1], y_upper[::-1]))
    lower = list(zip(x[1:], y_lower[1:]))  # evita duplicazione quasi-leading-edge
    coords = upper + lower

    output_dat.parent.mkdir(parents=True, exist_ok=True)
    with output_dat.open("w", encoding="utf-8") as f:
        f.write(f"{name}\n")
        for xi, yi in coords:
            f.write(f"{xi:.8f} {yi:.8f}\n")


def polar_has_numeric_data(polar_file: Path) -> bool:
    if not polar_file.exists() or polar_file.stat().st_size == 0:
        return False

    with polar_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                [float(p) for p in parts[:3]]
                return True
            except ValueError:
                continue
    return False

def alpha_sequence_to_target(alpha_target: float, step_abs: float = 0.5):
    alpha_start = 0.0

    if abs(alpha_target) < 1.0e-8:
        return alpha_start, alpha_target, step_abs

    step = step_abs if alpha_target > 0 else -step_abs
    return alpha_start, alpha_target, step

def run_xfoil_alpha_sequence_to_target(
    dat_file: Path,
    polar_file: Path,
    log_file: Path,
    reynolds: float,
    alpha: float,
    ncrit: float = 9.0,
    mach: float = MACH_DEFAULT,
    iterations: int = ITERATIONS_DEFAULT,
) -> None:
    xfoil_exe = find_xfoil_executable()
    work_dir = dat_file.parent
    dat_name = dat_file.name
    polar_name = polar_file.name

    if polar_file.exists():
        polar_file.unlink()
    if log_file.exists():
        log_file.unlink()

    alpha_start, alpha_end, alpha_step = alpha_sequence_to_target(
        float(alpha),
        step_abs=0.5,
    )

    commands = f"""
PLOP
G

LOAD {dat_name}

PANE
OPER
VISC {float(reynolds):.8g}
MACH {float(mach):.8g}
ITER {int(iterations)}
VPAR
N {float(ncrit):.8g}

PACC
{polar_name}

ASEQ {alpha_start:.8g} {alpha_end:.8g} {alpha_step:.8g}
PACC

QUIT
"""

    result = subprocess.run(
        [xfoil_exe],
        input=commands,
        text=True,
        capture_output=True,
        cwd=work_dir,
    )

    log_file.write_text(
        result.stdout + "\n\n--- STDERR ---\n\n" + result.stderr,
        encoding="utf-8",
        errors="ignore",
    )

    if not polar_has_numeric_data(polar_file):
        raise RuntimeError(
            "XFOIL non ha prodotto una polar numerica valida. "
            f"Controlla il log: {log_file}"
        )

def read_polar(polar_file: Path) -> pd.DataFrame:
    rows = []
    with polar_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                values = [float(p) for p in parts]
            except ValueError:
                continue
            rows.append(values)

    if not rows:
        raise RuntimeError(f"Nessun dato numerico trovato in {polar_file}")

    max_cols = max(len(row) for row in rows)
    normalized_rows = [row + [None] * (max_cols - len(row)) for row in rows]

    if max_cols >= 7:
        columns = ["Alpha", "Cl", "Cd", "Cdp", "Cm", "Top_Xtr", "Bot_Xtr"]
        columns += [f"Extra_{i}" for i in range(8, max_cols + 1)]
    elif max_cols == 5:
        columns = ["Alpha", "Cl", "Cd", "Cdp", "Cm"]
    elif max_cols == 3:
        columns = ["Alpha", "Cl", "Cm"]
    else:
        columns = [f"Col_{i}" for i in range(1, max_cols + 1)]

    return pd.DataFrame(normalized_rows, columns=columns[:max_cols])


def extract_converged_values_from_log(log_file: Path) -> pd.DataFrame:
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"a\s*=\s*([+-]?\d+(?:\.\d+)?)\s+CL\s*=\s*([+-]?\d+(?:\.\d+)?)"
        r".*?"
        r"Cm\s*=\s*([+-]?\d+(?:\.\d+)?)\s+CD\s*=\s*([+-]?\d+(?:\.\d+)?)"
        r"\s*=>\s*CDf\s*=\s*([+-]?\d+(?:\.\d+)?)\s+CDp\s*=\s*([+-]?\d+(?:\.\d+)?)",
        re.DOTALL,
    )

    matches = pattern.findall(text)
    if not matches:
        return pd.DataFrame()

    rows = []
    for alpha, cl, cm, cd, cdf, cdp in matches:
        rows.append(
            {
                "Alpha": float(alpha),
                "Cl_log": float(cl),
                "Cd_log": float(cd),
                "Cdf_log": float(cdf),
                "Cdp_log": float(cdp),
                "Cm_log": float(cm),
            }
        )
    return pd.DataFrame(rows).groupby("Alpha", as_index=False).last()


def fix_from_xfoil_log(polar: pd.DataFrame, log_file: Path) -> pd.DataFrame:
    log_df = extract_converged_values_from_log(log_file)
    if log_df.empty:
        print("Warning: non ho trovato CDp/CDf nel log XFOIL. Lascio la polar invariata.")
        return polar

    polar = polar.copy()
    if "Cdf" not in polar.columns:
        polar["Cdf"] = None

    for idx, row in polar.iterrows():
        alpha = float(row["Alpha"])
        nearest_idx = (log_df["Alpha"] - alpha).abs().idxmin()
        log_row = log_df.loc[nearest_idx]
        if abs(float(log_row["Alpha"]) - alpha) < 1.0e-6:
            if "Cdp" in polar.columns:
                polar.at[idx, "Cdp"] = float(log_row["Cdp_log"])
            polar.at[idx, "Cdf"] = float(log_row["Cdf_log"])
    return polar


def run_generated_profile_with_xfoil(
    y_profile: np.ndarray,
    reynolds: float,
    alpha: float,
    ncrit: float,
    work_dir: Optional[Path] = None,
    name: str = "generated_airfoil",
) -> pd.DataFrame:
    if work_dir is None:
        work_dir = PROJECT_ROOT / "results" / "xfoil_generated"
    work_dir.mkdir(parents=True, exist_ok=True)

    dat_file = work_dir / f"{name}.dat"
    polar_file = work_dir / f"polar_{name}.txt"
    log_file = work_dir / f"xfoil_log_{name}.txt"

    y_profile_to_xfoil_dat(y_profile, dat_file, name=name)

    run_xfoil_alpha_sequence_to_target(
        dat_file=dat_file,
        polar_file=polar_file,
        log_file=log_file,
        reynolds=reynolds,
        alpha=alpha,
        ncrit=ncrit,
    )

    polar = read_polar(polar_file)
    polar = fix_from_xfoil_log(polar, log_file)

    idx = (polar["Alpha"] - float(alpha)).abs().idxmin()
    polar_target = polar.loc[[idx]].copy()

    polar.to_csv(work_dir / f"polar_sequence_{name}.csv", index=False)
    polar_target.to_csv(work_dir / f"polar_{name}.csv", index=False)

    return polar_target