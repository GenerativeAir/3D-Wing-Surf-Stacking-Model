from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import neuralfoil as nf

from src.Autoencoder_m1 import AeroToAirfoil
from src.config import RESULTS_DIR, WEIGHTS_FILE_INVERSE, WEIGHTS_FILE_PRETRAINED
from src.main_data import FEATURE_NAMES, get_inverse_data


# ============================================================
# FINE-TUNING CONFIG
# ============================================================

# Usiamo NeuralFoil come evaluator veloce. XFOIL resta per predict.py / validazione finale.
NEURALFOIL_MODEL_SIZE = "large"

# SPSA = Simultaneous Perturbation Stochastic Approximation.
# E' un metodo gradient-free: stima una direzione di aggiornamento usando due valutazioni
# black-box della loss, senza fare backpropagation attraverso NeuralFoil.
FINETUNE_STEPS = 80
BLACKBOX_BATCH_SIZE = 8
EVAL_EVERY = 5
EVAL_SAMPLES = 64
RANDOM_SEED = 42

# Aggiorniamo solo l'ultimo Linear layer del modello per evitare una ricerca gradient-free
# su troppe decine/centinaia di migliaia di pesi.
FINETUNE_ONLY_LAST_LINEAR = True

SPSA_SIGMA = 2.0e-4
SPSA_LR = 5.0e-4
SPSA_GRAD_CLIP_NORM = 10.0

EPS = 1.0e-12

# Pesi della loss aerodinamica.
# NeuralFoil standalone restituisce CL, CD, CM, Top_Xtr, Bot_Xtr.
# CDp non e' sempre disponibile: se NeuralFoil non lo restituisce, il termine cdp viene saltato.
W_CL = 1.0
W_LOG_CD = 2.0
W_LOG_CDP = 0.0
W_CM = 0.5
W_TOP_XTR = 0.25
W_BOT_XTR = 0.25

# Penalizzazioni geometriche per evitare profili rumorosi/non fisici.
W_SMOOTH = 0.05
W_THICKNESS = 5.0
W_ANCHOR = 0.02
W_CONFIDENCE = 0.25
MIN_ANALYSIS_CONFIDENCE = 0.50

LOG_FILE = RESULTS_DIR / "finetune_inverse" / "finetune_log.csv"


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def y_profile_to_selig_coordinates(y_profile: np.ndarray, x_fixed: np.ndarray) -> np.ndarray:
    """Converte 100 ordinate y in coordinate Selig per NeuralFoil.

    Input interno:
        y[:50]  = dorso, su x_fixed da leading edge a trailing edge
        y[50:]  = pancia, su x_fixed da leading edge a trailing edge

    Output Selig:
        trailing edge superiore -> leading edge -> trailing edge inferiore
    """

    y = np.asarray(y_profile, dtype=float).reshape(-1)
    if y.shape[0] != 100:
        raise ValueError(f"Mi aspettavo 100 ordinate y, trovate {y.shape[0]}")

    x = np.asarray(x_fixed, dtype=float).reshape(-1)
    if x.shape[0] != 50:
        raise ValueError(f"Mi aspettavo 50 ascisse x, trovate {x.shape[0]}")

    y_upper = y[:50]
    y_lower = y[50:]

    upper = np.column_stack([x[::-1], y_upper[::-1]])
    lower = np.column_stack([x[1:], y_lower[1:]])  # evita duplicazione del leading edge

    return np.vstack([upper, lower]).astype(float)


def smoothness_loss_np(y_profile: np.ndarray) -> float:
    y = np.asarray(y_profile, dtype=float).reshape(-1)
    upper = y[:50]
    lower = y[50:]

    d2_upper = np.diff(upper, n=2)
    d2_lower = np.diff(lower, n=2)

    return float(np.mean(d2_upper ** 2) + np.mean(d2_lower ** 2))


def thickness_violation_loss_np(y_profile: np.ndarray, min_thickness: float = 1.0e-4) -> float:
    y = np.asarray(y_profile, dtype=float).reshape(-1)
    upper = y[:50]
    lower = y[50:]

    # Penalizza punti in cui la pancia supera il dorso.
    violation = np.maximum(lower + min_thickness - upper, 0.0)

    return float(np.mean(violation ** 2))


def safe_log(value: float) -> float:
    return float(math.log(max(float(value), EPS)))


def scalar_from_neuralfoil(value) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Output NeuralFoil vuoto")
    return float(arr[0])


# ============================================================
# NEURALFOIL EVALUATOR
# ============================================================

def evaluate_with_neuralfoil(
    y_profile: np.ndarray,
    cond: Dict[str, float],
    x_fixed: np.ndarray,
) -> Dict[str, float]:
    coords = y_profile_to_selig_coordinates(y_profile, x_fixed)

    aero = nf.get_aero_from_coordinates(
        coordinates=coords,
        alpha=float(cond["alpha"]),
        Re=float(cond["re"]),
        n_crit=float(cond["ncrit"]),
        model_size=NEURALFOIL_MODEL_SIZE,
    )

    result = {
        "cl": scalar_from_neuralfoil(aero["CL"]),
        "cd": scalar_from_neuralfoil(aero["CD"]),
        "cm": scalar_from_neuralfoil(aero["CM"]),
        "top_xtr": scalar_from_neuralfoil(aero["Top_Xtr"]),
        "bot_xtr": scalar_from_neuralfoil(aero["Bot_Xtr"]),
        "analysis_confidence": scalar_from_neuralfoil(aero.get("analysis_confidence", 1.0)),
    }

    # Non tutte le versioni/entrypoint di NeuralFoil restituiscono CDp.
    if "CDp" in aero:
        result["cdp"] = scalar_from_neuralfoil(aero["CDp"])
    elif "CDP" in aero:
        result["cdp"] = scalar_from_neuralfoil(aero["CDP"])

    return result


def aerodynamic_loss(
    pred_aero: Dict[str, float],
    target: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    terms: Dict[str, float] = {}

    terms["cl"] = W_CL * (pred_aero["cl"] - target["cl"]) ** 2
    terms["log_cd"] = W_LOG_CD * (safe_log(pred_aero["cd"]) - safe_log(target["cd"])) ** 2
    terms["cm"] = W_CM * (pred_aero["cm"] - target["cm"]) ** 2
    terms["top_xtr"] = W_TOP_XTR * (pred_aero["top_xtr"] - target["top_xtr"]) ** 2
    terms["bot_xtr"] = W_BOT_XTR * (pred_aero["bot_xtr"] - target["bot_xtr"]) ** 2

    if W_LOG_CDP > 0.0 and "cdp" in pred_aero and "cdp" in target:
        terms["log_cdp"] = W_LOG_CDP * (safe_log(pred_aero["cdp"]) - safe_log(target["cdp"])) ** 2

    confidence = pred_aero.get("analysis_confidence", 1.0)
    terms["confidence"] = W_CONFIDENCE * max(MIN_ANALYSIS_CONFIDENCE - confidence, 0.0) ** 2

    return float(sum(terms.values())), terms


# ============================================================
# MODEL PARAMETER HELPERS
# ============================================================

def get_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    if not FINETUNE_ONLY_LAST_LINEAR:
        for p in model.parameters():
            p.requires_grad = True
        return [p for p in model.parameters() if p.requires_grad]

    for p in model.parameters():
        p.requires_grad = False

    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise RuntimeError("Nessun layer nn.Linear trovato nel modello")

    last_linear = linear_layers[-1]
    for p in last_linear.parameters():
        p.requires_grad = True

    return [p for p in model.parameters() if p.requires_grad]


def parameters_to_vector(params: Sequence[nn.Parameter]) -> torch.Tensor:
    return nn.utils.parameters_to_vector([p.detach() for p in params])


def vector_to_parameters(vector: torch.Tensor, params: Sequence[nn.Parameter]) -> None:
    nn.utils.vector_to_parameters(vector, params)


# ============================================================
# BLACK-BOX LOSS
# ============================================================

def get_condition_from_sample(dataset, aero_x_norm: torch.Tensor) -> Dict[str, float]:
    cond = dataset.denormalize_features(aero_x_norm)
    return {
        "re": float(cond["re"]),
        "ncrit": float(cond["ncrit"]),
        "alpha": float(cond["alpha"]),
        "cl": float(cond["cl"]),
        "cd": float(cond["cd"]),
        "cdp": float(cond["cdp"]),
        "cm": float(cond["cm"]),
        "top_xtr": float(cond["top_xtr"]),
        "bot_xtr": float(cond["bot_xtr"]),
    }


def batch_blackbox_loss(
    model: nn.Module,
    anchor_model: nn.Module,
    dataset,
    indices: Sequence[int],
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    anchor_model.eval()

    total_loss = 0.0
    detail_sums: Dict[str, float] = {}
    valid_count = 0

    for idx in indices:
        aero_x_norm, _true_y_norm, _meta = dataset[int(idx)]
        aero_x_norm_device = aero_x_norm.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_y_norm = model(aero_x_norm_device).squeeze(0).cpu()
            anchor_y_norm = anchor_model(aero_x_norm_device).squeeze(0).cpu()

        pred_y = dataset.denormalize_coords(pred_y_norm)
        cond = get_condition_from_sample(dataset, aero_x_norm)

        try:
            pred_aero = evaluate_with_neuralfoil(
                y_profile=pred_y,
                cond=cond,
                x_fixed=dataset.x_fixed,
            )
        except Exception as exc:
            # Geometria troppo problematica o errore di conversione: penalita' forte.
            penalty = 1.0e3
            total_loss += penalty
            detail_sums["failed_eval"] = detail_sums.get("failed_eval", 0.0) + 1.0
            valid_count += 1
            continue

        aero_loss_value, aero_terms = aerodynamic_loss(pred_aero, cond)

        smooth = smoothness_loss_np(pred_y)
        thickness = thickness_violation_loss_np(pred_y)
        anchor = float(torch.mean((pred_y_norm - anchor_y_norm) ** 2).item())

        geom_loss = W_SMOOTH * smooth + W_THICKNESS * thickness + W_ANCHOR * anchor
        loss = aero_loss_value + geom_loss

        total_loss += float(loss)
        valid_count += 1

        for key, value in aero_terms.items():
            detail_sums[key] = detail_sums.get(key, 0.0) + float(value)
        detail_sums["smooth"] = detail_sums.get("smooth", 0.0) + float(W_SMOOTH * smooth)
        detail_sums["thickness"] = detail_sums.get("thickness", 0.0) + float(W_THICKNESS * thickness)
        detail_sums["anchor"] = detail_sums.get("anchor", 0.0) + float(W_ANCHOR * anchor)

    denom = max(valid_count, 1)
    details = {key: value / denom for key, value in detail_sums.items()}

    return total_loss / denom, details


def sample_indices(rng: np.random.Generator, dataset_len: int, n_samples: int) -> np.ndarray:
    n = min(int(n_samples), int(dataset_len))
    return rng.choice(dataset_len, size=n, replace=False)


# ============================================================
# CHECKPOINT
# ============================================================

def load_pretrained_model(device: torch.device, target_dim: int) -> Tuple[AeroToAirfoil, Dict]:
    checkpoint = torch.load(WEIGHTS_FILE_PRETRAINED, map_location=device)

    model = AeroToAirfoil(
        input_dim=checkpoint.get("input_dim", len(FEATURE_NAMES)),
        output_dim=checkpoint.get("output_dim", target_dim),
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    return model, checkpoint


def save_finetuned_checkpoint(
    model: nn.Module,
    pretrained_checkpoint: Dict,
    train_dataset,
    validation_dataset,
    step: int,
    best_loss: float,
    config: Dict,
) -> None:
    WEIGHTS_FILE_INVERSE.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "best_blackbox_loss": best_loss,
            "feature_names": FEATURE_NAMES,
            "scalers": train_dataset.scalers,
            "x_fixed": train_dataset.x_fixed,
            "input_dim": len(FEATURE_NAMES),
            "output_dim": train_dataset.target_dim,
            "pretrained_checkpoint": str(WEIGHTS_FILE_PRETRAINED),
            "finetune_config": config,
            "pretraining_metadata": {
                "epoch": pretrained_checkpoint.get("epoch"),
                "train_loss": pretrained_checkpoint.get("train_loss"),
                "val_loss": pretrained_checkpoint.get("val_loss"),
            },
        },
        WEIGHTS_FILE_INVERSE,
    )


def append_log(step: int, train_loss: float, eval_loss: float, details: Dict[str, float]) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    header = ["step", "train_batch_loss", "eval_loss"] + sorted(details.keys())
    row = {
        "step": step,
        "train_batch_loss": train_loss,
        "eval_loss": eval_loss,
        **details,
    }

    write_header = not LOG_FILE.exists()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(row.get(k, "")) for k in header) + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader, dataloader_v, train_dataset, validation_dataset, test_dataset = get_inverse_data()

    model, pretrained_checkpoint = load_pretrained_model(
        device=device,
        target_dim=train_dataset.target_dim,
    )

    # Usa gli scaler salvati nel checkpoint pretrained, se presenti.
    # Serve a garantire che fine-tuning e predict usino la stessa normalizzazione.
    if "scalers" in pretrained_checkpoint:
        train_dataset.scalers = pretrained_checkpoint["scalers"]
        validation_dataset.scalers = pretrained_checkpoint["scalers"]
        test_dataset.scalers = pretrained_checkpoint["scalers"]

    anchor_model, _ = load_pretrained_model(
        device=device,
        target_dim=train_dataset.target_dim,
    )
    anchor_model.eval()
    for p in anchor_model.parameters():
        p.requires_grad = False

    params = get_trainable_parameters(model)
    theta = parameters_to_vector(params).to(device)

    n_params = int(theta.numel())
    eval_indices = sample_indices(rng, len(validation_dataset), EVAL_SAMPLES)

    print("\nFine-tuning inverse model con NeuralFoil evaluator")
    print(f"  checkpoint pretrained: {WEIGHTS_FILE_PRETRAINED}")
    print(f"  checkpoint finale:     {WEIGHTS_FILE_INVERSE}")
    print(f"  validation samples:    {len(validation_dataset)}")
    print(f"  eval samples fissi:    {len(eval_indices)}")
    print(f"  parametri aggiornati:  {n_params}")
    print(f"  NeuralFoil model:      {NEURALFOIL_MODEL_SIZE}\n")

    best_loss, best_details = batch_blackbox_loss(
        model=model,
        anchor_model=anchor_model,
        dataset=validation_dataset,
        indices=eval_indices,
        device=device,
    )

    config = {
        "neuralfoil_model_size": NEURALFOIL_MODEL_SIZE,
        "finetune_steps": FINETUNE_STEPS,
        "blackbox_batch_size": BLACKBOX_BATCH_SIZE,
        "eval_every": EVAL_EVERY,
        "eval_samples": EVAL_SAMPLES,
        "random_seed": RANDOM_SEED,
        "finetune_only_last_linear": FINETUNE_ONLY_LAST_LINEAR,
        "spsa_sigma": SPSA_SIGMA,
        "spsa_lr": SPSA_LR,
        "spsa_grad_clip_norm": SPSA_GRAD_CLIP_NORM,
        "weights": {
            "cl": W_CL,
            "log_cd": W_LOG_CD,
            "log_cdp": W_LOG_CDP,
            "cm": W_CM,
            "top_xtr": W_TOP_XTR,
            "bot_xtr": W_BOT_XTR,
            "smooth": W_SMOOTH,
            "thickness": W_THICKNESS,
            "anchor": W_ANCHOR,
            "confidence": W_CONFIDENCE,
        },
    }

    save_finetuned_checkpoint(
        model=model,
        pretrained_checkpoint=pretrained_checkpoint,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        step=0,
        best_loss=best_loss,
        config=config,
    )

    print(f"Step 0000 | eval_loss={best_loss:.6f} | salvato checkpoint iniziale")

    for step in range(1, FINETUNE_STEPS + 1):
        train_indices = sample_indices(rng, len(validation_dataset), BLACKBOX_BATCH_SIZE)

        delta_np = rng.choice([-1.0, 1.0], size=n_params).astype(np.float32)
        delta = torch.tensor(delta_np, device=device)

        theta_plus = theta + SPSA_SIGMA * delta
        vector_to_parameters(theta_plus, params)
        loss_plus, _ = batch_blackbox_loss(
            model=model,
            anchor_model=anchor_model,
            dataset=validation_dataset,
            indices=train_indices,
            device=device,
        )

        theta_minus = theta - SPSA_SIGMA * delta
        vector_to_parameters(theta_minus, params)
        loss_minus, _ = batch_blackbox_loss(
            model=model,
            anchor_model=anchor_model,
            dataset=validation_dataset,
            indices=train_indices,
            device=device,
        )

        grad_estimate = ((loss_plus - loss_minus) / (2.0 * SPSA_SIGMA)) * delta
        grad_norm = torch.linalg.vector_norm(grad_estimate)

        if float(grad_norm.item()) > SPSA_GRAD_CLIP_NORM:
            grad_estimate = grad_estimate * (SPSA_GRAD_CLIP_NORM / (grad_norm + EPS))

        theta = theta - SPSA_LR * grad_estimate
        vector_to_parameters(theta, params)

        train_batch_loss, train_details = batch_blackbox_loss(
            model=model,
            anchor_model=anchor_model,
            dataset=validation_dataset,
            indices=train_indices,
            device=device,
        )

        if step % EVAL_EVERY == 0 or step == FINETUNE_STEPS:
            eval_loss, eval_details = batch_blackbox_loss(
                model=model,
                anchor_model=anchor_model,
                dataset=validation_dataset,
                indices=eval_indices,
                device=device,
            )

            append_log(step, train_batch_loss, eval_loss, eval_details)

            improved = eval_loss < best_loss
            if improved:
                best_loss = eval_loss
                best_details = eval_details
                save_finetuned_checkpoint(
                    model=model,
                    pretrained_checkpoint=pretrained_checkpoint,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    step=step,
                    best_loss=best_loss,
                    config=config,
                )

            status = "SAVED" if improved else "-----"
            print(
                f"Step {step:04d}/{FINETUNE_STEPS} | "
                f"train_loss={train_batch_loss:.6f} | "
                f"eval_loss={eval_loss:.6f} | "
                f"best={best_loss:.6f} | {status}"
            )

    print("\nFine-tuning completato.")
    print(f"Miglior checkpoint fine-tuned salvato in: {WEIGHTS_FILE_INVERSE}")
    print(f"Log CSV: {LOG_FILE}")


if __name__ == "__main__":
    main()
