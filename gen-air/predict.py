import torch
import matplotlib.pyplot as plt
import numpy as np

from src.Autoencoder_m1 import AeroToAirfoil
from src.main_data import get_inverse_data, FEATURE_NAMES
from src.config import RESULTS_DIR, WEIGHTS_FILE_INVERSE
from src.run_xfoil import run_generated_profile_with_xfoil


PREDICT_RANDOM_SEED = 42
RUN_XFOIL = True

# Imposta qui l'alpha desiderato.
TARGET_ALPHA = 4.0
ALPHA_TOL = 1.0e-6


def choose_random_sample_with_alpha(test_dataset, target_alpha, seed=PREDICT_RANDOM_SEED, alpha_tol=ALPHA_TOL):
    candidate_indices = []

    for idx, sample in enumerate(test_dataset.samples):
        alpha = float(sample["meta"]["alpha"])
        if np.isclose(alpha, target_alpha, atol=alpha_tol):
            candidate_indices.append(idx)

    if not candidate_indices:
        available_alphas = sorted(
            {round(float(sample["meta"]["alpha"]), 6) for sample in test_dataset.samples}
        )
        raise ValueError(
            f"Nessuna riga nel test_dataset con alpha={target_alpha}. "
            f"Alpha disponibili per il profilo test: {available_alphas}"
        )

    rng = np.random.default_rng(seed)
    return int(rng.choice(candidate_indices))


def main():
    _, _, _, _, test_dataset = get_inverse_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(WEIGHTS_FILE_INVERSE, map_location=device)

    model = AeroToAirfoil(
        input_dim=checkpoint.get("input_dim", len(FEATURE_NAMES)),
        output_dim=checkpoint.get("output_dim", test_dataset.target_dim),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample_index = choose_random_sample_with_alpha(
        test_dataset=test_dataset,
        target_alpha=TARGET_ALPHA,
        seed=PREDICT_RANDOM_SEED,
    )

    aero_x_norm, true_y_norm, meta = test_dataset[sample_index]

    with torch.no_grad():
        pred_y_norm = model(aero_x_norm.unsqueeze(0).to(device)).squeeze(0).cpu()

    pred_y = test_dataset.denormalize_coords(pred_y_norm)
    true_y = test_dataset.denormalize_coords(true_y_norm)
    cond = test_dataset.denormalize_features(aero_x_norm)

    print(f"\nSample scelto randomicamente tra le righe del test set con alpha={TARGET_ALPHA}")
    print(f"  sample_index: {sample_index}")

    print("\nCondizioni aerodinamiche date in input alla rete:")
    for key in ["re", "ncrit", "alpha", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr"]:
        print(f"  {key}: {cond[key]}")

    print("\nMeta originale:")
    print(f"  airfoil_name: {meta['airfoil_name']}")
    print(f"  polar_file: {meta['polar_file']}")

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

        print("\nValori polar usati come input alla rete:")
        for key in ["alpha", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr"]:
            print(f"  {key}: {cond[key]}")


if __name__ == "__main__":
    main()