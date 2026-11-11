import torch
import torch.optim as optim
import torch.nn as nn

from src.Autoencoder_m1 import AeroToAirfoil
from src.main_data import get_inverse_data, FEATURE_NAMES, X_FIXED
from src.config import WEIGHTS_FILE_PRETRAINED

dataloader, dataloader_v, train_dataset, validation_dataset, _ = get_inverse_data()


# =========================
# TRAIN CONFIG
# =========================

EPOCHS = 500
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-6
PRINT_EVERY = 20


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n_batches = 0
    with torch.no_grad():
        for aero_x, coords_y, _meta in loader:
            aero_x = aero_x.to(device)
            coords_y = coords_y.to(device)
            pred_y = model(aero_x)
            loss = criterion(pred_y, coords_y)
            total += float(loss.item())
            n_batches += 1
    return total / max(n_batches, 1)


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss):
    WEIGHTS_FILE_PRETRAINED.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "feature_names": FEATURE_NAMES,
            "scalers": train_dataset.scalers,
            "x_fixed": X_FIXED,
            "input_dim": len(FEATURE_NAMES),
            "output_dim": train_dataset.target_dim,
        },
        WEIGHTS_FILE_PRETRAINED,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AeroToAirfoil(input_dim=len(FEATURE_NAMES), output_dim=train_dataset.target_dim).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_total = 0.0
        n_batches = 0

        for aero_x, coords_y, _meta in dataloader:
            aero_x = aero_x.to(device)
            coords_y = coords_y.to(device)

            pred_y = model(aero_x)
            loss = criterion(pred_y, coords_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += float(loss.item())
            n_batches += 1

        train_loss = train_total / max(n_batches, 1)
        val_loss = evaluate(model, dataloader_v, criterion, device)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss)

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(
                f"Epoch {epoch:04d}/{EPOCHS} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | best_val={best_val:.6f}"
            )

    print(f"\nMiglior checkpoint salvato in: {WEIGHTS_FILE_PRETRAINED}")


if __name__ == "__main__":
    main()
