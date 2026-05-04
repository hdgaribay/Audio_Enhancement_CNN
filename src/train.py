"""
train.py

Trains the CNNDenoiser model on VoiceBank-DEMAND dataset.

What it does:
    1. Loads config from config.yaml
    2. Builds train and val datasets/loaders (90/10 split of manifest train rows)
    3. Builds model and moves to GPU if available
    4. Trains for N epochs with:
       - MSELoss on log power spectrograms (LPS→LPS, same domain)
       - Gradient clipping (max_norm=1.0)
       - ReduceLROnPlateau scheduler stepping on val loss
    5. Prints train/val loss and current LR after each epoch
    6. Saves checkpoint after each epoch

Usage:
    python scripts/train_cnn.py --config config.yaml

Outputs:
    checkpoints/epoch_01.pt
    checkpoints/epoch_02.pt
    ...
"""

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from src.dataset import SpeechDataset
from src.models.cnn_denoiser import CNNDenoiser, count_parameters


def train(config):
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    train_set = SpeechDataset(config["manifest"], split="train")
    val_set   = SpeechDataset(config["manifest"], split="val")

    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )

    model     = CNNDenoiser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.get("lr_patience", 3),
    )
    criterion = nn.MSELoss()

    count_parameters(model)

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(config["checkpoint_dir"], "best.pt")

    for epoch in range(1, config["epochs"] + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for noisy_lps, _, clean_lps, _lps_mean, _lps_std in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            noisy_lps = noisy_lps.unsqueeze(1).to(device)
            clean_lps = clean_lps.unsqueeze(1).to(device)

            optimizer.zero_grad()
            prediction = model(noisy_lps)
            loss = criterion(prediction, clean_lps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_lps, _, clean_lps, _lps_mean, _lps_std in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                noisy_lps = noisy_lps.unsqueeze(1).to(device)
                clean_lps = clean_lps.unsqueeze(1).to(device)
                prediction = model(noisy_lps)
                val_loss += criterion(prediction, clean_lps).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{config['epochs']}  "
            f"train_loss: {train_loss:.4f}  "
            f"val_loss: {val_loss:.4f}  "
            f"lr: {current_lr:.2e}"
        )

        checkpoint_path = os.path.join(config["checkpoint_dir"], f"epoch_{epoch:02d}.pt")
        ckpt = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss":      train_loss,
            "val_loss":        val_loss,
        }
        torch.save(ckpt, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, best_ckpt_path)
            print(f"New best val_loss: {best_val_loss:.4f} → saved {best_ckpt_path}")

        # Rotate per-epoch checkpoints; best.pt is named separately so it is never deleted.
        keep_last = config.get("keep_last", 5)
        all_ckpts = sorted(glob.glob(os.path.join(config["checkpoint_dir"], "epoch_*.pt")))
        for old in all_ckpts[:-keep_last]:
            os.remove(old)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)