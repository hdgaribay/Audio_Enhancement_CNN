"""
scripts/train_iq.py

Entry point for training the Stage 1 IQ denoiser.

What it does:
    1. Loads config from config.yaml
    2. Builds train and val IQ datasets
    3. Builds IQDenoiser model and moves to GPU if available
    4. Trains for N epochs with:
       - L1Loss on IQ signals
       - Gradient clipping (max_norm=1.0)
       - ReduceLROnPlateau scheduler stepping on val loss
    5. Prints train/val loss after each epoch
    6. Saves checkpoint after each epoch

Usage:
    python scripts/train_iq.py --config config.yaml

Outputs:
    checkpoints/iq_epoch_01.pt
    checkpoints/iq_epoch_02.pt
    ...
    checkpoints/iq_best.pt
"""

import argparse
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from src.datasets.iq_dataset import IQDataset
from src.models.iq_denoiser import IQDenoiser, count_parameters

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def train_iq(config):
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # build datasets
    train_set = IQDataset(config["manifest"], config, split="train")
    val_set   = IQDataset(config["manifest"], config, split="val")

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

    # build model
    model     = IQDenoiser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.get("lr_patience", 3),
    )
    # L1Loss works well for signal reconstruction —
    # it's less sensitive to outliers than MSE
    criterion = nn.L1Loss()

    count_parameters(model)

    best_val_loss  = float("inf")
    best_ckpt_path = os.path.join(config["checkpoint_dir"], "iq_best.pt")

    for epoch in range(1, config["epochs"] + 1):

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for noisy_iq, clean_iq, _ in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            # noisy_iq and clean_iq are (B, 2, IQ_FRAME_LENGTH)
            noisy_iq = noisy_iq.to(device)
            clean_iq = clean_iq.to(device)

            optimizer.zero_grad()
            prediction = model(noisy_iq)
            loss       = criterion(prediction, clean_iq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_iq, clean_iq, _ in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                noisy_iq = noisy_iq.to(device)
                clean_iq = clean_iq.to(device)
                prediction = model(noisy_iq)
                val_loss  += criterion(prediction, clean_iq).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{config['epochs']}  "
            f"train_loss: {train_loss:.4f}  "
            f"val_loss: {val_loss:.4f}  "
            f"lr: {current_lr:.2e}"
        )

        # ── save checkpoint ───────────────────────────────────────────────────
        checkpoint_path = os.path.join(
            config["checkpoint_dir"], f"iq_epoch_{epoch:02d}.pt"
        )
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

        # keep only last N checkpoints
        keep_last = config.get("keep_last", 5)
        all_ckpts = sorted(glob.glob(
            os.path.join(config["checkpoint_dir"], "iq_epoch_*.pt")
        ))
        for old in all_ckpts[:-keep_last]:
            os.remove(old)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_iq(config)