"""
scripts/train_iq.py

Entry point for training the Stage 1 IQ denoiser.

What it does:
    1. Loads config from config.yaml
    2. Checks for existing checkpoints and resumes if found
    3. Builds train and val IQ datasets
    4. Builds IQDenoiser model and moves to GPU if available
    5. Trains for N epochs with:
       - L1Loss on IQ signals
       - Gradient clipping (max_norm=1.0)
       - ReduceLROnPlateau scheduler stepping on val loss
    6. Prints train/val loss after each epoch
    7. Saves checkpoint after each epoch

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
import sys
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.iq_dataset import IQDataset
from src.models.iq_denoiser import IQDenoiser, count_parameters


def find_latest_checkpoint(checkpoint_dir):
    """
    Looks for the most recent iq_epoch_XX.pt file in checkpoint_dir.
    Returns the path if found, None if no checkpoints exist yet.

    This is how resume works — we find the last saved epoch and
    load it instead of starting from scratch.
    """
    pattern = os.path.join(checkpoint_dir, "iq_epoch_*.pt")
    checkpoints = sorted(glob.glob(pattern))

    if not checkpoints:
        return None

    # sorted() puts them in order, so last one is most recent
    latest = checkpoints[-1]
    print(f"Found existing checkpoint: {latest}")
    return latest


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
    criterion = nn.L1Loss()

    count_parameters(model)

    # ── resume from checkpoint if one exists ──────────────────────────────────
    start_epoch    = 1
    best_val_loss  = float("inf")
    best_ckpt_path = os.path.join(config["checkpoint_dir"], "iq_best.pt")

    latest_ckpt = find_latest_checkpoint(config["checkpoint_dir"])

    if latest_ckpt:
        print(f"Resuming training from: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", ckpt["val_loss"])
        print(f"Resuming from epoch {start_epoch}, best val_loss so far: {best_val_loss:.4f}")
    else:
        print("No checkpoint found — starting from scratch")

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config["epochs"] + 1):

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for noisy_iq, clean_iq, _ in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
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
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss":     train_loss,
            "val_loss":       val_loss,
            "best_val_loss":  best_val_loss,
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

        # ── sync checkpoints to Drive after every epoch ───────────────────────
        # this ensures we don't lose progress if Colab resets
        drive_ckpt_dir = "/content/drive/MyDrive/Audio_Enhancement_CNN/checkpoints"
        if os.path.exists("/content/drive"):
            import shutil
            os.makedirs(drive_ckpt_dir, exist_ok=True)
            shutil.copytree(
                config["checkpoint_dir"],
                drive_ckpt_dir,
                dirs_exist_ok=True
            )
            print(f"Checkpoints synced to Drive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_iq(config)