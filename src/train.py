"""
train.py

Trains the CNNDenoiser model on VoiceBank-DEMAND dataset.

What it does:
    1. Loads config from config.yaml
    2. Builds dataset and DataLoader
    3. Builds model and moves to GPU if available
    4. Trains for N epochs, saving a checkpoint after each epoch
    5. Prints loss after each epoch

Usage:
    python src/train.py --config config.yaml

Outputs:
    checkpoints/epoch_01.pt
    checkpoints/epoch_02.pt
    ...
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from src.dataset import SpeechDataset
from src.models.cnn_denoiser import CNNDenoiser, count_parameters

def train(config):
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    train_set = SpeechDataset(config["manifest"], split="train")
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )
    model = CNNDenoiser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    count_parameters(model)
    for epoch in range(1, config["epochs"] + 1):
        total_loss = 0.0                    
        model.train()
        for noisy_lps, noisy_phase, clean_mag in tqdm(train_loader, desc=f"Epoch {epoch}"):
            noisy_lps = noisy_lps.unsqueeze(1).to(device)
            clean_mag = clean_mag.unsqueeze(1).to(device)
            optimizer.zero_grad()
            prediction = model(noisy_lps)
            loss = criterion(prediction, clean_mag)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{config['epochs']}  loss: {avg_loss:.4f}")
        checkpoint_path = os.path.join(config["checkpoint_dir"], f"epoch_{epoch:02d}.pt")
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss":            avg_loss
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)