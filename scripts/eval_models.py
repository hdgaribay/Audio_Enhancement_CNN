"""
scripts/eval_models.py

Evaluates trained CNNDenoiser on the test set.
Prints PESQ and STOI scores for noisy input and CNN output.

Usage:
    python scripts/eval_models.py --config config.yaml --checkpoint checkpoints/epoch_30.pt
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import yaml
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import SpeechDataset
from src.models.cnn_denoiser import CNNDenoiser
from src.stft import compute_istft
from src.metrics.pesq import compute_pesq
from src.metrics.stoi import compute_stoi

def evaluate(config, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set = SpeechDataset(config["manifest"], split="test")
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"]
    )
    model = CNNDenoiser().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    pesq_noisy_scores = []
    pesq_cnn_scores   = []
    stoi_noisy_scores = []
    stoi_cnn_scores   = []
    with torch.no_grad():
        for noisy_lps, noisy_phase, clean_mag in tqdm(test_loader, desc="Evaluating"):
            # move to device
            noisy_input = noisy_lps.unsqueeze(1).to(device)

            # forward pass
            prediction = model(noisy_input)

            # back to numpy — remove batch and channel dims
            prediction  = prediction.squeeze(0).squeeze(0).cpu().numpy()
            noisy_mag   = noisy_lps.squeeze(0).numpy()
            noisy_phase = noisy_phase.squeeze(0).numpy()
            clean_mag_np = clean_mag.squeeze(0).numpy()

            # reconstruct waveforms via ISTFT
            enhanced_wav = compute_istft(prediction, noisy_phase)
            clean_wav    = compute_istft(clean_mag_np, noisy_phase)
            noisy_wav    = compute_istft(noisy_mag, noisy_phase)

            # compute scores
            pesq_noisy_scores.append(compute_pesq(clean_wav, noisy_wav))
            pesq_cnn_scores.append(compute_pesq(clean_wav, enhanced_wav))
            stoi_noisy_scores.append(compute_stoi(clean_wav, noisy_wav))
            stoi_cnn_scores.append(compute_stoi(clean_wav, enhanced_wav))
    print("\n===== RESULTS =====")
    print(f"{'Metric':<10} {'Noisy':>10} {'CNN':>10}")
    print(f"{'PESQ':<10} {np.mean(pesq_noisy_scores):>10.3f} {np.mean(pesq_cnn_scores):>10.3f}")
    print(f"{'STOI':<10} {np.mean(stoi_noisy_scores):>10.3f} {np.mean(stoi_cnn_scores):>10.3f}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluate(config, args.checkpoint)