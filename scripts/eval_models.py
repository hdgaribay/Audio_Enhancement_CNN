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


def lps_to_mag(lps):
    """Convert log power spectrogram back to linear magnitude: LPS = log(mag²) → mag = exp(LPS/2)."""
    return np.exp(lps / 2)


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
        for noisy_lps, noisy_phase, clean_lps in tqdm(test_loader, desc="Evaluating"):
            noisy_input = noisy_lps.unsqueeze(1).to(device)
            pred_lps    = model(noisy_input)

            # back to numpy — remove batch and channel dims
            pred_lps_np   = pred_lps.squeeze(0).squeeze(0).cpu().numpy()
            noisy_lps_np  = noisy_lps.squeeze(0).numpy()
            noisy_phase_np = noisy_phase.squeeze(0).numpy()
            clean_lps_np  = clean_lps.squeeze(0).numpy()

            # convert LPS → linear magnitude for ISTFT
            enhanced_mag = lps_to_mag(pred_lps_np)
            noisy_mag    = lps_to_mag(noisy_lps_np)
            clean_mag    = lps_to_mag(clean_lps_np)

            enhanced_wav = compute_istft(enhanced_mag, noisy_phase_np)
            clean_wav    = compute_istft(clean_mag,    noisy_phase_np)
            noisy_wav    = compute_istft(noisy_mag,    noisy_phase_np)

            # align lengths before metric computation
            min_len      = min(len(enhanced_wav), len(clean_wav), len(noisy_wav))
            enhanced_wav = enhanced_wav[:min_len]
            clean_wav    = clean_wav[:min_len]
            noisy_wav    = noisy_wav[:min_len]

            try:
                pesq_noisy_scores.append(compute_pesq(clean_wav, noisy_wav))
                pesq_cnn_scores.append(compute_pesq(clean_wav, enhanced_wav))
            except Exception as e:
                print(f"  PESQ skipped: {e}")
                pesq_noisy_scores.append(np.nan)
                pesq_cnn_scores.append(np.nan)

            stoi_noisy_scores.append(compute_stoi(clean_wav, noisy_wav))
            stoi_cnn_scores.append(compute_stoi(clean_wav, enhanced_wav))

    print("\n===== RESULTS =====")
    print(f"{'Metric':<10} {'Noisy':>10} {'CNN':>10}")
    print(f"{'PESQ':<10} {np.nanmean(pesq_noisy_scores):>10.3f} {np.nanmean(pesq_cnn_scores):>10.3f}")
    print(f"{'STOI':<10} {np.mean(stoi_noisy_scores):>10.3f} {np.mean(stoi_cnn_scores):>10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluate(config, args.checkpoint)
