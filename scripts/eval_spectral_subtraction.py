"""
scripts/eval_spectral_subtraction.py

Evaluates the classical spectral subtraction baseline on the test set.
Prints PESQ and STOI scores for the noisy input and spectral subtraction output.

Usage:
    python scripts/eval_spectral_subtraction.py --config config.yaml
    python scripts/eval_spectral_subtraction.py --config config.yaml --noise-frames 10
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

import numpy as np
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.baselines.spectral_subtraction import spectral_subtraction
from src.dataset import SpeechDataset
from src.metrics.pesq import compute_pesq
from src.metrics.stoi import compute_stoi
from src.stft import compute_istft


def lps_to_mag(lps):
    """Convert log power spectrogram back to linear magnitude: LPS = log(mag^2) -> mag = exp(LPS/2)."""
    return np.exp(lps / 2)


def evaluate(config, noise_frames):
    test_set = SpeechDataset(config["manifest"], split="test")
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
    )

    pesq_noisy_scores = []
    pesq_sub_scores = []
    stoi_noisy_scores = []
    stoi_sub_scores = []

    for noisy_lps, noisy_phase, clean_lps, lps_mean, lps_std in tqdm(test_loader, desc="Evaluating spectral subtraction"):
        mean = lps_mean.item()
        std = lps_std.item()

        noisy_lps_np = noisy_lps.squeeze(0).numpy() * std + mean
        clean_lps_np = clean_lps.squeeze(0).numpy() * std + mean
        noisy_phase_np = noisy_phase.squeeze(0).numpy()

        noisy_mag = lps_to_mag(noisy_lps_np)
        clean_mag = lps_to_mag(clean_lps_np)

        noisy_wav = compute_istft(noisy_mag, noisy_phase_np)
        clean_wav = compute_istft(clean_mag, noisy_phase_np)
        sub_wav = spectral_subtraction(noisy_wav, n_noise_frames=noise_frames)

        min_len = min(len(noisy_wav), len(clean_wav), len(sub_wav))
        noisy_wav = noisy_wav[:min_len]
        clean_wav = clean_wav[:min_len]
        sub_wav = sub_wav[:min_len]

        try:
            pesq_noisy_scores.append(compute_pesq(clean_wav, noisy_wav))
            pesq_sub_scores.append(compute_pesq(clean_wav, sub_wav))
        except Exception as e:
            print(f"  PESQ skipped: {e}")
            pesq_noisy_scores.append(np.nan)
            pesq_sub_scores.append(np.nan)

        stoi_noisy_scores.append(compute_stoi(clean_wav, noisy_wav))
        stoi_sub_scores.append(compute_stoi(clean_wav, sub_wav))

    print("\n===== SPECTRAL SUBTRACTION RESULTS =====")
    print(f"{'Metric':<10} {'Noisy':>10} {'SpecSub':>10}")
    print(f"{'PESQ':<10} {np.nanmean(pesq_noisy_scores):>10.3f} {np.nanmean(pesq_sub_scores):>10.3f}")
    print(f"{'STOI':<10} {np.mean(stoi_noisy_scores):>10.3f} {np.mean(stoi_sub_scores):>10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--noise-frames", type=int, default=10)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluate(config, args.noise_frames)
