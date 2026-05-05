"""
scripts/eval_models.py

Evaluates the full two-stage pipeline on the test set.
Prints PESQ and STOI scores for three conditions:

    1. Noisy input     — no processing, baseline
    2. After Stage 1   — after IQ denoising only
    3. After Stage 2   — after full pipeline (IQ + audio denoising)

Usage:
    python scripts/eval_models.py \
        --config config.yaml \
        --iq_checkpoint    checkpoints/iq_best.pt \
        --audio_checkpoint checkpoints/best.pt

The score table lets you see exactly how much each stage contributes:
    Noisy → Stage 1 improvement = IQ denoiser contribution
    Stage 1 → Stage 2 improvement = audio denoiser contribution
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.speech_dataset import SpeechDataset
from src.models.iq_denoiser import IQDenoiser
from src.models.cnn_denoiser import CNNDenoiser
from src.stft import compute_istft, compute_stft, compute_lps
from src.iq.modulate import modulate
from src.iq.noise import apply_noise
from src.iq.demodulate import demodulate
from src.metrics.pesq_and_stoi import compute_pesq, compute_stoi


def lps_to_mag(lps):
    """Convert log power spectrogram back to linear magnitude."""
    return np.exp(lps / 2)


def run_iq_stage(waveform, iq_model, config, device):
    """
    Run a waveform through the IQ stage.
    Same chunked processing as run_pipeline.py.
    """
    i_sig, q_sig, n_bits = modulate(waveform)
    i_noisy, q_noisy     = apply_noise(i_sig, q_sig, config)

    chunk_size    = 16000
    iq_tensor     = torch.from_numpy(
        np.stack([i_noisy, q_noisy], axis=0).astype(np.float32)
    ).unsqueeze(0).to(device)

    iq_length     = iq_tensor.shape[2]
    output_chunks = []

    with torch.no_grad():
        for start in range(0, iq_length, chunk_size):
            end   = min(start + chunk_size, iq_length)
            chunk = iq_tensor[:, :, start:end]

            if chunk.shape[2] < chunk_size:
                pad   = chunk_size - chunk.shape[2]
                chunk = torch.nn.functional.pad(chunk, (0, pad))

            output_chunk = iq_model(chunk)

            if end == iq_length and iq_length % chunk_size != 0:
                output_chunk = output_chunk[:, :, :chunk.shape[2] - pad]

            output_chunks.append(output_chunk)

    clean_iq    = torch.cat(output_chunks, dim=2)
    clean_iq_np = clean_iq.squeeze(0).cpu().numpy()

    reconstructed = demodulate(clean_iq_np[0], clean_iq_np[1], n_bits)
    return reconstructed.astype(np.float32)


def run_audio_stage(waveform, audio_model, device):
    """
    Run a waveform through the audio CNN stage.
    """
    mag, phase = compute_stft(waveform)
    lps        = compute_lps(mag)

    lps_mean = float(np.mean(lps))
    lps_std  = max(float(np.std(lps)), 1e-6)
    lps_norm = (lps - lps_mean) / lps_std

    lps_tensor = torch.from_numpy(
        lps_norm.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_lps_norm = audio_model(lps_tensor)

    pred_lps     = pred_lps_norm.squeeze(0).squeeze(0).cpu().numpy() * lps_std + lps_mean
    enhanced_mag = lps_to_mag(pred_lps)

    return compute_istft(enhanced_mag, phase).astype(np.float32)


def evaluate(config, iq_checkpoint, audio_checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load test dataset
    test_set    = SpeechDataset(config["manifest"], split="test")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # load models
    iq_model = IQDenoiser().to(device)
    iq_model.load_state_dict(
        torch.load(iq_checkpoint, map_location=device)["model_state"]
    )
    iq_model.eval()

    audio_model = CNNDenoiser().to(device)
    audio_model.load_state_dict(
        torch.load(audio_checkpoint, map_location=device)["model_state"]
    )
    audio_model.eval()

    print(f"IQ checkpoint    : {iq_checkpoint}")
    print(f"Audio checkpoint : {audio_checkpoint}\n")

    # score accumulators for all three conditions
    pesq_noisy,  stoi_noisy  = [], []
    pesq_stage1, stoi_stage1 = [], []
    pesq_stage2, stoi_stage2 = [], []

    with torch.no_grad():
        for noisy_lps, noisy_phase, clean_lps, lps_mean, lps_std in tqdm(
            test_loader, desc="Evaluating"
        ):
            mean = lps_mean.item()
            std  = lps_std.item()

            # reconstruct waveforms from LPS
            noisy_lps_np   = noisy_lps.squeeze(0).numpy() * std + mean
            clean_lps_np   = clean_lps.squeeze(0).numpy() * std + mean
            noisy_phase_np = noisy_phase.squeeze(0).numpy()

            noisy_wav = compute_istft(lps_to_mag(noisy_lps_np), noisy_phase_np)
            clean_wav = compute_istft(lps_to_mag(clean_lps_np), noisy_phase_np)

            # run through IQ stage
            after_stage1 = run_iq_stage(noisy_wav, iq_model, config, device)

            # run through audio stage
            after_stage2 = run_audio_stage(after_stage1, audio_model, device)

            # align lengths
            min_len      = min(len(clean_wav), len(noisy_wav),
                               len(after_stage1), len(after_stage2))
            clean_wav    = clean_wav[:min_len]
            noisy_wav    = noisy_wav[:min_len]
            after_stage1 = after_stage1[:min_len]
            after_stage2 = after_stage2[:min_len]

            # compute metrics for all three conditions
            try:
                pesq_noisy.append(compute_pesq(clean_wav, noisy_wav))
                pesq_stage1.append(compute_pesq(clean_wav, after_stage1))
                pesq_stage2.append(compute_pesq(clean_wav, after_stage2))
            except Exception as e:
                print(f"  PESQ skipped: {e}")
                pesq_noisy.append(np.nan)
                pesq_stage1.append(np.nan)
                pesq_stage2.append(np.nan)

            stoi_noisy.append(compute_stoi(clean_wav, noisy_wav))
            stoi_stage1.append(compute_stoi(clean_wav, after_stage1))
            stoi_stage2.append(compute_stoi(clean_wav, after_stage2))

    # print results table
    print("\n===== RESULTS =====")
    print(f"{'Metric':<10} {'Noisy':>10} {'Stage 1':>10} {'Stage 2':>10}")
    print("-" * 42)
    print(
        f"{'PESQ':<10} "
        f"{np.nanmean(pesq_noisy):>10.3f} "
        f"{np.nanmean(pesq_stage1):>10.3f} "
        f"{np.nanmean(pesq_stage2):>10.3f}"
    )
    print(
        f"{'STOI':<10} "
        f"{np.mean(stoi_noisy):>10.3f} "
        f"{np.mean(stoi_stage1):>10.3f} "
        f"{np.mean(stoi_stage2):>10.3f}"
    )
    print("\nHigher is better for both metrics.")
    print("PESQ range: -0.5 to 4.5")
    print("STOI range: 0 to 1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",           type=str, default="config.yaml")
    parser.add_argument("--iq_checkpoint",    type=str, required=True)
    parser.add_argument("--audio_checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluate(config, args.iq_checkpoint, args.audio_checkpoint)