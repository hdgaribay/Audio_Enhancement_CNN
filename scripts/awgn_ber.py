"""
scripts/awgn_ber.py

QPSK AWGN channel simulation.
Compares BER of noisy speech vs CNN-enhanced speech vs theoretical QPSK.

Usage:
    python scripts/awgn_ber.py --config config.yaml --checkpoint checkpoints/epoch_30.pt

Outputs:
    outputs/ber_curve.png
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import torch
from scipy.special import erfc
from tqdm import tqdm
from src.audio_io import load_audio
from src.stft import compute_stft, compute_lps, compute_istft
from src.models.cnn_denoiser import CNNDenoiser
import pandas as pd


def wav_to_bits(wav):
    wav = wav / (np.max(np.abs(wav)) + 1e-8)
    samples_int = ((wav + 1) * 127.5).astype(np.uint8)
    bits = np.unpackbits(samples_int)
    return bits


def qpsk_modulate(bits):
    bits = bits[:len(bits) // 2 * 2]
    bits_paired = bits.reshape(-1, 2)
    mapping = {(0, 0): 1+1j, (0, 1): -1+1j, (1, 1): -1-1j, (1, 0): 1-1j}
    symbols = np.array([mapping[tuple(b)] for b in bits_paired])
    return symbols


def add_awgn(symbols, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    noise_std  = np.sqrt(1 / (2 * snr_linear))
    noise      = noise_std * (np.random.randn(*symbols.shape) +
                              1j * np.random.randn(*symbols.shape))
    return symbols + noise


def qpsk_demodulate(received, original_bits):
    rx_bits       = np.zeros(len(received) * 2, dtype=int)
    rx_bits[0::2] = (received.real < 0).astype(int)
    rx_bits[1::2] = (received.imag < 0).astype(int)
    n             = min(len(rx_bits), len(original_bits))
    return np.mean(rx_bits[:n] != original_bits[:n])


def enhance_wav(wav, model, device):
    mag, phase = compute_stft(wav)
    lps        = compute_lps(mag)

    lps_tensor = torch.from_numpy(lps.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_lps = model(lps_tensor)

    # model outputs LPS — convert back to linear magnitude for ISTFT
    enhanced_lps = pred_lps.squeeze(0).squeeze(0).cpu().numpy()
    enhanced_mag = np.exp(enhanced_lps / 2)

    return compute_istft(enhanced_mag, phase)


def run_simulation(config, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNDenoiser().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    snr_range = list(range(config["awgn_snr_min"], config["awgn_snr_max"] + 1))    ber_noisy    = np.zeros(len(snr_range))
    ber_enhanced = np.zeros(len(snr_range))
    ber_theory   = np.zeros(len(snr_range))
    n_files      = 0

    df      = pd.read_csv(config["manifest"])
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Simulating"):
        noisy_wav, _ = load_audio(row["noisy_path"])

        enhanced_wav = enhance_wav(noisy_wav, model, device)

        min_len      = min(len(noisy_wav), len(enhanced_wav))
        noisy_wav    = noisy_wav[:min_len]
        enhanced_wav = enhanced_wav[:min_len]

        bits_noisy    = wav_to_bits(noisy_wav)
        bits_enhanced = wav_to_bits(enhanced_wav)

        for i, snr_db in enumerate(snr_range):
            snr_linear = 10 ** (snr_db / 10)

            symbols_noisy    = qpsk_modulate(bits_noisy)
            received_noisy   = add_awgn(symbols_noisy, snr_db)
            ber_noisy[i]    += qpsk_demodulate(received_noisy, bits_noisy)

            symbols_enhanced  = qpsk_modulate(bits_enhanced)
            received_enhanced = add_awgn(symbols_enhanced, snr_db)
            ber_enhanced[i]  += qpsk_demodulate(received_enhanced, bits_enhanced)

            ber_theory[i] += 0.5 * erfc(np.sqrt(snr_linear))

        n_files += 1

    ber_noisy    /= n_files
    ber_enhanced /= n_files
    ber_theory   /= n_files

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.semilogy(snr_range, ber_noisy,    'r--o', label="Noisy speech")
    plt.semilogy(snr_range, ber_enhanced, 'b--o', label="CNN enhanced")
    plt.semilogy(snr_range, ber_theory,   'k--',  label="Theoretical QPSK")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("QPSK BER vs SNR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/ber_curve.png")
    plt.close()
    print("BER curve saved to outputs/ber_curve.png")

    print(f"\n{'SNR (dB)':<10} {'Noisy BER':>12} {'CNN BER':>12} {'Theory BER':>12}")
    for i, snr in enumerate(snr_range):
        print(f"{snr:<10} {ber_noisy[i]:>12.4f} {ber_enhanced[i]:>12.4f} {ber_theory[i]:>12.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_simulation(config, args.checkpoint)
