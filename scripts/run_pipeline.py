"""
scripts/run_pipeline.py

Runs a noisy audio file through the full two-stage pipeline:

    Stage 1 — IQ denoiser:
        noisy audio → IQ signal → add noise → IQ denoiser → clean IQ → audio

    Stage 2 — Audio denoiser:
        reconstructed audio → CNNDenoiser → final clean audio

Usage:
    python scripts/run_pipeline.py \
        --input  path/to/noisy.wav \
        --output path/to/clean.wav \
        --iq_checkpoint    checkpoints/iq_best.pt \
        --audio_checkpoint checkpoints/best.pt \
        --config config.yaml

Outputs:
    - final clean wav file at --output path
    - intermediate files in outputs/pipeline/ for debugging:
        after_stage1.wav  ← audio after IQ denoising
        after_stage2.wav  ← audio after audio denoising (same as output)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import torch
import yaml
import soundfile as sf
from pathlib import Path

from src.audio_io import load_audio, save_audio
from src.iq.modulate import modulate
from src.iq.noise import apply_noise
from src.iq.demodulate import demodulate
from src.stft import compute_stft, compute_lps, compute_istft
from src.models.iq_denoiser import IQDenoiser
from src.models.cnn_denoiser import CNNDenoiser


def run_stage1(waveform, model, config, device):
    """
    Stage 1 — IQ denoiser.

    Takes a noisy waveform, converts to IQ, adds channel noise,
    runs through IQ denoiser, converts back to audio.

    This simulates what would happen in a real radio transmission:
        1. Audio gets modulated to IQ for transmission
        2. Channel corrupts the IQ signal (AWGN + IQ imbalance)
        3. IQ denoiser cleans up the corruption
        4. Clean IQ gets demodulated back to audio
    """
    print("  Stage 1: modulating audio to IQ...")
    i_sig, q_sig, n_bits = modulate(waveform)

    print("  Stage 1: applying channel noise...")
    i_noisy, q_noisy = apply_noise(i_sig, q_sig, config)

    # stack I and Q into (2, signal_length) tensor
    iq_tensor = torch.from_numpy(
        np.stack([i_noisy, q_noisy], axis=0).astype(np.float32)
    ).unsqueeze(0).to(device)   # add batch dimension → (1, 2, signal_length)

    print("  Stage 1: running IQ denoiser...")
    with torch.no_grad():
        # process in chunks to avoid running out of memory
        # IQ signals are very long (32x audio length)
        chunk_size  = 16000     # matches IQ_FRAME_LENGTH in iq_dataset.py
        iq_length   = iq_tensor.shape[2]
        output_chunks = []

        for start in range(0, iq_length, chunk_size):
            end   = min(start + chunk_size, iq_length)
            chunk = iq_tensor[:, :, start:end]

            # pad last chunk if needed
            if chunk.shape[2] < chunk_size:
                pad = chunk_size - chunk.shape[2]
                chunk = torch.nn.functional.pad(chunk, (0, pad))

            output_chunk = model(chunk)

            # trim padding from last chunk
            if end == iq_length and iq_length % chunk_size != 0:
                output_chunk = output_chunk[:, :, :chunk.shape[2] - pad]

            output_chunks.append(output_chunk)

        clean_iq = torch.cat(output_chunks, dim=2)

    # convert back to numpy and demodulate
    clean_iq_np = clean_iq.squeeze(0).cpu().numpy()
    i_clean     = clean_iq_np[0]
    q_clean     = clean_iq_np[1]

    print("  Stage 1: demodulating IQ back to audio...")
    reconstructed = demodulate(i_clean, q_clean, n_bits)

    return reconstructed.astype(np.float32)


def run_stage2(waveform, model, device):
    """
    Stage 2 — Audio denoiser (CNNDenoiser).

    Takes a waveform, converts to log power spectrogram,
    runs through CNNDenoiser, converts back to audio.

    This cleans up any remaining artifacts from the IQ
    demodulation process and any residual noise.
    """
    print("  Stage 2: computing spectrogram...")
    mag, phase = compute_stft(waveform)
    lps        = compute_lps(mag)

    # normalize
    lps_mean = float(np.mean(lps))
    lps_std  = max(float(np.std(lps)), 1e-6)
    lps_norm = (lps - lps_mean) / lps_std

    # convert to tensor → (1, 1, freq, time)
    lps_tensor = torch.from_numpy(
        lps_norm.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)

    print("  Stage 2: running audio denoiser...")
    with torch.no_grad():
        pred_lps_norm = model(lps_tensor)

    # denormalize
    pred_lps = pred_lps_norm.squeeze(0).squeeze(0).cpu().numpy() * lps_std + lps_mean

    # convert LPS back to magnitude
    enhanced_mag = np.exp(pred_lps / 2)

    print("  Stage 2: reconstructing audio...")
    enhanced_wav = compute_istft(enhanced_mag, phase)

    return enhanced_wav.astype(np.float32)


def run_pipeline(input_path, output_path, iq_checkpoint, audio_checkpoint, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # create output directories
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    debug_dir = Path("outputs/pipeline")
    debug_dir.mkdir(parents=True, exist_ok=True)

    # load models
    print("Loading models...")
    iq_model = IQDenoiser().to(device)
    iq_ckpt  = torch.load(iq_checkpoint, map_location=device)
    iq_model.load_state_dict(iq_ckpt["model_state"])
    iq_model.eval()
    print(f"  IQ denoiser loaded from: {iq_checkpoint}")

    audio_model = CNNDenoiser().to(device)
    audio_ckpt  = torch.load(audio_checkpoint, map_location=device)
    audio_model.load_state_dict(audio_ckpt["model_state"])
    audio_model.eval()
    print(f"  Audio denoiser loaded from: {audio_checkpoint}\n")

    # load input audio
    print(f"Loading input: {input_path}")
    waveform, sr = load_audio(input_path)
    print(f"  Duration: {len(waveform)/sr:.2f}s  Sample rate: {sr}Hz\n")

    # run stage 1
    print("Running Stage 1 — IQ denoiser...")
    after_stage1 = run_stage1(waveform, iq_model, config, device)
    save_audio(debug_dir / "after_stage1.wav", after_stage1, sr)
    print(f"  Stage 1 complete. Saved to {debug_dir}/after_stage1.wav\n")

    # run stage 2
    print("Running Stage 2 — Audio denoiser...")
    after_stage2 = run_stage2(after_stage1, audio_model, device)
    save_audio(output_path, after_stage2, sr)
    print(f"  Stage 2 complete. Saved to {output_path}\n")

    print("Pipeline complete!")
    print(f"  Input          : {input_path}")
    print(f"  After Stage 1  : {debug_dir}/after_stage1.wav")
    print(f"  Final output   : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",            type=str, required=True)
    parser.add_argument("--output",           type=str, required=True)
    parser.add_argument("--iq_checkpoint",    type=str, required=True)
    parser.add_argument("--audio_checkpoint", type=str, required=True)
    parser.add_argument("--config",           type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_pipeline(
        args.input,
        args.output,
        args.iq_checkpoint,
        args.audio_checkpoint,
        config
    )