"""
scripts/mix_dataset.py

Mixes LibriSpeech clean speech with DEMAND noise to create clean/noisy pairs.

Usage:
    python scripts/mix_dataset.py

Outputs:
    data/mixed/
        train/
            clean/   ← original LibriSpeech speech
            noisy/   ← speech + DEMAND noise mixed together
        test/
            clean/
            noisy/

Run this before make_manifest.py
"""

import numpy as np
import soundfile as sf
import soxr
from pathlib import Path
from tqdm import tqdm
import random
import yaml

# ── Load config ───────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

TARGET_SR  = config["sample_rate"]        # 16000
SNR_MIN    = config["mix_snr_min"]        # 0
SNR_MAX    = config["mix_snr_max"]        # 20

LIBRISPEECH_DIR = Path("data/librispeech/LibriSpeech/train-clean-100")
DEMAND_DIR      = Path("data/demand")
OUTPUT_DIR      = Path("data/mixed")

# 90% train, 10% test split
TRAIN_RATIO = 0.9

random.seed(config["seed"])
np.random.seed(config["seed"])

# ── Helper functions ──────────────────────────────────────────────────────────

def load_audio(path):
    """Load an audio file and resample to 16kHz if needed."""
    data, sr = sf.read(path)
    if data.ndim == 2:          # if stereo, average channels to mono
        data = data.mean(axis=1)
    if sr != TARGET_SR:         # if wrong sample rate, resample
        data = soxr.resample(data, sr, TARGET_SR)
    return data.astype(np.float32)


def mix_at_snr(speech, noise, snr_db):
    """
    Mix speech and noise at a target SNR level.
    SNR (Signal to Noise Ratio) controls how loud the noise is.
    Higher SNR = quieter noise = cleaner audio.
    """
    # measure the power (average loudness) of each signal
    speech_power = np.mean(speech ** 2)
    noise_power  = np.mean(noise ** 2)

    # figure out how much to scale the noise so we hit the target SNR
    # this is the standard formula: SNR = 10 * log10(speech_power / noise_power)
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / (noise_power + 1e-8))

    return speech + scale * noise


def get_all_speech_files():
    """Collect all .flac files from LibriSpeech."""
    files = list(LIBRISPEECH_DIR.rglob("*.flac"))
    random.shuffle(files)
    return files


def get_all_noise_files():
    """Collect all channel wav files from DEMAND, grouped by environment."""
    noise_files = list(DEMAND_DIR.rglob("ch*.wav"))
    return noise_files


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # create output folders
    for split in ["train", "test"]:
        Path(f"data/mixed/{split}/clean").mkdir(parents=True, exist_ok=True)
        Path(f"data/mixed/{split}/noisy").mkdir(parents=True, exist_ok=True)

    speech_files = get_all_speech_files()
    noise_files  = get_all_noise_files()

    print(f"Found {len(speech_files)} speech files")
    print(f"Found {len(noise_files)} noise files")

    # split into train and test
    n_train = int(len(speech_files) * TRAIN_RATIO)
    train_files = speech_files[:n_train]
    test_files  = speech_files[n_train:]

    print(f"Train: {len(train_files)} | Test: {len(test_files)}")
    print("Mixing...")

    for split, files in [("train", train_files), ("test", test_files)]:
        for speech_path in tqdm(files, desc=f"Mixing {split}"):

            # load clean speech
            speech = load_audio(speech_path)

            # pick a random noise file and load it
            noise_path = random.choice(noise_files)
            noise      = load_audio(noise_path)

            # if noise is shorter than speech, tile it to be long enough
            # think of it like tiling a short bathroom floor tile
            # to cover a longer floor
            if len(noise) < len(speech):
                repeats = int(np.ceil(len(speech) / len(noise)))
                noise   = np.tile(noise, repeats)

            # pick a random start point in the noise
            # so we don't always use the beginning
            max_start = len(noise) - len(speech)
            start     = random.randint(0, max_start)
            noise     = noise[start:start + len(speech)]

            # pick a random SNR between 0 and 20
            snr_db = random.uniform(SNR_MIN, SNR_MAX)

            # mix them
            noisy = mix_at_snr(speech, noise, snr_db)

            # clip to [-1, 1] to prevent audio distortion
            noisy = np.clip(noisy, -1.0, 1.0)

            # save both versions using the original filename
            stem = speech_path.stem     # filename without extension
            clean_out = Path(f"data/mixed/{split}/clean/{stem}.wav")
            noisy_out = Path(f"data/mixed/{split}/noisy/{stem}.wav")

            sf.write(clean_out, speech, TARGET_SR)
            sf.write(noisy_out, noisy,  TARGET_SR)

    print("\nMixing complete!")
    print("Next step: run scripts/make_manifest.py")


if __name__ == "__main__":
    main()