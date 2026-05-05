"""
dataset.py

PyTorch Dataset for VoiceBank-DEMAND speech enhancement.

What it does:
    1. Reads clean/noisy wav pairs from dataset_manifest.csv
    2. Loads each wav file as a numpy array
    3. Fixes each waveform to exactly 30700 samples (pad or random-crop for train,
       head-crop for val/test)
    4. Computes STFT → returns noisy_lps, noisy_phase, clean_lps as tensors

Returns per sample:
    noisy_lps   : (256, 308) — normalized log power spectrogram of noisy audio  (model input)
    noisy_phase : (256, 308) — phase of noisy audio                              (used in ISTFT at inference)
    clean_lps   : (256, 308) — normalized log power spectrogram of clean audio  (training target)
    lps_mean    : scalar     — mean used for normalization  (needed to denormalize at inference)
    lps_std     : scalar     — std  used for normalization  (needed to denormalize at inference)

Splits:
    "train" — 90% of manifest train rows, random crop
    "val"   — first 10% of manifest train rows, deterministic crop
    "test"  — manifest test rows, deterministic crop

Usage:
    from src.dataset import SpeechDataset
    from torch.utils.data import DataLoader

    train_set = SpeechDataset("dataset_manifest.csv", split="train")
    loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)

    for noisy_lps, noisy_phase, clean_lps, lps_mean, lps_std in loader:
        # noisy_lps   : (B, 256, 308)  normalized
        # noisy_phase : (B, 256, 308)
        # clean_lps   : (B, 256, 308)  normalized (same stats as noisy)
        # lps_mean    : (B,)
        # lps_std     : (B,)
        pass

Requirements:
    - Run extract_dataset.py and make_manifest.py first
    - src/stft.py and src/audio_io.py must be present
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from src.audio_io import load_audio
from src.stft import compute_stft, compute_lps

FRAME_LENGTH = 30700


class SpeechDataset(Dataset):
    def __init__(self, manifest_path: str, split: str):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"
        df = pd.read_csv(manifest_path)

        if split in ("train", "val"):
            train_df = df[df["split"] == "train"].reset_index(drop=True)
            val_n = max(1, int(len(train_df) * 0.1))
            if split == "val":
                self.data = train_df.iloc[:val_n].reset_index(drop=True)
                self.is_train = False
            else:
                self.data = train_df.iloc[val_n:].reset_index(drop=True)
                self.is_train = True
        else:
            self.data = df[df["split"] == "test"].reset_index(drop=True)
            self.is_train = False

        print(f"[SpeechDataset] {split}: {len(self.data)} pairs loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        clean_wav, _ = load_audio(row["clean_path"])
        noisy_wav, _ = load_audio(row["noisy_path"])

        # One shared crop start so clean and noisy are temporally aligned.
        length = min(len(clean_wav), len(noisy_wav))
        if self.is_train and length > FRAME_LENGTH:
            start = np.random.randint(0, length - FRAME_LENGTH)
        else:
            start = 0

        clean_wav = self._fix_length(clean_wav, start)
        noisy_wav = self._fix_length(noisy_wav, start)

        noisy_mag, noisy_phase = compute_stft(noisy_wav)
        clean_mag, _           = compute_stft(clean_wav)

        noisy_lps = compute_lps(noisy_mag)
        clean_lps = compute_lps(clean_mag)

        noisy_lps   = torch.from_numpy(noisy_lps.astype(np.float32))
        noisy_phase = torch.from_numpy(noisy_phase.astype(np.float32))
        clean_lps   = torch.from_numpy(clean_lps.astype(np.float32))

        lps_mean = float(noisy_lps.mean())
        lps_std  = max(float(noisy_lps.std()), 1e-6)
        noisy_lps = (noisy_lps - lps_mean) / lps_std
        clean_lps = (clean_lps - lps_mean) / lps_std

        return (
            noisy_lps,
            noisy_phase,
            clean_lps,
            torch.tensor(lps_mean),
            torch.tensor(lps_std),
        )

    def _fix_length(self, wav: np.ndarray, start: int = 0) -> np.ndarray:
        length = len(wav)
        if length == FRAME_LENGTH:
            return wav
        elif length < FRAME_LENGTH:
            return np.pad(wav, (0, FRAME_LENGTH - length))
        else:
            return wav[start:start + FRAME_LENGTH]


if __name__ == "__main__":
    dataset = SpeechDataset("dataset_manifest.csv", split="train")
    print(f"Dataset length: {len(dataset)}")

    noisy_lps, noisy_phase, clean_lps, lps_mean, lps_std = dataset[0]
    print(f"noisy_lps shape   : {noisy_lps.shape}")
    print(f"noisy_phase shape : {noisy_phase.shape}")
    print(f"clean_lps shape   : {clean_lps.shape}")
    print(f"lps_mean          : {lps_mean:.4f}")
    print(f"lps_std           : {lps_std:.4f}")
