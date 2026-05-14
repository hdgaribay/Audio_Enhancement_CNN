"""
src/datasets/iq_dataset.py

PyTorch Dataset for training the Stage 1 IQ denoiser.

What it does:
    1. Reads clean wav files from dataset_manifest.csv
    2. Crops audio BEFORE modulating for speed
       (modulating 500 samples is 32x faster than modulating 16000)
    3. Modulates the cropped audio to IQ signals
    4. Applies AWGN and IQ imbalance noise
    5. Crops/pads IQ to exactly IQ_FRAME_LENGTH
    6. Returns (noisy_iq, clean_iq, n_bits) as tensors

Why we crop audio before modulating:
    Modulation makes signals 32x longer (8 samples per symbol x 4 bits
    per audio sample). Modulating a full 16000 sample clip produces
    512000 IQ samples, most of which get thrown away during cropping.
    Instead we crop to 500 audio samples first, then modulate to get
    exactly 16000 IQ samples — 32x less work per sample.

Returns per sample:
    noisy_iq  : (2, IQ_FRAME_LENGTH) — noisy I and Q channels (model input)
    clean_iq  : (2, IQ_FRAME_LENGTH) — clean I and Q channels (training target)
    n_bits    : int — original number of bits (needed for demodulation)

Splits:
    "train" — 90% of manifest train rows, random crop
    "val"   — first 10% of manifest train rows, deterministic crop
    "test"  — manifest test rows, deterministic crop

Usage:
    from src.datasets.iq_dataset import IQDataset
    from torch.utils.data import DataLoader

    train_set = IQDataset("dataset_manifest.csv", config, split="train")
    loader    = DataLoader(train_set, batch_size=16, shuffle=True)

    for noisy_iq, clean_iq, n_bits in loader:
        # noisy_iq : (B, 2, IQ_FRAME_LENGTH)
        # clean_iq : (B, 2, IQ_FRAME_LENGTH)
        # n_bits   : (B,)
        pass

Requirements:
    - Run mix_datasets.py and make_manifest.py first
    - src/iq/modulate.py and src/iq/noise.py must be present
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.audio_io import load_audio
from src.iq.modulate import modulate
from src.iq.noise import apply_noise

# how many IQ samples per training example
# 16000 IQ samples = 500 audio samples (16000 / 32)
# 32 = 8 samples per symbol x 4 bits per audio sample
IQ_FRAME_LENGTH    = 16000
AUDIO_FRAME_LENGTH = IQ_FRAME_LENGTH // 32   # 500 audio samples


class IQDataset(Dataset):
    def __init__(self, manifest_path: str, config: dict, split: str):
        """
        Args:
            manifest_path : path to dataset_manifest.csv
            config        : dict loaded from config.yaml
            split         : "train", "val", or "test"
        """
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.config = config

        # load manifest and split into train/val/test
        df = pd.read_csv(manifest_path)

        if split in ("train", "val"):
            train_df = df[df["split"] == "train"].reset_index(drop=True)
            train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            val_n    = max(1, int(len(train_df) * 0.1))
            if split == "val":
                self.data     = train_df.iloc[:val_n].reset_index(drop=True)
                self.is_train = False
            else:
                self.data     = train_df.iloc[val_n:].reset_index(drop=True)
                self.is_train = True
        else:
            self.data     = df[df["split"] == "test"].reset_index(drop=True)
            self.is_train = False

        print(f"[IQDataset] {split}: {len(self.data)} files loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # step 1 — load clean audio
        clean_wav, _ = load_audio(row["clean_path"])

        # step 2 — crop audio BEFORE modulating
        # much faster: modulate 500 samples instead of 16000
        # random crop for train, fixed crop for val/test
        if self.is_train and len(clean_wav) > AUDIO_FRAME_LENGTH:
            start     = np.random.randint(0, len(clean_wav) - AUDIO_FRAME_LENGTH)
            clean_wav = clean_wav[start:start + AUDIO_FRAME_LENGTH]
        else:
            clean_wav = clean_wav[:AUDIO_FRAME_LENGTH]

        # step 3 — modulate short clip to IQ
        # 500 audio samples → 16000 IQ samples
        i_clean, q_clean, n_bits = modulate(clean_wav)

        # step 4 — stack I and Q into (2, signal_length)
        clean_iq = np.stack([i_clean, q_clean], axis=0)

        # step 5 — crop or pad to exactly IQ_FRAME_LENGTH
        clean_iq = self._crop_or_pad(clean_iq)

        # step 6 — apply noise to get noisy version
        # apply AFTER cropping to avoid wasting computation
        i_noisy, q_noisy = apply_noise(
            clean_iq[0],
            clean_iq[1],
            self.config
        )
        noisy_iq = np.stack([i_noisy, q_noisy], axis=0)

        # step 7 — convert to tensors
        noisy_iq_tensor = torch.from_numpy(noisy_iq.astype(np.float32))
        clean_iq_tensor = torch.from_numpy(clean_iq.astype(np.float32))

        return noisy_iq_tensor, clean_iq_tensor, n_bits

    def _crop_or_pad(self, iq: np.ndarray) -> np.ndarray:
        """
        Crop or pad IQ signal to exactly IQ_FRAME_LENGTH samples.

        Args:
            iq : numpy array of shape (2, signal_length)

        If signal is shorter than IQ_FRAME_LENGTH, pad with zeros.
        If signal is longer, crop from the start.
        """
        length = iq.shape[1]

        if length == IQ_FRAME_LENGTH:
            # already correct length
            return iq
        elif length < IQ_FRAME_LENGTH:
            # pad with zeros on the right
            # ((0,0), (0, pad)) means: don't pad channels, pad signal length
            pad_amount = IQ_FRAME_LENGTH - length
            return np.pad(iq, ((0, 0), (0, pad_amount)))
        else:
            # crop from start
            return iq[:, :IQ_FRAME_LENGTH]


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset = IQDataset("dataset_manifest.csv", config, split="train")
    print(f"Dataset length : {len(dataset)}")

    noisy_iq, clean_iq, n_bits = dataset[0]
    print(f"noisy_iq shape : {noisy_iq.shape}")
    print(f"clean_iq shape : {clean_iq.shape}")
    print(f"n_bits         : {n_bits}")
    print(f"noisy_iq range : [{noisy_iq.min():.3f}, {noisy_iq.max():.3f}]")
    print(f"clean_iq range : [{clean_iq.min():.3f}, {clean_iq.max():.3f}]")