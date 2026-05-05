"""
src/datasets/iq_dataset.py

PyTorch Dataset for training the Stage 1 IQ denoiser.

What it does:
    1. Reads clean wav files from dataset_manifest.csv
    2. Loads each wav file as a numpy array
    3. Modulates the clean audio to IQ signals
    4. Applies AWGN and IQ imbalance noise
    5. Crops both clean and noisy IQ to a fixed length
    6. Returns (noisy_iq, clean_iq, n_bits) as tensors

Why we crop:
    A single audio clip of 16000 samples becomes 512000 IQ samples
    after modulation (32x longer). Processing 512000 samples at once
    during training would be extremely slow and memory intensive.
    Instead we crop to IQ_FRAME_LENGTH (16000 IQ samples) which
    represents ~500 audio samples worth of data — enough context
    for the model to learn noise patterns without being too slow.

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

# how many IQ samples to use per training example
# 16000 IQ samples = ~500 audio samples worth of data
# this is 32x shorter than the full IQ signal (512000)
# which makes training much faster without losing too much context
IQ_FRAME_LENGTH = 16000


class IQDataset(Dataset):
    def __init__(self, manifest_path: str, config: dict, split: str):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.config = config

        df = pd.read_csv(manifest_path)

        if split in ("train", "val"):
            train_df = df[df["split"] == "train"].reset_index(drop=True)
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

        # load clean audio only — we generate noisy version ourselves
        clean_wav, _ = load_audio(row["clean_path"])

        # modulate to IQ
        i_clean, q_clean, n_bits = modulate(clean_wav)

        # stack into (2, signal_length)
        clean_iq = np.stack([i_clean, q_clean], axis=0)

        # crop to IQ_FRAME_LENGTH
        length = clean_iq.shape[1]
        if self.is_train and length > IQ_FRAME_LENGTH:
            start = np.random.randint(0, length - IQ_FRAME_LENGTH)
        else:
            start = 0

        clean_iq = self._crop_or_pad(clean_iq, start)

        # apply noise after cropping to avoid wasting computation
        i_noisy, q_noisy = apply_noise(
            clean_iq[0],
            clean_iq[1],
            self.config
        )
        noisy_iq = np.stack([i_noisy, q_noisy], axis=0)

        # convert to tensors
        noisy_iq_tensor = torch.from_numpy(noisy_iq.astype(np.float32))
        clean_iq_tensor = torch.from_numpy(clean_iq.astype(np.float32))

        return noisy_iq_tensor, clean_iq_tensor, n_bits

    def _crop_or_pad(self, iq: np.ndarray, start: int = 0) -> np.ndarray:
        """
        Crop or pad IQ signal to exactly IQ_FRAME_LENGTH samples.
        If signal is shorter than IQ_FRAME_LENGTH, pad with zeros.
        If signal is longer, crop from start position.
        """
        length = iq.shape[1]

        if length == IQ_FRAME_LENGTH:
            return iq
        elif length < IQ_FRAME_LENGTH:
            pad_amount = IQ_FRAME_LENGTH - length
            return np.pad(iq, ((0, 0), (0, pad_amount)))
        else:
            return iq[:, start:start + IQ_FRAME_LENGTH]


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset = IQDataset("dataset_manifest.csv", config, split="train")
    print(f"Dataset length: {len(dataset)}")

    noisy_iq, clean_iq, n_bits = dataset[0]
    print(f"noisy_iq shape : {noisy_iq.shape}")
    print(f"clean_iq shape : {clean_iq.shape}")
    print(f"n_bits         : {n_bits}")
    print(f"noisy_iq range : [{noisy_iq.min():.3f}, {noisy_iq.max():.3f}]")
    print(f"clean_iq range : [{clean_iq.min():.3f}, {clean_iq.max():.3f}]")