"""
dataset.py

PyTorch Dataset for VoiceBank-DEMAND speech enhancement.

What it does:
    1. Reads clean/noisy wav pairs from dataset_manifest.csv
    2. Loads each wav file as a numpy array
    3. Fixes each waveform to exactly 30700 samples (pad or crop)
    4. Computes STFT → returns noisy_lps, noisy_phase, clean_mag as tensors

Returns per sample:
    noisy_lps   : (256, 308) — log power spectrogram of noisy audio  (model input)
    noisy_phase : (256, 308) — phase of noisy audio                  (used in ISTFT at inference)
    clean_mag   : (256, 308) — magnitude of clean audio              (training target)

Usage:
    from src.dataset import SpeechDataset
    from torch.utils.data import DataLoader

    train_set = SpeechDataset("dataset_manifest.csv", split="train")
    loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)

    for noisy_lps, noisy_phase, clean_mag in loader:
        # noisy_lps   : (B, 256, 308)
        # noisy_phase : (B, 256, 308)
        # clean_mag   : (B, 256, 308)
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
        assert split in ("train", "test"), f"split must be 'train' or 'test', got '{split}'"
        df = pd.read_csv(manifest_path)
        self.data = df[df["split"] == split].reset_index(drop = True)
        print(f"[SpeechDataset] {split}: {len(self.data)} pairs loaded")
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        row= self.data.iloc[idx]
        clean_wav, _ = load_audio(row["clean_path"])
        noisy_wav, _ = load_audio(row["noisy_path"]) 
        # fix length
        clean_wav = self._fix_length(clean_wav, is_train=(row["split"] == "train"))
        noisy_wav = self._fix_length(noisy_wav, is_train=(row["split"] == "train"))
        # compute STFT
        noisy_mag, noisy_phase = compute_stft(noisy_wav)
        clean_mag, _           = compute_stft(clean_wav)
        noisy_lps              = compute_lps(noisy_mag)
        # return tensors
        noisy_lps   = torch.from_numpy(noisy_lps.astype(np.float32))
        noisy_phase = torch.from_numpy(noisy_phase.astype(np.float32))
        clean_mag   = torch.from_numpy(clean_mag.astype(np.float32)) 
        return noisy_lps, noisy_phase, clean_mag
    def _fix_length(self, wav: np.ndarray, is_train: bool):
        length = len(wav)
        if length == FRAME_LENGTH:
            return wav
        elif length < FRAME_LENGTH:
            pad_amount = FRAME_LENGTH-length
            return np.pad(wav,(0,pad_amount))
        else:

            if is_train:
                start = np.random.randint(0,length-FRAME_LENGTH)
            else:
                start = 0
            return wav[start: start+FRAME_LENGTH]

if __name__ == "__main__":
    dataset = SpeechDataset("dataset_manifest.csv", split="train")
    print(f"Dataset length: {len(dataset)}")

    noisy_lps, noisy_phase, clean_mag = dataset[0]
    print(f"noisy_lps shape   : {noisy_lps.shape}")
    print(f"noisy_phase shape : {noisy_phase.shape}")
    print(f"clean_mag shape   : {clean_mag.shape}")

