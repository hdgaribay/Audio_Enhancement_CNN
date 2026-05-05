"""
Datasets/download_datasets.py

Downloads LibriSpeech train-clean-100 and the DEMAND noise dataset.

Usage:
    python Datasets/download_datasets.py

Outputs:
    data/librispeech/   ← clean speech (.flac files)
    data/demand/        ← noise recordings (.wav files)
"""

import torchaudio
import requests
import zipfile
import io
from pathlib import Path
from tqdm import tqdm


# ── LibriSpeech ───────────────────────────────────────────────────────────────
print("Downloading LibriSpeech train-clean-100...")
print("Warning: this is ~6GB, it may take a while.\n")

Path("data/librispeech").mkdir(parents=True, exist_ok=True)

torchaudio.datasets.LIBRISPEECH(
    root="data/librispeech",
    url="train-clean-100",
    download=True
)
print("LibriSpeech download complete.\n")


# ── DEMAND ────────────────────────────────────────────────────────────────────
print("Downloading DEMAND noise dataset from HuggingFace...")

from huggingface_hub import snapshot_download


DEMAND_DIR = Path("data/demand")
DEMAND_DIR.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="voice-biomarkers/DEMAND-acoustic-noise",
    repo_type="dataset",
    local_dir="data/demand"
)

# ── Extract DEMAND parquet files to wav ───────────────────────────────────────
print("Extracting DEMAND parquet files to wav...")
import pandas as pd
import soundfile as sf
import io

DEMAND_WAV_DIR = Path("data/demand/wav")
DEMAND_WAV_DIR.mkdir(parents=True, exist_ok=True)

parquet_files = sorted(Path("data/demand/data").glob("*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

for parquet_path in tqdm(parquet_files, desc="Extracting DEMAND"):
    df = pd.read_parquet(parquet_path)
    
    for _, row in df.iterrows():
        # each row is one channel of one noise environment
        # file_name is like 'ch01', 'ch02' etc
        # we need to figure out which environment this is from
        # the parquet filename tells us e.g. train-00000 = first environment
        env_id    = parquet_path.stem    # e.g. train-00000-of-00022
        filename  = f"{env_id}_{row['file_name']}.wav"
        out_path  = DEMAND_WAV_DIR / filename
        
        # extract wav bytes and save
        buffer = io.BytesIO(row['audio']['bytes'])
        data, samplerate = sf.read(buffer)
        
        # only save 16kHz files — skip 48kHz
        if samplerate == 16000:
            sf.write(out_path, data, samplerate)

print(f"DEMAND extraction complete!")
print(f"Wav files saved to: {DEMAND_WAV_DIR}")

print("DEMAND download complete.")
print("\nBoth datasets ready.")
print("Next step: run scripts/mix_datasets.py")