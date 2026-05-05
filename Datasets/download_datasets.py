"""
Datasets/download_datasets.py

Downloads LibriSpeech train-clean-100 and the DEMAND noise dataset.

Usage:
    python Datasets/download_datasets.py

Outputs:
    data/librispeech/   ← clean speech (.flac files)
    data/demand/wav/    ← noise recordings (.wav files)
"""

import torchaudio
import io
import pandas as pd
import soundfile as sf
from huggingface_hub import snapshot_download
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


# ── DEMAND — download ─────────────────────────────────────────────────────────
print("Downloading DEMAND noise dataset from HuggingFace...")

DEMAND_DIR = Path("data/demand")
DEMAND_DIR.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="voice-biomarkers/DEMAND-acoustic-noise",
    repo_type="dataset",
    local_dir="data/demand"
)
print("DEMAND download complete.\n")


# ── DEMAND — extract parquet to wav ──────────────────────────────────────────
print("Extracting DEMAND parquet files to wav...")

DEMAND_WAV_DIR = Path("data/demand/wav")
DEMAND_WAV_DIR.mkdir(parents=True, exist_ok=True)

parquet_files = sorted(Path("data/demand/data").glob("*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

for parquet_path in tqdm(parquet_files, desc="Extracting DEMAND"):
    df = pd.read_parquet(parquet_path)

    for _, row in df.iterrows():
        # each row is one channel of one noise environment
        # file_name is like 'ch01', 'ch02' etc
        # parquet filename tells us which environment
        # e.g. train-00000-of-00022 = first environment
        env_id   = parquet_path.stem
        filename = f"{env_id}_{row['file_name']}.wav"
        out_path = DEMAND_WAV_DIR / filename

        # extract wav bytes and save
        buffer = io.BytesIO(row['audio']['bytes'])
        data, samplerate = sf.read(buffer)

        # only save 16kHz — skip 48kHz versions
        if samplerate == 16000:
            sf.write(out_path, data, samplerate)

print("DEMAND extraction complete!")
print(f"Wav files saved to: {DEMAND_WAV_DIR}")
print("\nBoth datasets ready.")
print("Next step: run scripts/mix_datasets.py")