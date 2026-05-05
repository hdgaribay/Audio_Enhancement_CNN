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

print("DEMAND download complete.")
print("\nBoth datasets ready.")
print("Next step: run scripts/mix_datasets.py")