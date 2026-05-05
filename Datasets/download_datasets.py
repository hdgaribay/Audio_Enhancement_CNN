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
print("Downloading DEMAND noise dataset...")

DEMAND_URL = "https://zenodo.org/records/1227121/files/DEMAND.zip"DEMAND_DIR = Path("data/demand")
DEMAND_DIR.mkdir(parents=True, exist_ok=True)

# stream=True means we download in chunks instead of loading
# the whole file into memory at once — important for large files
response = requests.get(DEMAND_URL, stream=True)
total_size = int(response.headers.get("content-length", 0))
chunk_size = 1024 * 1024  # 1MB chunks

print(f"File size: {total_size / 1e9:.1f} GB")

# download with a progress bar
data = b""
with tqdm(total=total_size, unit="B", unit_scale=True, desc="DEMAND") as bar:
    for chunk in response.iter_content(chunk_size=chunk_size):
        data += chunk
        bar.update(len(chunk))

print("Extracting DEMAND zip...")
with zipfile.ZipFile(io.BytesIO(data)) as z:
    z.extractall(DEMAND_DIR)

print("DEMAND download complete.\n")
print("Both datasets ready.")
print("Next step: run scripts/mix_dataset.py to create clean/noisy pairs.")