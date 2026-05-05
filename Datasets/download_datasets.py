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
# ── DEMAND ────────────────────────────────────────────────────────────────────
print("Downloading DEMAND noise dataset (16k versions)...")

DEMAND_DIR = Path("data/demand")
DEMAND_DIR.mkdir(parents=True, exist_ok=True)

# each environment is a separate zip file — we only need 16k to match our sample rate
DEMAND_FILES = [
    "DKITCHEN_16k.zip", "DLIVING_16k.zip",  "DWASHING_16k.zip",
    "NFIELD_16k.zip",   "NPARK_16k.zip",    "NRIVER_16k.zip",
    "OHALLWAY_16k.zip", "OMEETING_16k.zip", "OOFFICE_16k.zip",
    "PCAFETER_16k.zip", "PRESTO_16k.zip",   "PSTATION_16k.zip",
    "SPSQUARE_16k.zip", "STRAFFIC_16k.zip", "TBUS_16k.zip",
    "TCAR_16k.zip",     "TMETRO_16k.zip",
]

BASE_URL = "https://zenodo.org/records/1227121/files"

for filename in DEMAND_FILES:
    print(f"\nDownloading {filename}...")
    url = f"{BASE_URL}/{filename}?download=1"
    response = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
    total_size = int(response.headers.get("content-length", 0))

    data = b""
    with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            data += chunk
            bar.update(len(chunk))

    print(f"Extracting {filename}...")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(DEMAND_DIR)

print("\nDEMAND download complete.")
print("Both datasets ready.")
print("Next step: run scripts/mix_dataset.py to create clean/noisy pairs.")