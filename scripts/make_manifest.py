"""
make_manifest.py

Scans data/voicebank_demand/ for extracted wav files and organizes
clean and noisy pairs by id into dataset_manifest.csv.

Usage:
    python scripts/make_manifest.py

Requirements:
    - Run extract_dataset.py before this script
    - Must be run from the project root (Audio_Enhancement_CNN/)
"""

import pandas as pd
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT  = Path("data/voicebank_demand")
OUTPUT_CSV = Path("dataset_manifest.csv")

# ── Scan Folders ──────────────────────────────────────────────────────────────
rows = []

for split in ["train", "test"]:
    clean_dir = DATA_ROOT / split / "clean"
    noisy_dir = DATA_ROOT / split / "noisy"

    clean_files = sorted(clean_dir.glob("*.wav"))

    print(f"Found {len(clean_files)} {split} files...")

    for clean_path in clean_files:
        file_id    = clean_path.stem
        noisy_path = noisy_dir / clean_path.name

        if not noisy_path.exists():
            print(f"  WARNING: missing noisy file for {file_id}, skipping...")
            continue

        rows.append({
            "id":         file_id,
            "split":      split,
            "clean_path": str(clean_path),
            "noisy_path": str(noisy_path)
        })

# ── Write CSV ─────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
train_count = len(df[df['split'] == 'train'])
test_count  = len(df[df['split'] == 'test'])

print(f"\nManifest written to {OUTPUT_CSV}")
print(f"  Train pairs : {train_count}")
print(f"  Test pairs  : {test_count}")
print(f"  Total pairs : {len(df)}")
print(f"\nFirst few rows:")
print(df.head())