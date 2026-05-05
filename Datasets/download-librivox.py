"""
scripts/download_librispeech.py

Downloads and extracts the LibriSpeech subsets we need for training and testing.

What it downloads:
    train-clean-100  (~6 GB)  — 100 hours of clean speech for training
    test-clean       (~346 MB) — standard test set

Output structure after running:
    data/librispeech/
        train-clean-100/
            <speaker_id>/
                <chapter_id>/
                    *.flac
        test-clean/
            <speaker_id>/
                <chapter_id>/
                    *.flac

Usage:
    python scripts/download_librispeech.py

Note:
    Run from the project root folder (Audio_Enhancement_CNN/).
    train-clean-100 is about 6 GB — this will take a while depending on your connection.
"""

import os
import tarfile
import urllib.request
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/librispeech")

# These are the two splits we need
SPLITS = {
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "test-clean":      "https://www.openslr.org/resources/12/test-clean.tar.gz",
}


# ── Progress bar for download ─────────────────────────────────────────────────
def make_progress_hook(filename):
    """Prints download progress as a percentage."""
    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb_done  = downloaded  / 1e6
            mb_total = total_size  / 1e6
            print(f"\r  {filename}: {pct:.1f}%  ({mb_done:.0f} / {mb_total:.0f} MB)", end="", flush=True)
    return hook


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, url in SPLITS.items():
        tar_path = OUTPUT_DIR / f"{split_name}.tar.gz"
        split_dir = OUTPUT_DIR / split_name

        # ── Step 1: Download ──────────────────────────────────────────────────
        if tar_path.exists():
            print(f"[{split_name}] Archive already exists, skipping download.")
        else:
            print(f"[{split_name}] Downloading from {url}")
            urllib.request.urlretrieve(url, tar_path, reporthook=make_progress_hook(split_name))
            print()  # newline after progress bar

        # ── Step 2: Extract ───────────────────────────────────────────────────
        if split_dir.exists():
            print(f"[{split_name}] Already extracted, skipping.")
        else:
            print(f"[{split_name}] Extracting...")
            with tarfile.open(tar_path, "r:gz") as tar:
                # LibriSpeech tarballs extract into a LibriSpeech/ subfolder.
                # We strip that prefix so files land directly in OUTPUT_DIR.
                members = tar.getmembers()
                for m in members:
                    # e.g. "LibriSpeech/train-clean-100/..." -> "train-clean-100/..."
                    parts = Path(m.name).parts
                    if len(parts) > 1 and parts[0] == "LibriSpeech":
                        m.name = str(Path(*parts[1:]))
                tar.extractall(OUTPUT_DIR, members=members)
            print(f"[{split_name}] Extracted to {OUTPUT_DIR / split_name}")

        # ── Step 3: Verify ────────────────────────────────────────────────────
        flac_files = list(split_dir.rglob("*.flac"))
        print(f"[{split_name}] Found {len(flac_files):,} .flac files.")

    print("\nLibriSpeech download complete.")
    print(f"Data is in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()