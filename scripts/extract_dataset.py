"""
extract_dataset.py

Run this script ONCE after cloning the repo to convert the HuggingFace
parquet files into individual .wav files in the correct folder structure.

Usage:
    python scripts/extract_dataset.py

Requirements:
    - Download dataset via:
        python -c "
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id='JacobLinCool/VoiceBank-DEMAND-16k',
            repo_type='dataset',
            local_dir='data/voicebank_demand'
        )
        "
    - Run from the project root folder (Audio_Enhancement_CNN/)
    - Conda environment must be activated:
        conda activate Audio_Enhancement_CNN

Output structure after running:
    data/voicebank_demand/
        train/
            clean/   
            noisy/ 
        test/
            clean/   
            noisy/  
"""

import pandas as pd
import soundfile as sf
from pathlib import Path
import io

# ── Folder Setup ────────────────────────────────────────────────────────────
Path("data/voicebank_demand/train/clean").mkdir(parents=True, exist_ok=True)
Path("data/voicebank_demand/train/noisy").mkdir(parents=True, exist_ok=True)
Path("data/voicebank_demand/test/clean").mkdir(parents=True, exist_ok=True)
Path("data/voicebank_demand/test/noisy").mkdir(parents=True, exist_ok=True)

# ── Helper ───────────────────────────────────────────────────────────────────
def save_audio(audio_dict, output_path):
    """Extract raw bytes from parquet cell and save as .wav file."""
    buffer = io.BytesIO(audio_dict['bytes'])
    data, samplerate = sf.read(buffer)
    sf.write(output_path, data, samplerate)

# ── Load Parquet Files ───────────────────────────────────────────────────────
print("Loading train parquet files...")
train_dfs = []
for i in range(5):
    path = f"data/voicebank_demand/data/train-0000{i}-of-00005.parquet"
    train_dfs.append(pd.read_parquet(path))
    print(f"  Loaded {path}")
train_df = pd.concat(train_dfs, ignore_index=True)

print("Loading test parquet file...")
test_df = pd.read_parquet("data/voicebank_demand/data/test-00000-of-00001.parquet")

print(f"\nTrain pairs: {len(train_df)}")
print(f"Test pairs:  {len(test_df)}")

# ── Extract Train Wav Files ──────────────────────────────────────────────────
print(f"\nExtracting {len(train_df)} train pairs to wav...")
for i, row in train_df.iterrows():
    save_audio(row['clean'], f"data/voicebank_demand/train/clean/{row['id']}.wav")
    save_audio(row['noisy'], f"data/voicebank_demand/train/noisy/{row['id']}.wav")
    if i % 500 == 0:
        print(f"  {i}/{len(train_df)} done...")

# ── Extract Test Wav Files ───────────────────────────────────────────────────
print(f"\nExtracting {len(test_df)} test pairs to wav...")
for i, row in test_df.iterrows():
    save_audio(row['clean'], f"data/voicebank_demand/test/clean/{row['id']}.wav")
    save_audio(row['noisy'], f"data/voicebank_demand/test/noisy/{row['id']}.wav")
    if i % 100 == 0:
        print(f"  {i}/{len(test_df)} done...")

# ── Verify ───────────────────────────────────────────────────────────────────
print("\nVerifying extracted files...")
train_clean = list(Path("data/voicebank_demand/train/clean").glob("*.wav"))
train_noisy = list(Path("data/voicebank_demand/train/noisy").glob("*.wav"))
test_clean  = list(Path("data/voicebank_demand/test/clean").glob("*.wav"))
test_noisy  = list(Path("data/voicebank_demand/test/noisy").glob("*.wav"))

print(f"  Train clean : {len(train_clean)} files")
print(f"  Train noisy : {len(train_noisy)} files")
print(f"  Test clean  : {len(test_clean)} files")
print(f"  Test noisy  : {len(test_noisy)} files")

assert len(train_clean) == len(train_noisy), "Train clean/noisy count mismatch!"
assert len(test_clean)  == len(test_noisy),  "Test clean/noisy count mismatch!"
print("\nAll counts match. Dataset ready.")