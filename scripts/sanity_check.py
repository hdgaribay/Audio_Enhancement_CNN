"""
load 10 random sound files from the dataset, prints sample rate and duration. plots a spectrogram to /outputs/figures/

usage: python scripts/sanity_check.py

Before running:
install dataset using extract_dataset.py
"""

import pandas as pd
import random
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from pathlib import Path

CSV_PATH = Path("dataset_manifest.csv")
FIGURE_PATH = Path("outputs/figures")
NUM_SAMPLES = 10

df = pd.read_csv(CSV_PATH)
random_samples = df.sample(NUM_SAMPLES)

FIGURE_PATH.mkdir(parents=True,exist_ok=True)

for i, row in random_samples.iterrows():
    print(f"file id: {row['id']}")
    clean_data, samplerate = sf.read(row['clean_path'])
    noisy_data, samplerate = sf.read(row['noisy_path'])
    print(f"Sample Rate: {samplerate}")
    print(f"Duration: {len(clean_data)/samplerate:.2f}")
plt.figure(figsize=(10, 4))
plt.specgram(clean_data, Fs=samplerate)
plt.title("Clean Spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.savefig(FIGURE_PATH / "sanity_check_spectrogram.png")
plt.close()
print(f"\nSpectrogram saved to {FIGURE_PATH}/sanity_check_spectrogram.png")