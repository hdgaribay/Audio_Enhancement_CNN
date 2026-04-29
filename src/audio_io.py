"""
audio_io.py

Contains load function that takes a path, loads wav file and returns audio as numpy array.
Contains save function that takes a path, a numpy array, and sample rate and writes wav to disk.

Usage:
    from src.audio_io import load_audio, save_audio

Requirements:
    - Run extract_dataset.py to obtain dataset
    - Ensure you have dataset_manifest.csv
"""

import soundfile as sf
import numpy as np
from pathlib import Path


def load_audio(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    data, samplerate = sf.read(path)
    if samplerate != 16000:
        raise ValueError(f"Expected sample rate 16000, got {samplerate}")
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, samplerate


def save_audio(path, data, samplerate):
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(save_path, data, samplerate)
