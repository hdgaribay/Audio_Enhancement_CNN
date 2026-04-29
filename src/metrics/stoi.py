"""
src/metrics/stoi.py

Wrapper around the stoi library.

Args:
    clean_wav    : numpy array, reference clean waveform
    enhanced_wav : numpy array, model output waveform

Returns:
    STOI score (float), range 0 to 1, higher is better

Usage:
    from src.metrics.stoi import compute_stoi
    score = compute_stoi(clean_wav, enhanced_wav)
"""
from pystoi import stoi

def compute_stoi(clean_wav, enhanced_wav):
    score = stoi(clean_wav, enhanced_wav, 16000, extended=False)
    return score