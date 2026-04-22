"""
src/metrics/pesq.py

Wrapper around the pesq library.

Args:
    clean_wav    : numpy array, reference clean waveform
    enhanced_wav : numpy array, model output waveform

Returns:
    PESQ score (float), range -0.5 to 4.5, higher is better

Usage:
    from src.metrics.pesq import compute_pesq
    score = compute_pesq(clean_wav, enhanced_wav)
"""
import numpy as np
from pesq import pesq

def compute_pesq(clean_wav, enhanced_wav):
    score = pesq(16000, clean_wav, enhanced_wav, "wb")
    return score