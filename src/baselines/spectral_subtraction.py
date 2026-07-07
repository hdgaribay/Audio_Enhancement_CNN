"""
src/baselines/spectral_subtraction.py

Classical spectral subtraction baseline for speech enhancement.
Estimates noise from the first N frames and subtracts from magnitude spectrum.

Args:
    noisy_wav     : numpy array, noisy waveform
    n_noise_frames: int, number of frames to use for noise estimation (default 10)

Returns:
    enhanced waveform as numpy array

Usage:
    from src.baselines.spectral_subtraction import spectral_subtraction
    enhanced = spectral_subtraction(noisy_wav)
"""

import numpy as np
from src.stft import compute_stft, compute_istft
def spectral_subtraction(noisy_wav, n_noise_frames=10):
    #compute STFT
    mag,phase = compute_stft(noisy_wav)
    # noise estimate
    noise_estimate = np.mean(mag[:,:n_noise_frames],axis = 1, keepdims = True)

    clean_mag = mag-noise_estimate
    clean_mag = np.maximum(clean_mag,0.0)

    clean_waveform = compute_istft(clean_mag,phase)
    return clean_waveform