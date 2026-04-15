"""
stft.py

Contains 3 functions:
    compute_stft(waveform)          - takes raw audio array, returns magnitude and phase
    compute_lps(magnitude)          - converts magnitude to log power spectrogram
    compute_istft(magnitude, phase) - reconstructs waveform from magnitude and phase

Usage:
    Imported by other scripts:
    from src.stft import compute_stft, compute_lps, compute_istft

Requirements:
    - Dataset extracted using extract_dataset.py
    - Audio loaded using load_audio() from src.audio_io
"""

import numpy as np
import torch
import torchaudio

def compute_stft(waveform):
    waveform_tensor = torch.from_numpy(waveform.astype(np.float32))
    stft_result = torch.stft(
        waveform_tensor,
        n_fft=510,
        hop_length=100,
        win_length=510,
        window=torch.hann_window(510),
        return_complex=True
    )
    magnitude = torch.abs(stft_result)
    phase     = torch.angle(stft_result)
    magnitude = magnitude.numpy()
    phase     = phase.numpy()
    return magnitude, phase

def compute_lps(magnitude):
    lps = np.log(magnitude ** 2 + 1e-8)
    return lps

def compute_istft(magnitude, phase):
    magnitude = np.pad(magnitude, ((0, 1), (0, 0)))
    phase     = np.pad(phase,     ((0, 1), (0, 0)))
    complex_stft   = magnitude * np.exp(1j * phase)
    complex_tensor = torch.from_numpy(complex_stft.astype(np.complex64))
    istft_result   = torch.istft(
        complex_tensor,
        n_fft=510,
        hop_length=100,
        win_length=510,
        window=torch.hann_window(510)
    )
    return istft_result.numpy()