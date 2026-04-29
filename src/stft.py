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

_WIN_SIZE = 510
_HOP      = 100
# Created once at import time; reused on every call to avoid per-call allocation.
_WINDOW   = torch.hann_window(_WIN_SIZE)


def compute_stft(waveform):
    waveform_tensor = torch.from_numpy(waveform.astype(np.float32))
    stft_result = torch.stft(
        waveform_tensor,
        n_fft=_WIN_SIZE,
        hop_length=_HOP,
        win_length=_WIN_SIZE,
        window=_WINDOW,
        return_complex=True
    )
    magnitude = torch.abs(stft_result).numpy()
    phase     = torch.angle(stft_result).numpy()
    return magnitude, phase


def compute_lps(magnitude):
    return np.log(magnitude ** 2 + 1e-8)


def compute_istft(magnitude, phase):
    complex_stft   = magnitude * np.exp(1j * phase)
    complex_tensor = torch.from_numpy(complex_stft.astype(np.complex64))
    istft_result   = torch.istft(
        complex_tensor,
        n_fft=_WIN_SIZE,
        hop_length=_HOP,
        win_length=_WIN_SIZE,
        window=_WINDOW
    )
    return istft_result.numpy()
