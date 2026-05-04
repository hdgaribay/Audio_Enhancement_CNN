"""
stft.py

Contains 3 functions:
    compute_stft(waveform)          - takes raw audio array, returns magnitude and phase
    compute_lps(magnitude)          - converts magnitude to log power spectrogram
    compute_istft(magnitude, phase) - reconstructs waveform from magnitude and phase

STFT parameters are read once from config.yaml at import time so config is the
single source of truth. Override via the function kwargs only for ad-hoc use.

Usage:
    from src.stft import compute_stft, compute_lps, compute_istft
"""

from pathlib import Path
import numpy as np
import torch
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_CONFIG_PATH, "r") as _f:
    _cfg = yaml.safe_load(_f)

_N_FFT      = int(_cfg["n_fft"])
_HOP        = int(_cfg["hop_length"])
_WIN_LENGTH = int(_cfg["win_length"])
# Created once at import time; reused on every call to avoid per-call allocation.
_WINDOW     = torch.hann_window(_WIN_LENGTH)


def compute_stft(waveform, n_fft=_N_FFT, hop_length=_HOP, win_length=_WIN_LENGTH):
    waveform_tensor = torch.from_numpy(waveform.astype(np.float32))
    window = _WINDOW if win_length == _WIN_LENGTH else torch.hann_window(win_length)
    stft_result = torch.stft(
        waveform_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True
    )
    magnitude = torch.abs(stft_result).numpy()
    phase     = torch.angle(stft_result).numpy()
    return magnitude, phase


def compute_lps(magnitude):
    return np.log(magnitude ** 2 + 1e-8)


def compute_istft(magnitude, phase, n_fft=_N_FFT, hop_length=_HOP, win_length=_WIN_LENGTH):
    complex_stft   = magnitude * np.exp(1j * phase)
    complex_tensor = torch.from_numpy(complex_stft.astype(np.complex64))
    window = _WINDOW if win_length == _WIN_LENGTH else torch.hann_window(win_length)
    istft_result   = torch.istft(
        complex_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )
    return istft_result.numpy()