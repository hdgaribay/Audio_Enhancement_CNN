"""
src/iq/noise.py

Adds noise to IQ signals.

Currently supported:
    - AWGN (Additive White Gaussian Noise)
    - IQ imbalance (amplitude and phase mismatch between I and Q channels)

Designed to be easily extended — add a new function and call it in apply_noise().

Usage:
    from src.iq.noise import apply_noise
    i_noisy, q_noisy = apply_noise(i_signal, q_signal, config)
"""

import numpy as np


def add_awgn(i_signal, q_signal, snr_db):
    """
    Add Additive White Gaussian Noise at a target SNR.

    AWGN is completely random noise that affects both I and Q equally.
    It simulates thermal noise in real radio hardware — it's always
    present to some degree in any real transmission.

    Higher SNR = less noise = cleaner signal.
    """
    # measure signal power across both channels
    signal_power = np.mean(i_signal ** 2 + q_signal ** 2)

    # calculate how much noise power we need to hit target SNR
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std   = np.sqrt(noise_power / 2)

    # generate independent random noise for I and Q
    # gaussian noise = np.random.randn, which gives values
    # centered around 0 with a standard deviation of 1
    i_noise = np.random.randn(len(i_signal)) * noise_std
    q_noise = np.random.randn(len(q_signal)) * noise_std

    return i_signal + i_noise, q_signal + q_noise


def add_iq_imbalance(i_signal, q_signal, amplitude_imbalance, phase_imbalance_deg):
    """
    Simulate IQ imbalance — a real hardware imperfection where the I and Q
    channels are not perfectly matched.

    Two types:
    - Amplitude imbalance: Q channel is slightly louder or quieter than I
    - Phase imbalance: the 90 degree angle between I and Q is slightly off

    In a perfect radio, I and Q are exactly equal amplitude and exactly
    90 degrees apart. Real hardware is never perfect.

    Args:
        amplitude_imbalance  : multiplier for Q channel (e.g. 1.05 = 5% louder)
        phase_imbalance_deg  : degrees of phase error (e.g. 2.0 = 2 degrees off)
    """
    # convert phase error from degrees to radians
    # radians are what numpy's trig functions expect
    phase_rad = np.deg2rad(phase_imbalance_deg)

    # apply amplitude imbalance to Q channel
    q_signal = q_signal * amplitude_imbalance

    # apply phase imbalance — rotate Q channel slightly
    # this mixes a tiny bit of I into Q and vice versa
    i_out = i_signal - q_signal * np.sin(phase_rad)
    q_out = q_signal * np.cos(phase_rad)

    return i_out, q_out


def apply_noise(i_signal, q_signal, config):
    """
    Apply all noise types to an IQ signal.

    This is the main function called by the rest of the pipeline.
    Adding new noise types in the future = add a function above
    and one line here.

    Args:
        i_signal : numpy array, in-phase component
        q_signal : numpy array, quadrature component
        config   : dict loaded from config.yaml
    """
    # pick a random SNR within the configured range
    snr_db = np.random.uniform(
        config["awgn_snr_min"],
        config["awgn_snr_max"]
    )
    i_signal, q_signal = add_awgn(i_signal, q_signal, snr_db)

    # pick random IQ imbalance values within configured ranges
    amplitude = np.random.uniform(
        config["iq_imbalance"]["amplitude_range"][0],
        config["iq_imbalance"]["amplitude_range"][1]
    )
    phase = np.random.uniform(
        config["iq_imbalance"]["phase_range"][0],
        config["iq_imbalance"]["phase_range"][1]
    )
    i_signal, q_signal = add_iq_imbalance(i_signal, q_signal, amplitude, phase)

    return i_signal, q_signal


if __name__ == "__main__":
    import yaml

    # quick test using a clean sine wave as a stand-in for real IQ data
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    t        = np.linspace(0, 1, 16000)
    i_clean  = np.cos(2 * np.pi * 4000 * t).astype(np.float32)
    q_clean  = np.sin(2 * np.pi * 4000 * t).astype(np.float32)

    i_noisy, q_noisy = apply_noise(i_clean, q_clean, config)

    print(f"I clean range  : [{i_clean.min():.3f},  {i_clean.max():.3f}]")
    print(f"I noisy range  : [{i_noisy.min():.3f}, {i_noisy.max():.3f}]")
    print(f"Q clean range  : [{q_clean.min():.3f},  {q_clean.max():.3f}]")
    print(f"Q noisy range  : [{q_noisy.min():.3f}, {q_noisy.max():.3f}]")
    print("Noise applied successfully!")