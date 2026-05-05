"""
src/iq/demodulate.py

Converts QPSK IQ data back to an audio waveform.
This is the reverse of modulate.py.

Pipeline:
    IQ signal → remove carrier → downsample → QPSK symbols → bits → audio

Usage:
    from src.iq.demodulate import demodulate
    waveform = demodulate(i_signal, q_signal, n_bits)
"""

import numpy as np

# these must match the values in modulate.py exactly
# if they don't match, demodulation will produce garbage
SAMPLES_PER_SYMBOL = 8      # how many samples represent one QPSK symbol
CARRIER_FREQ       = 4000   # Hz — the frequency we modulated onto
SAMPLE_RATE        = 16000  # Hz — audio sample rate


def remove_carrier(signal, carrier_freq, sample_rate, is_q=False):
    """
    Multiply the received signal by the same carrier wave used during modulation.
    This shifts the signal back down from the carrier frequency to baseband
    (centered around 0 Hz), which is called 'downconversion' in radio engineering.

    Think of it like tuning a radio — you multiply by the station frequency
    to extract just that station's content.

    I channel uses cosine, Q channel uses sine — must match modulate.py exactly.
    """
    n = np.arange(len(signal))
    t = n / sample_rate     # convert sample index to time in seconds

    if is_q:
        carrier = np.sin(2 * np.pi * carrier_freq * t)
    else:
        carrier = np.cos(2 * np.pi * carrier_freq * t)

    return signal * carrier


def downsample(signal, samples_per_symbol):
    """
    Collapse each group of samples back into one symbol value
    by averaging the group together.

    This is the reverse of upsample() in modulate.py.

    Example with samples_per_symbol=4:
        input : [0.9, 1.1, 1.0, 0.95, -1.1, -0.9, -1.0, -0.95]
        output: [0.9875, -0.9875]   ← one value per symbol

    Averaging helps cancel out noise — if noise pushes one sample
    slightly high, another slightly low, the average stays close
    to the true symbol value.
    """
    # figure out how many complete symbols we have
    n_symbols = len(signal) // samples_per_symbol

    # trim any leftover samples that don't form a complete symbol
    signal = signal[:n_symbols * samples_per_symbol]

    # reshape into a 2D array of (n_symbols, samples_per_symbol)
    # then average across each row to get one value per symbol
    return signal.reshape(n_symbols, samples_per_symbol).mean(axis=1)


def qpsk_symbols_to_bits(i_symbols, q_symbols):
    """
    Convert I and Q symbol values back to bits using a simple sign decision.

    In a perfect system, QPSK symbols are always exactly +1 or -1.
    Noise pushes them slightly off, but they should still be closer
    to the correct value than the wrong one (as long as noise isn't too extreme).

    Decision rule — just look at the sign:
        I > 0 → first bit of pair is 0
        I < 0 → first bit of pair is 1
        Q > 0 → second bit of pair is 0
        Q < 0 → second bit of pair is 1

    This is called 'hard decision decoding' — we commit to one answer
    rather than keeping probabilities.
    """
    # convert sign of I to bit 0 of each pair
    bit0 = (i_symbols < 0).astype(np.uint8)

    # convert sign of Q to bit 1 of each pair
    bit1 = (q_symbols < 0).astype(np.uint8)

    # interleave back into a single flat bit stream
    # bits[0::2] picks every even index  → bit0
    # bits[1::2] picks every odd index   → bit1
    # e.g. bit0=[0,1], bit1=[1,0] → bits=[0,1,1,0]
    bits = np.zeros(len(i_symbols) * 2, dtype=np.uint8)
    bits[0::2] = bit0
    bits[1::2] = bit1

    return bits


def bits_to_audio(bits, n_bits):
    """
    Convert a stream of bits back to a float32 audio waveform.
    Exact reverse of audio_to_bits() in modulate.py.

    Steps:
        1. Trim to the original number of bits
        2. Pack every 8 bits back into one integer (0-255)
        3. Convert each integer back to a float in [-1, 1]

    Example:
        bits [0,0,0,0,0,1,0,1] → integer 5 → float (5/127.5) - 1 = -0.961
    """
    # trim to original bit count so we reconstruct exactly the right length
    bits = bits[:n_bits]

    # make sure we have a multiple of 8 bits (one byte per audio sample)
    bits = bits[:len(bits) // 8 * 8]

    # pack 8 bits back into one byte value (0 to 255)
    samples_int = np.packbits(bits)

    # reverse the scaling from audio_to_bits:
    # there we did: int = (float + 1) * 127.5
    # so reverse:   float = (int / 127.5) - 1
    waveform = (samples_int.astype(np.float32) / 127.5) - 1.0

    return waveform


def demodulate(i_signal, q_signal, n_bits,
               samples_per_symbol=SAMPLES_PER_SYMBOL,
               carrier_freq=CARRIER_FREQ,
               sample_rate=SAMPLE_RATE):
    """
    Full demodulation pipeline: QPSK IQ signal → audio waveform.
    Reverses every step of modulate() in modulate.py.

    Args:
        i_signal           : numpy array, in-phase component (noisy or clean)
        q_signal           : numpy array, quadrature component (noisy or clean)
        n_bits             : int, original number of bits — returned by modulate()
                             needed so we reconstruct exactly the right length
        samples_per_symbol : int, must match value used in modulate()
        carrier_freq       : int, must match value used in modulate()
        sample_rate        : int, must match value used in modulate()

    Returns:
        waveform           : numpy array float32, reconstructed audio
    """
    # step 1 — multiply by carrier to shift signal back to baseband
    i_baseband = remove_carrier(i_signal, carrier_freq, sample_rate, is_q=False)
    q_baseband = remove_carrier(q_signal, carrier_freq, sample_rate, is_q=True)

    # step 2 — average groups of samples back into one value per symbol
    i_symbols = downsample(i_baseband, samples_per_symbol)
    q_symbols = downsample(q_baseband, samples_per_symbol)

    # step 3 — look at sign of each symbol to recover bits
    bits = qpsk_symbols_to_bits(i_symbols, q_symbols)

    # step 4 — pack bits back into integers, then scale to float audio
    waveform = bits_to_audio(bits, n_bits)

    return waveform


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from src.iq.modulate import modulate
    from src.iq.noise import apply_noise
    import yaml

    # use a 440 Hz sine wave (musical A note) as test audio
    sample_rate = 16000
    t           = np.linspace(0, 1, sample_rate, endpoint=False)
    original    = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # ── test WITHOUT noise ────────────────────────────────────────────────────
    # this tells us the baseline error from just the modulation/demodulation
    # process itself (quantization error from converting floats to 8-bit ints)
    i_sig, q_sig, n_bits = modulate(original)
    reconstructed        = demodulate(i_sig, q_sig, n_bits)
    min_len              = min(len(original), len(reconstructed))
    error                = np.mean(np.abs(original[:min_len] - reconstructed[:min_len]))

    print(f"Original length      : {len(original)}")
    print(f"Reconstructed length : {len(reconstructed)}")
    print(f"Mean absolute error  : {error:.6f}")
    print(f"Round trip successful : {error < 0.01}")

    # ── test WITH noise ───────────────────────────────────────────────────────
    # this shows how much damage AWGN + IQ imbalance does to the signal
    # the Stage 1 IQ denoiser will learn to fix exactly this damage
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    i_noisy, q_noisy    = apply_noise(i_sig, q_sig, config)
    reconstructed_noisy = demodulate(i_noisy, q_noisy, n_bits)
    min_len             = min(len(original), len(reconstructed_noisy))
    error_noisy         = np.mean(np.abs(original[:min_len] - reconstructed_noisy[:min_len]))

    print(f"\n--- With noise ---")
    print(f"Mean absolute error  : {error_noisy:.6f}")
    print(f"Round trip successful : {error_noisy < 0.5}")