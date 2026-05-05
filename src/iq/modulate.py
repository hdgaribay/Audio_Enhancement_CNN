"""
src/iq/modulate.py

Converts audio waveforms to QPSK IQ data.

Pipeline:
    audio waveform → bits → QPSK symbols → IQ signal

Usage:
    from src.iq.modulate import modulate
    i_signal, q_signal = modulate(waveform)
"""

import numpy as np

# QPSK maps every pair of bits to one of 4 points in IQ space
# each point is a combination of I (real) and Q (imaginary)
# visually they sit at the four corners of a square:
#
#   Q
#   |
#   00 (+1+j) . | . (+1-j) 01
#              --|--
#   11 (-1+j) . | . (-1-j) 10
#              ---------> I
#
QPSK_MAP = {
    (0, 0):  (1.0,  1.0),   # top right
    (0, 1):  (1.0, -1.0),   # bottom right
    (1, 0): (-1.0,  1.0),   # top left
    (1, 1): (-1.0, -1.0),   # bottom left
}

SAMPLES_PER_SYMBOL = 8      # how many audio samples per QPSK symbol
CARRIER_FREQ       = 4000   # Hz
SAMPLE_RATE        = 16000  # Hz


def audio_to_bits(waveform):
    """
    Convert a float32 audio waveform to a stream of bits.

    Steps:
        1. Normalize audio to [-1, 1] range
        2. Convert each sample to an 8-bit integer (0-255)
        3. Unpack each integer into 8 individual bits
    """
    # normalize to [-1, 1] just in case
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)

    # convert to 8-bit integers (0 to 255)
    # +1 shifts range from [-1,1] to [0,2]
    # *127.5 scales to [0, 255]
    samples_int = ((waveform + 1) * 127.5).astype(np.uint8)

    # unpack each integer into 8 bits
    # e.g. 5 → [0, 0, 0, 0, 0, 1, 0, 1]
    bits = np.unpackbits(samples_int)
    return bits


def bits_to_qpsk_symbols(bits):
    """
    Group bits into pairs and map each pair to a QPSK symbol.
    Returns two arrays: I (in-phase) and Q (quadrature) components.

    e.g. bits [0,1,1,0,0,0] → pairs [(0,1),(1,0),(0,0)]
                             → I: [1.0, -1.0,  1.0]
                             → Q: [-1.0, 1.0,  1.0]
    """
    # make sure we have an even number of bits
    # if odd, drop the last bit
    bits = bits[:len(bits) // 2 * 2]

    # group into pairs
    pairs = bits.reshape(-1, 2)

    # map each pair to I and Q values
    i_symbols = np.array([QPSK_MAP[tuple(p)][0] for p in pairs])
    q_symbols = np.array([QPSK_MAP[tuple(p)][1] for p in pairs])

    return i_symbols, q_symbols


def upsample(symbols, samples_per_symbol):
    """
    Stretch each symbol out over multiple samples.
    e.g. [1.0, -1.0] with samples_per_symbol=4 → [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]

    Think of it like holding each note for a fixed duration.
    """
    return np.repeat(symbols, samples_per_symbol)


def apply_carrier(signal, carrier_freq, sample_rate, is_q=False):
    """
    Multiply the signal by a carrier wave (a pure sine or cosine).
    This shifts the signal up to the carrier frequency.

    I channel uses cosine, Q channel uses sine — this is what
    makes them separable at the receiver end.
    """
    n = np.arange(len(signal))
    t = n / sample_rate

    if is_q:
        carrier = np.sin(2 * np.pi * carrier_freq * t)
    else:
        carrier = np.cos(2 * np.pi * carrier_freq * t)

    return signal * carrier


def modulate(waveform,
             samples_per_symbol=SAMPLES_PER_SYMBOL,
             carrier_freq=CARRIER_FREQ,
             sample_rate=SAMPLE_RATE):
    """
    Full pipeline: audio waveform → QPSK IQ signal.

    Returns:
        i_signal : numpy array, in-phase component
        q_signal : numpy array, quadrature component
        n_bits   : int, original number of bits (needed for demodulation)
    """
    # step 1 — convert audio to bits
    bits = audio_to_bits(waveform)
    n_bits = len(bits)

    # step 2 — map bits to QPSK symbols
    i_symbols, q_symbols = bits_to_qpsk_symbols(bits)

    # step 3 — upsample: stretch symbols across multiple samples
    i_upsampled = upsample(i_symbols, samples_per_symbol)
    q_upsampled = upsample(q_symbols, samples_per_symbol)

    # step 4 — apply carrier wave
    i_signal = apply_carrier(i_upsampled, carrier_freq, sample_rate, is_q=False)
    q_signal = apply_carrier(q_upsampled, carrier_freq, sample_rate, is_q=True)

    return i_signal, q_signal, n_bits


if __name__ == "__main__":
    # quick test
    test_wav = np.random.randn(16000).astype(np.float32)
    i_sig, q_sig, n_bits = modulate(test_wav)

    print(f"Input samples  : {len(test_wav)}")
    print(f"Bits generated : {n_bits}")
    print(f"I signal length: {len(i_sig)}")
    print(f"Q signal length: {len(q_sig)}")
    print(f"I signal range : [{i_sig.min():.3f}, {i_sig.max():.3f}]")
    print(f"Q signal range : [{q_sig.min():.3f}, {q_sig.max():.3f}]")