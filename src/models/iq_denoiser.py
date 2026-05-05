"""
src/models/iq_denoiser.py

1D CNN for cleaning noisy IQ signals.

The model takes noisy I and Q channels as input and predicts
the clean versions. It mirrors the architecture of CNNDenoiser
but works on 1D signals instead of 2D spectrograms.

Input:  noisy IQ signal  (batch, 2, signal_length)
            channel 0 = I (in-phase)
            channel 1 = Q (quadrature)
Output: predicted clean IQ signal  (batch, 2, signal_length)

Architecture:
    Encoder: 2 → 16 → 32 → 64   (1D convolutions)
    Decoder: 64 → 32 → 16 → 2   (1D convolutions)
    Skip connections: encoder layer1 → decoder layer2
                      encoder layer2 → decoder layer1

Usage:
    from src.models.iq_denoiser import IQDenoiser
    model = IQDenoiser()
    clean_iq = model(noisy_iq)   # (B, 2, L) → (B, 2, L)
"""

import torch
import torch.nn as nn


def conv_block(in_channels, out_channels, kernel_size=9):
    """
    One convolutional block: Conv1d → BatchNorm → ReLU.

    Using kernel_size=9 (vs 5x5 in CNNDenoiser) because:
    - IQ signals are 1D so we need a wider kernel to capture
      patterns that span more samples
    - 9 samples at 16kHz covers about 0.5ms which is a
      meaningful chunk of a radio symbol

    padding = kernel_size // 2 keeps output the same length as input.
    e.g. kernel_size=9 → padding=4
    """
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels,
                  kernel_size=kernel_size,
                  padding=kernel_size // 2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )


class IQDenoiser(nn.Module):
    def __init__(self):
        super(IQDenoiser, self).__init__()

        # ── encoder — learns to extract features from noisy IQ ───────────────
        # each layer looks for increasingly complex patterns
        # layer1: basic amplitude/phase patterns
        # layer2: symbol-level patterns
        # layer3: sequence-level patterns
        self.encoder1 = conv_block(2,  16)   # 2 channels in  (I and Q)
        self.encoder2 = conv_block(16, 32)
        self.encoder3 = conv_block(32, 64)

        # ── decoder — reconstructs clean IQ from features ────────────────────
        # skip connections add back fine detail that gets lost in the encoder
        # decoder1 input is 64 channels from encoder3
        # decoder2 input is 32+32=64 channels (decoder1 output + encoder2 skip)
        # decoder3 input is 16+16=32 channels (decoder2 output + encoder1 skip)
        self.decoder1 = conv_block(64,    32)
        self.decoder2 = conv_block(32+32, 16)   # +32 from encoder2 skip
        self.decoder3 = conv_block(16+16,  2)   # +16 from encoder1 skip

        # final layer maps back to 2 output channels (clean I and Q)
        # no BatchNorm or ReLU here — we want the raw output values
        self.output_layer = nn.Conv1d(2, 2, kernel_size=9, padding=4)

    def forward(self, x):
        """
        x shape: (batch, 2, signal_length)
            batch         = number of samples processed at once
            2             = I and Q channels
            signal_length = number of IQ samples
        """
        # encode — extract features at each level
        e1 = self.encoder1(x)    # (B, 16, L)
        e2 = self.encoder2(e1)   # (B, 32, L)
        e3 = self.encoder3(e2)   # (B, 64, L)

        # decode — reconstruct clean signal
        # torch.cat joins tensors along the channel dimension
        # this is how skip connections work — we glue the encoder
        # output onto the decoder input so detail isn't lost
        d1 = self.decoder1(e3)                        # (B, 32, L)
        d2 = self.decoder2(torch.cat([d1, e2], dim=1))  # (B, 16, L)
        d3 = self.decoder3(torch.cat([d2, e1], dim=1))  # (B, 2,  L)

        return self.output_layer(d3)                  # (B, 2,  L)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
    return total


if __name__ == "__main__":
    model = IQDenoiser()
    count_parameters(model)

    # test with a random batch
    # batch=2, channels=2 (I and Q), length=512000
    # 512000 = 16000 audio samples * 8 samples per symbol * 4 (bits per sample / 2)
    x = torch.randn(2, 2, 512000)
    y = model(x)

    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {y.shape}")
    print(f"Shapes match : {x.shape == y.shape}")