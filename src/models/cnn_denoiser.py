"""
cnn_denoiser.py

U-Net-shaped fully-convolutional 2D CNN for speech enhancement.

Input:  noisy log power spectrogram  (batch, 1, 256, 308)
Output: predicted clean log power spectrogram  (batch, 1, 256, 308)

Architecture:
    Encoder: 1→16→32→64 channels
    Decoder: 64→32→16→1 channels
    Skip connections from each encoder layer to its symmetric decoder layer
    preserve fine-grained spectral detail that the bottleneck would otherwise discard.

At inference, convert output to magnitude via: mag = exp(lps / 2)

Usage:
    from src.models.cnn_denoiser import CNNDenoiser
    model = CNNDenoiser()
    lps_out = model(noisy_lps)   # (B, 1, 256, 308) → (B, 1, 256, 308)
"""
import torch
import torch.nn as nn


class CNNDenoiser(nn.Module):
    def __init__(self):
        super(CNNDenoiser, self).__init__()
        # encoder
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # decoder — channel counts match encoder for skip addition
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.output_layer = nn.Conv2d(16, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3) + x2  # skip from encoder layer2
        x5 = self.layer5(x4) + x1  # skip from encoder layer1
        return self.output_layer(x5)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
    return total


if __name__ == "__main__":
    model = CNNDenoiser()
    count_parameters(model)

    x = torch.randn(2, 1, 256, 308)
    y = model(x)
    print(y.shape)
