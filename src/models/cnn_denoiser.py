"""
cnn_denoiser.py

A simple fully-convolutional 2D CNN for speech enhancement.

Input:  noisy log power spectrogram  (batch, 1, 256, 302)
Output: predicted clean magnitude    (batch, 1, 256, 302)

The model treats the spectrogram like a grayscale image and learns
to suppress noise patterns using 2D convolutions.

Usage:
    from src.models.cnn_denoiser import CNNDenoiser
    model = CNNDenoiser()
    output = model(noisy_lps)   # (B, 1, 256, 302) → (B, 1, 256, 302)
"""
import torch
import torch.nn as nn
class CNNDenoiser(nn.Module):
    """
    5-layer fully convolutional denoiser.

    Every layer uses:
        - kernel_size=5  (looks at a 5x5 patch of the spectrogram)
        - padding=2      (keeps spatial size exactly the same)
        - BatchNorm      (stabilizes training)
        - ReLU           (adds non-linearity, except the last layer)
    """
    def __init__(self):
        super(CNNDenoiser,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5,padding = 2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5,padding = 2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5,padding = 2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.output_layer = nn.Conv2d(
            in_channels = 16, out_channels = 1, kernel_size = 5, padding = 2
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.output_layer(x)
        return x
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
    return total
    
if __name__ == "__main__":
    model = CNNDenoiser()
    count_parameters(model)

    x = torch.randn(2, 1, 256, 302)
    y = model(x)
    print(y.shape)

    

    
