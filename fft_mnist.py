# -----------------------------------------------------------------------------
# Copyright (c) 2025 Mahesh Godavarti. All Rights Reserved.
#
# License: This software is provided for non-commercial research purposes only.
# Any commercial use, including but not limited to use in a product, service,
# or for-profit research, is strictly prohibited without explicit written
# permission from the copyright holder.
#
# Patent Pending: Certain aspects of this software are the subject of a
# pending patent application.
#
# Contact: m@qalaxia.com
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

down_sample = 7

# Custom transform to apply 2D FFT and take magnitude
class FFTMagnitudeTransform:
    def __call__(self, x):
        # Input is a PIL image; convert to tensor and normalize
        x = transforms.ToTensor()(x)
        # Apply 2D FFT and take magnitude
        fft = torch.fft.fft2(x)
        magnitude = torch.abs(fft)
        return magnitude

# Custom transform to apply 2D FFT and return real and imaginary parts stacked
class FFTComplexTransform:
    def __call__(self, x):
        # Convert image to tensor
        x = transforms.ToTensor()(x)  # Shape: (1, H, W) if grayscale
        if x.ndim == 2:
            x = x.unsqueeze(0)  # Ensure (1, H, W)

        # Get dimensions
        C, H, W = x.shape
        #assert H % down_sample == 0 and W % down_sample == 0, "H and W must be divisible by downsampling down_sample"

        # FFT on each channel
        fft = torch.fft.fft2(x)  # Complex tensor (C, H, W)
        fft = torch.fft.fftshift(fft, dim=(-2, -1))

        # Crop center frequencies
        h_crop = H // down_sample
        w_crop = W // down_sample
        h_start = (H - h_crop) // 2
        w_start = (W - w_crop) // 2
        fft_cropped = fft[:, h_start:h_start + h_crop, w_start:w_start + w_crop]

        # Extract real and imaginary parts
        real = fft_cropped.real
        imag = fft_cropped.imag

        # Return concatenated tensor: (2*C, H', W')
        return torch.cat([real, imag], dim=0)

# Simple classifier model
class FFTClassifier(nn.Module):
    def __init__(self):
        super(FFTClassifier, self).__init__()
        self.fc1 = nn.Linear(2*int(28/down_sample) * int(28/down_sample), 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 2 * int(28/down_sample) * int(28/down_sample))  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------- Training / Testing ----------
def train(model, loader, optimizer, criterion, device, epoch):
    model.train()
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

def test(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += pred.eq(y).sum().item()
    print(f"Test Accuracy: {100. * correct / len(loader.dataset):.2f}%")

# ---------- Main ----------
def main():
    batch_size = 64
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = FFTComplexTransform()
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    model = FFTClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, criterion, device, epoch)
        test(model, test_loader, device)

if __name__ == "__main__":
    main()

