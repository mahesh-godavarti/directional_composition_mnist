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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

# ---------- Composition Layer ----------
class CompositionLayer(nn.Module):
    def __init__(self, embedding_dim, height=28, width=28):
        super().__init__()
        assert embedding_dim % 2 == 0, "Embedding dimension must be even"
        self.embedding_dim = embedding_dim
        self.H = height
        self.W = width
        self.D2 = embedding_dim // 2

        self.vector = torch.tensor([1.0, 0.0]*self.D2)

        # Initialize this when embedding_dim = 32 or higher
        #self.theta_x = nn.Parameter(torch.randn(self.D2))
        #self.theta_y = nn.Parameter(torch.randn(self.D2))

        # Initialize this way when embedding_dim = 8 or 2
        self.theta_x = nn.Parameter(2*torch.pi*torch.range(1, self.D2) / self.W)
        self.theta_y = nn.Parameter(2*torch.pi*torch.range(1, self.D2) / self.W)

        # i, j grid flattened to H*W
        ii, jj = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        self.register_buffer('i_grid', ii.reshape(-1))  # (H*W,)
        self.register_buffer('j_grid', jj.reshape(-1))  # (H*W,)

    def forward(self, x):
        B, _, H, W = x.shape
        assert H == self.H and W == self.W, "Input spatial size mismatch"
        D2 = self.D2

        x = x.view(B, 1, H * W).permute(0, 2, 1)  # (B, H*W, 1)
        v = x * self.vector
        v = v.view(B, H * W, D2, 2)

        # Compute θ_total = θ_x * i + θ_y * j
        theta_total = (
            self.i_grid[:, None] * self.theta_x[None, :] + 
            self.j_grid[:, None] * self.theta_y[None, :]
        )  # shape: (H*W, D/2)

        cos_theta = torch.cos(theta_total)  # (H*W, D/2)
        sin_theta = torch.sin(theta_total)  # (H*W, D/2)

        # Rotation: [cos, -sin; sin, cos] applied to each 2D subspace
        x_comp = v[..., 0]  # (B, H*W, D/2)
        y_comp = v[..., 1]  # (B, H*W, D/2)

        x_rot = x_comp * cos_theta.unsqueeze(0) + y_comp * sin_theta.unsqueeze(0)
        y_rot = -x_comp * sin_theta.unsqueeze(0) + y_comp * cos_theta.unsqueeze(0)

        v_rot = torch.stack([x_rot, y_rot], dim=-1)  # (B, H*W, D/2, 2)
        v_out = v_rot.sum(dim=1).reshape(B, self.embedding_dim)  # (B, D)

        return v_out

# ---------- Classifier ----------
class CompositionClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.comp = CompositionLayer(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        v = self.comp(x)  # (B, D)
        x = F.relu(self.fc1(v))
        return self.fc2(x)

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
    embedding_dim = 8  # must be even
    batch_size = 64
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    model = CompositionClassifier(embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, criterion, device, epoch)
        test(model, test_loader, device)

if __name__ == "__main__":
    main()

