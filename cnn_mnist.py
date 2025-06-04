import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=1000, shuffle=False
)

# CNN feature extractor that outputs a 32D vector
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 28x28 -> 14x14

            nn.Conv2d(16, 32, 3, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 14x14 -> 7x7
        )
        self.fc = nn.Linear(32 * 7 * 7, 32)  # Output 32D vector

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Classifier that takes 32D vector and outputs 10 digits
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.fc(x)

# Combine models
feature_extractor = CNNFeatureExtractor().to(device)
classifier = Classifier().to(device)

# Combine parameters for optimization
params = list(feature_extractor.parameters()) + list(classifier.parameters())
optimizer = optim.Adam(params)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):
    feature_extractor.train()
    classifier.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        features = feature_extractor(data)
        output = classifier(features)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete")

    # Evaluation
    feature_extractor.eval()
    classifier.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            features = feature_extractor(data)
            output = classifier(features)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    print(f"Test Accuracy: {correct / len(test_loader.dataset):.4f}")

