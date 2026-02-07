"""
Binary Classification CNN Model
Basit CNN modeli ile meme/arka plan sınıflandırması
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Basit CNN modeli - Binary Classification için."""

    def __init__(self, num_classes: int = 2):
        """
        Args:
            num_classes: Sınıf sayısı (default: 2 - binary classification)
        """
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 kanal (grayscale)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, 1, 224, 224)

        Returns:
            Output logits (batch_size, num_classes)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ImprovedCNN(nn.Module):
    """Geliştirilmiş CNN modeli - Daha derin mimari."""

    def __init__(self, num_classes: int = 2):
        """
        Args:
            num_classes: Sınıf sayısı
        """
        super(ImprovedCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def get_model(model_name: str = "simple", num_classes: int = 2, device: str = "cpu") -> nn.Module:
    """
    Model factory fonksiyonu.

    Args:
        model_name: Model adı ("simple" veya "improved")
        num_classes: Sınıf sayısı
        device: Cihaz ("cpu" veya "cuda")

    Returns:
        Model instance
    """
    if model_name == "simple":
        model = SimpleCNN(num_classes=num_classes)
    elif model_name == "improved":
        model = ImprovedCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    return model.to(device)

