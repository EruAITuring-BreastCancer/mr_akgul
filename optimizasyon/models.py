import torch.nn as nn
from torchvision import models
from typing import Optional
import torch
import os

class DenseNetModel(nn.Module):
    def __init__(self, num_classes: int, model_size: str = '121', pretrained: bool = True, dropout_rate: float = 0.3):
        super(DenseNetModel, self).__init__()
        if model_size == '121':
            self.model = models.densenet121(weights=None)
            if pretrained:
                weight_path = '/app/medical_weights/DenseNet121.pt'
                if os.path.exists(weight_path):
                    state_dict = torch.load(weight_path, map_location='cpu')
                    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
                    self.model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Geçersiz boyut.")

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def get_model(model_name: str, num_classes: int, model_size: Optional[str] = None, pretrained: bool = True, dropout_rate: float = 0.3) -> nn.Module:
    model_name = model_name.lower()
    os.environ['TORCH_HOME'] = '/media/agah/Sata/torch_cache'
    
    if model_name == 'densenet':
        size = model_size or '121'
        return DenseNetModel(num_classes, size, pretrained, dropout_rate)
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")
