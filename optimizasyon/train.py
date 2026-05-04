import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns

class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_classes: int, model_name: str = 'model', class_weights: Optional[torch.Tensor] = None, device: str = 'cuda', learning_rate: float = 1e-3, weight_decay: float = 1e-4, output_dir: str = 'outputs', num_epochs: int = 30, beta1: float = 0.9, beta2: float = 0.999, l1_lambda: float = 0.0):
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.l1_lambda = l1_lambda

        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2)

        self.early_stop_patience = 5
        self.epochs_no_improve = 0
        self.early_stop_triggered = False

        self.history = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'learning_rates': []}
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.start_epoch = 0

    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            if self.l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in self.model.parameters() if p.requires_grad)
                loss = loss + self.l1_lambda * l1_norm

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        epoch_f1 = f1_score(all_targets, all_preds, average='macro') * 100
        return epoch_loss, epoch_acc, epoch_f1

    def validate(self, epoch: int) -> Tuple[float, float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        epoch_f1 = f1_score(all_targets, all_preds, average='macro') * 100
        return epoch_loss, epoch_acc, epoch_f1

    def train(self, num_epochs: int, save_best: bool = True) -> Dict:
        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            if self.early_stop_triggered:
                break
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            val_loss, val_acc, val_f1 = self.validate(epoch)
            self.scheduler.step(val_f1)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(current_lr)

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                if save_best:
                    self.save_checkpoint(epoch, filename=f'{self.model_name}_best_model.pth', is_best=True)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.early_stop_patience:
                    self.early_stop_triggered = True

        self.save_checkpoint(self.start_epoch + num_epochs - 1, filename=f'{self.model_name}_last_model.pth')
        self.start_epoch += num_epochs
        return self.history

    def save_checkpoint(self, epoch: int, filename: str = 'checkpoint.pth', is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'history': self.history
        }
        filepath = self.output_dir / filename
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        self.history = checkpoint['history']
        self.start_epoch = checkpoint['epoch'] + 1
