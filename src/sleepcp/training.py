"""Training and inference helpers."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SleepDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.y.numel()

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Seeding alone does not pin cuDNN: with algorithm autotuning enabled, the
    # convolution kernels chosen (and some of their reductions) vary between
    # processes, so the same seed can give slightly different weights. That is
    # immaterial for the reported conclusions but not for the claim that the
    # released code reproduces the numbers, so pin the kernels as well.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        SleepDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = criterion(model(x), y)
            total += float(loss.item()) * y.numel()
            count += y.numel()
    return total / max(count, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    class_weights: np.ndarray | None = None,
) -> tuple[nn.Module, list[dict]]:
    model = model.to(device)
    if class_weights is not None:
        weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * y.numel()
            count += y.numel()
        scheduler.step()
        train_loss = total / max(count, 1)
        val_loss = (
            evaluate_loss(model, val_loader, device, criterion)
            if val_loader is not None
            else train_loss
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def class_weights(y: np.ndarray, n_classes: int = 5) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    return weights.astype(np.float32)
