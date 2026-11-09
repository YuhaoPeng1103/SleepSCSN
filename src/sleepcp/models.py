"""Neural network backbones for sleep staging."""

from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    """A compact EEGNet-style 1D CNN for single-channel sleep EEG.

    The input is ``(batch, 1, n_samples)``. The architecture follows the
    depthwise-separable design of EEGNet with adaptive pooling so that it can
    accept recordings of different lengths.
    """

    def __init__(
        self,
        n_classes: int = 5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.25,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False)
        self.bn1 = _norm2d(norm, F1)
        self.conv2 = nn.Conv2d(
            F1, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1, bias=False
        )
        self.bn2 = _norm2d(norm, F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(
            F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )
        self.conv4 = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = _norm2d(norm, F2)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(F2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv2d wants (B, C, H, W). The loader supplies (B, T); a 3D input is
        # read as (B, T) with an explicit channel axis.
        if x.ndim == 2:
            x = x[:, None, None, :]
        elif x.ndim == 3:
            x = x[:, None, :, :]
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = torch.nn.functional.elu(x)
        x = self.drop1(self.pool1(x))
        x = self.conv4(self.conv3(x))
        x = self.bn3(x)
        x = torch.nn.functional.elu(x)
        x = self.drop2(self.pool2(x))
        x = torch.flatten(x, 1)
        return self.fc(x)


class SleepCNN(nn.Module):
    """A small VGG-style 1D CNN, used as a second backbone.

    Deliberately architecturally different from :class:`EEGNet` (more
    channels, larger receptive field, global average pooling) so that the
    calibration results are not an artefact of one architecture.
    """

    def __init__(
        self, n_classes: int = 5, dropout: float = 0.3, norm: str = "batch"
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 64, 7, norm),
            nn.MaxPool1d(2),
            self._block(64, 128, 5, norm),
            nn.MaxPool1d(2),
            self._block(128, 128, 5, norm),
            nn.MaxPool1d(2),
            self._block(128, 256, 3, norm),
            nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    @staticmethod
    def _block(cin: int, cout: int, kernel: int, norm: str = "batch") -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=kernel, padding=kernel // 2, bias=False),
            _norm1d(norm, cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.head(self.features(x))


def _norm2d(kind: str, channels: int) -> nn.Module:
    """Normalisation layer for a 2-D convolutional block.

    ``batch`` is the original architecture. ``instance`` normalises each
    example separately, so the statistics of a batch cannot carry the
    identity of the subjects in it.
    """
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "group":
        return nn.GroupNorm(min(8, channels), channels)
    raise ValueError(f"unknown norm: {kind}")


def _norm1d(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(channels)
    if kind == "instance":
        return nn.InstanceNorm1d(channels, affine=True)
    if kind == "group":
        return nn.GroupNorm(min(8, channels), channels)
    raise ValueError(f"unknown norm: {kind}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, n_classes: int = 5, norm: str = "batch") -> nn.Module:
    name = name.lower()
    if name == "eegnet":
        return EEGNet(n_classes=n_classes, norm=norm)
    if name in {"sleepcnn", "cnn"}:
        return SleepCNN(n_classes=n_classes, norm=norm)
    raise ValueError(f"unknown model: {name}")
