"""Свёрточная Q-сеть с раздельными головами ценности и преимущества (dueling).

Размеры подобраны под кадр 40×150: после свёрток остаётся карта 3×21,
на которой ещё различимы высота птицы и расстояние до кактуса.
Та же сеть пересобрана на TensorFlow.js в web/viewer.js — меняйте вместе.
"""
import torch
from torch import nn


class QNet(nn.Module):
    def __init__(self, n_actions, stack=4, height=40, width=150):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(stack, 32, 6, stride=3), nn.ReLU(),  # 12×49
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),     # 5×23
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),     # 3×21
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, stack, height, width)).shape[1]
        self.value = nn.Sequential(nn.Linear(n_flat, 256), nn.ReLU(), nn.Linear(256, 1))
        self.advantage = nn.Sequential(nn.Linear(n_flat, 256), nn.ReLU(), nn.Linear(256, n_actions))

    def forward(self, x):
        """x — uint8 кадры [B, stack, H, W]."""
        f = self.conv(x.float() / 255.0)
        a = self.advantage(f)
        return self.value(f) + a - a.mean(dim=1, keepdim=True)
