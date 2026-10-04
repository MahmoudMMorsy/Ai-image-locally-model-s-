#!/usr/bin/env python3
"""SkipPredictor architecture used in training (CPU). Checkpoint: skip_predictor_slim.pt"""
from __future__ import annotations
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class SP(nn.Module):
    def __init__(self, latent=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent, 384), nn.ReLU(True), nn.Linear(384, 192 * 8 * 8))
        self.to_s3 = nn.Sequential(nn.ConvTranspose2d(192, 192, 4, 2, 1), nn.ReLU(True), nn.Conv2d(192, 192, 3, 1, 1), nn.ReLU(True))
        self.to_s2 = nn.Sequential(nn.ConvTranspose2d(192, 96, 4, 2, 1), nn.ReLU(True), nn.Conv2d(96, 96, 3, 1, 1), nn.ReLU(True))
        self.to_s1 = nn.Sequential(nn.ConvTranspose2d(96, 48, 4, 2, 1), nn.ReLU(True), nn.Conv2d(48, 48, 3, 1, 1), nn.ReLU(True))

    def forward(self, z):
        h = self.fc(z).view(-1, 192, 8, 8)
        x3 = self.to_s3(h)
        x2 = self.to_s2(x3)
        x1 = self.to_s1(x2)
        return {"x1": x1, "x2": x2, "x3": x3}

def focal_ce(logits, target, weight=None, gamma=1.5):
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
    log_pt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    loss = -((1 - pt) ** gamma) * log_pt
    if weight is not None:
        loss = loss * weight[target]
    return loss.mean()

def main():
    print("SP architecture for arcade_palette. Train with full bank+UNet env.")
    print("Latest: see STATUS.md · checkpoint skip_predictor_slim.pt")

if __name__ == "__main__":
    main()
