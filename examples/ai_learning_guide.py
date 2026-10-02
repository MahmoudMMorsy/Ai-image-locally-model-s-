"""
AI Model Training & Learning Demo Guide Script
Demonstrates building and running a lightweight Neural Generator model using PyTorch.
"""
import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class LightweightAIGenerator(nn.Module):
    def __init__(self, input_dim=16, output_channels=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 64 * output_channels),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.size(0)
        out = self.fc(z)
        return out.view(batch_size, 4, 64, 64)

def main():
    print("=== AI Model Learning & Development Demo ===")
    os.makedirs("examples/ai_learning_demo", exist_ok=True)

    # 1. Initialize Neural Network Model
    model = LightweightAIGenerator(input_dim=16, output_channels=4)
    model.eval()
    print("1. Model Architecture Initialized successfully.")

    # 2. Vector Latent Input (Concept Representation)
    latent_vector = torch.randn(1, 16)

    # 3. Model Inference (Image Synthesis)
    with torch.no_grad():
        output_tensor = model(latent_vector)

    arr = (output_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGBA")

    output_path = "examples/ai_learning_demo/neural_output_sample.png"
    img.save(output_path)
    print(f"2. Generated Neural Artifact saved to: {output_path}")
    print("=== Demo Completed Successfully ===")

if __name__ == "__main__":
    main()
