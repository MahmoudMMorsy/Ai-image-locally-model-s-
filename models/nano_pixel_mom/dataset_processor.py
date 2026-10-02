import os
import json
import zipfile
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class PixelMomDataset(Dataset):
    def __init__(self, dataset_dir="dataset_gdrive_extracted/images_2534_only", image_size=(64, 64)):
        self.dataset_dir = dataset_dir
        self.sprites_dir = os.path.join(dataset_dir, "sprites")
        self.manifest_path = os.path.join(dataset_dir, "manifest.jsonl")
        self.image_size = image_size

        # Auto-extract dataset_gdrive.zip if extracted directory does not exist or is empty
        if not os.path.exists(self.sprites_dir) and os.path.exists("dataset_gdrive.zip"):
            try:
                os.makedirs("dataset_gdrive_extracted", exist_ok=True)
                with zipfile.ZipFile("dataset_gdrive.zip", "r") as zip_ref:
                    zip_ref.extractall("dataset_gdrive_extracted")
            except Exception as e:
                print(f"Error extracting dataset_gdrive.zip: {e}")

        self.samples = []
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        file_path = os.path.join(self.sprites_dir, item["file"])
                        if os.path.exists(file_path):
                            self.samples.append({
                                "file_path": file_path,
                                "caption": item.get("caption", "pixel art character")
                            })

        # Fallback to direct directory listing if manifest isn't present
        if not self.samples and os.path.exists(self.sprites_dir):
            for fname in os.listdir(self.sprites_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append({
                        "file_path": os.path.join(self.sprites_dir, fname),
                        "caption": "pixel art character"
                    })

        # Secondary fallback: check other image directories if still empty
        if not self.samples:
            fallback_dirs = ["dataset_training_images/raw", "dataset_training_images/clean", "examples"]
            for fdir in fallback_dirs:
                if os.path.exists(fdir):
                    for root, _, files in os.walk(fdir):
                        for file in files:
                            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                                self.samples.append({
                                    "file_path": os.path.join(root, file),
                                    "caption": "pixel art character"
                                })

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(), # scale [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)) # [-1, 1]
        ])

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        if self.samples and idx < len(self.samples):
            sample = self.samples[idx]
            img = Image.open(sample["file_path"]).convert("RGBA")
        else:
            # Synthetic fallback tensor
            img = Image.new("RGBA", self.image_size, (128, 128, 128, 255))

        tensor = self.transform(img)
        cond = torch.randn(64)
        return tensor, cond
