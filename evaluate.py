import os
import time
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


# --------------------------------------------------
# Dataset loader for image folders
# --------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self, root, image_size=64):
        self.root = Path(root)
        self.paths = list(self.root.glob("*.png")) + list(self.root.glob("*.jpg"))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# --------------------------------------------------
# Load images from folder
# --------------------------------------------------

def load_images(folder, image_size=64, batch_size=32):
    dataset = ImageFolderDataset(folder, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


# --------------------------------------------------
# FID + Inception Score
# --------------------------------------------------

def evaluate_metrics(real_dir, fake_dir, device="cuda"):

    real_loader = load_images(real_dir)
    fake_loader = load_images(fake_dir)

    fid = FrechetInceptionDistance(
        feature=2048
    ).to(device)

    inception = InceptionScore().to(device)

    # --------------------------
    # REAL IMAGES
    # --------------------------
    for imgs in real_loader:
        imgs = imgs.to(device)
        fid.update(imgs, real=True)

    # --------------------------
    # FAKE IMAGES
    # --------------------------
    for imgs in fake_loader:
        imgs = imgs.to(device)
        fid.update(imgs, real=False)
        inception.update(imgs)

    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()

    return fid_score, is_mean.item(), is_std.item()


# --------------------------------------------------
# Inference time benchmark
# --------------------------------------------------

def measure_inference_time(generate_fn, num_images=100):

    start = time.perf_counter()

    _ = generate_fn(num_images=num_images)

    end = time.perf_counter()

    total = end - start

    return {
        "total_time": total,
        "avg_per_image": total / num_images
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    real_dir = "datasets/flowers"
    fake_dir = "outputs/samples/20260613_151543"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Evaluating...")

    fid, is_mean, is_std = evaluate_metrics(
        real_dir,
        fake_dir,
        device=device
    )

    print("\n--- Results ---")
    print(f"FID: {fid:.2f}")
    print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")


if __name__ == "__main__":
    main()