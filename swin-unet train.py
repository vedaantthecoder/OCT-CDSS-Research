# Swin-UNet Fine-Tuning on Retinal OCT Dataset
# This script fine-tunes a Swin-UNet segmentation model on grayscale
# retinal OCT images using multi-class Dice loss.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


class RetinalOCTDataset(Dataset):
    """
    PyTorch Dataset for retinal OCT image segmentation.

    Images are assumed to be grayscale and stored as NumPy arrays.
    Masks contain integer class labels for each pixel.
    """

    def __init__(self, images, masks):
        # Normalize image intensities to [0, 1]
        self.images = images.astype("float32") / 255.0

        # Keep masks as integer class indices
        self.masks = masks.astype("int64")

    def __len__(self):
        # Total number of samples
        return len(self.images)

    def __getitem__(self, idx):
        # Resize image to match Swin-UNet input resolution
        img = cv2.resize(self.images[idx], (224, 224))

        # Resize mask using nearest-neighbor interpolation to
        # preserve discrete class labels
        mask = cv2.resize(
            self.masks[idx],
            (224, 224),
            interpolation=cv2.INTER_NEAREST
        )

        # Convert image to tensor and add channel dimension
        img = torch.tensor(img).unsqueeze(0)  # (1, 224, 224)

        # Convert mask to tensor (H, W)
        mask = torch.tensor(mask)

        return img, mask


def dice_loss(pred, target, smooth=1):
    """
    Multi-class Dice loss for semantic segmentation.

    pred:   Raw logits of shape (B, C, H, W)
    target: Integer mask of shape (B, H, W)
    """

    # Convert logits to class probabilities
    pred = torch.softmax(pred, dim=1)

    # One-hot encode ground-truth masks
    target = F.one_hot(
        target.long(),
        num_classes=8
    ).permute(0, 3, 1, 2).float()

    # Compute Dice score per class
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Return Dice loss
    return 1 - dice.mean()


# Swin-UNet model definition
from model import SwinUnet


# Load preprocessed OCT images and segmentation masks
images = np.load("resized_images.npy")
labels = np.load("resized_labeledimages.npy")


# Split dataset into training and validation sets (80/20)
split = int(0.8 * len(images))
train_dataset = RetinalOCTDataset(images[:split], labels[:split])
val_dataset = RetinalOCTDataset(images[split:], labels[split:])

# DataLoaders handle batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)


# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Swin-UNet for 1-channel OCT images and 8 output classes
model = SwinUnet(
    img_size=(224, 224),
    in_chans=1,
    num_classes=8
).to(device)

# Adam optimizer for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Training and validation loop
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for img, mask in tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{num_epochs}"
    ):
        img, mask = img.to(device), mask.to(device)

        # Forward pass
        pred = model(img)

        # Compute Dice loss
        loss = dice_loss(pred, mask)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(
        f"Epoch {epoch + 1}: "
        f"Train Loss = {train_loss / len(train_loader):.4f}"
    )

    # Validation phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = dice_loss(pred, mask)
            val_loss += loss.item()

    print(
        f"Epoch {epoch + 1}: "
        f"Val Loss = {val_loss / len(val_loader):.4f}"
    )


# Save trained model weights
torch.save(model.state_dict(), "swin_unet_retinal.pth")
