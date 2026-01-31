"""
Evaluation script for Swin-UNet on a validation split of a retinal OCT dataset.

What this does
- Loads a Swin-UNet model checkpoint
- Builds the validation set as the last 20% of the dataset
- Computes per-class Dice score and mean Dice score over the validation set

Assumptions
- Images are grayscale stored in resized_images.npy (shape: N x H x W)
- Masks are integer labels stored in resized_labeledimages.npy (shape: N x H x W)
- Mask values are class indices in [0, 7] for 8 classes
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from model import SwinUnet
import cv2


class RetinalOCTDataset(torch.utils.data.Dataset):
    """
    Minimal Dataset wrapper for (image, mask) pairs.

    Notes
    - Images are normalized to [0, 1]
    - Images and masks are resized to 224x224
    - Masks use nearest-neighbor resize to preserve discrete class IDs
    """

    def __init__(self, images, masks):
        self.images = images.astype("float32") / 255.0
        self.masks = masks.astype("int64")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.resize(self.images[idx], (224, 224))
        mask = cv2.resize(
            self.masks[idx],
            (224, 224),
            interpolation=cv2.INTER_NEAREST
        )

        img = torch.tensor(img).unsqueeze(0)  # (1, 224, 224)
        mask = torch.tensor(mask)             # (224, 224)
        return img, mask


def dice_score_per_class(logits, targets, num_classes=8, eps=1e-6):
    """
    Computes Dice score per class for a single batch.

    Args:
      logits:  model output logits, shape (B, C, H, W)
      targets: ground-truth labels, shape (B, H, W) with int class IDs
      num_classes: number of segmentation classes
      eps: numerical stability constant

    Returns:
      List[float] of length num_classes with Dice scores.
    """
    scores = []

    # Convert logits -> predicted class IDs per pixel
    preds = torch.argmax(logits, dim=1)  # (B, H, W)

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        dice = (2.0 * intersection) / (union + eps)
        scores.append(dice.item())

    return scores


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model with the same configuration used during training
model = SwinUnet(img_size=(224, 224), in_chans=1, num_classes=8).to(device)

# Load trained weights
model.load_state_dict(torch.load("swin_unet_retinal.pth", map_location=device))
model.eval()

# Load the full dataset
images = np.load("resized_images.npy")
labels = np.load("resized_labeledimages.npy")

# Recreate the same 80/20 split used in training
split_idx = int(0.8 * len(images))
val_images = images[split_idx:]
val_labels = labels[split_idx:]

val_dataset = RetinalOCTDataset(val_images, val_labels)
val_loader = DataLoader(val_dataset, batch_size=1)

# Accumulate Dice scores across the validation set
total_dice = np.zeros(8, dtype=np.float64)

with torch.no_grad():
    for img, mask in val_loader:
        img, mask = img.to(device), mask.to(device)

        # Forward pass
        pred = model(img)

        # Per-class Dice for this sample
        dice_scores = dice_score_per_class(pred, mask, num_classes=8)
        total_dice += np.array(dice_scores, dtype=np.float64)

# Average over all validation samples
avg_dice = total_dice / len(val_loader)

print("\nPer-class Dice Scores:")
for i, score in enumerate(avg_dice):
    print(f"Class {i}: Dice = {score:.4f}")

print(f"\nMean Dice Score: {np.mean(avg_dice):.4f}")
