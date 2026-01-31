"""
Speckle-Aware ViT with Dynamic Speckle Weighting

What changed in this version:
1) FastSpeckleMap now computes a speckle strength score (mean absolute residual) per image.
2) The speckle residual channel is scaled by a normalized weight derived from that score.
3) Added min_val and max_val to control how speckle strength maps to a [0, 1] weight.
4) Kept the 2-channel ViT patch projection (conv_proj) approach, rather than the CNN frontend approach.

Notes:
- If min_val/max_val are poorly chosen, weight may saturate at 0 or 1 for most images.
  Consider printing or sampling speckle_amount statistics to tune these parameters.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision.models.vision_transformer import vit_b_16


class FastSpeckleMap:
    """
    Create a 2-channel tensor from a grayscale OCT image:
      Channel 0: intensity
      Channel 1: dynamically weighted speckle residual

    The speckle residual is computed as:
      speckle = img_tensor - GaussianBlur(img_tensor)

    A scalar speckle_amount is computed per image:
      speckle_amount = mean(abs(speckle))

    Then a per-image weight in [0, 1] is computed by linear mapping:
      weight = clamp((speckle_amount - min_val) / (max_val - min_val), 0, 1)

    Channel 1 becomes:
      weighted_speckle = speckle * weight
    """
    def __init__(self, min_val=0.0, max_val=0.2):
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def __call__(self, img):
        # Convert to grayscale to ensure consistent intensity channel.
        img = img.convert("L")

        # Convert to tensor in [0, 1], shape: (1, H, W)
        img_tensor = transforms.ToTensor()(img)

        # Low-pass component for residual computation.
        blurred = transforms.GaussianBlur(5)(img_tensor)

        # Speckle residual (high-frequency component).
        speckle = img_tensor - blurred

        # Speckle strength proxy (scalar).
        speckle_amount = speckle.abs().mean()

        # Normalize into [0, 1] using min_val/max_val bounds.
        denom = (self.max_val - self.min_val)
        if denom <= 1e-8:
            # Guard against division by zero; fall back to no scaling.
            weight = torch.tensor(1.0, device=speckle.device, dtype=speckle.dtype)
        else:
            weight = ((speckle_amount - self.min_val) / denom).clamp(0.0, 1.0)

        # Apply the dynamic weight to the speckle channel.
        weighted_speckle = speckle * weight

        # Return a 2-channel tensor: (2, H, W)
        return torch.cat([img_tensor, weighted_speckle], dim=0)


class SpeckleAwareViT(nn.Module):
    """
    Speckle-aware Vision Transformer using torchvision's ViT-B/16 backbone.

    Changes from stock ViT:
    - Replaces vit.conv_proj to accept 2-channel input (intensity + speckle).
    - Replaces classification head to output 4 classes.
    """
    def __init__(self, num_classes=4):
        super().__init__()

        # Load a pretrained ViT-B/16 backbone.
        # NOTE: In newer torchvision versions, the preferred API is weights=...
        # This script keeps pretrained=True to match your original code.
        self.vit = vit_b_16(pretrained=True)

        # Patch projection: change input channels from 3 -> 2.
        self.vit.conv_proj = nn.Conv2d(2, 768, kernel_size=16, stride=16)

        # Replace classification head.
        self.vit.heads.head = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.vit(x)


def plot_confusion_matrix(cm, class_names):
    """
    Render a confusion matrix heatmap with integer counts.
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_scores, n_classes):
    """
    Plot one-vs-rest ROC curves for multi-class classification.
    """
    y_true_bin = np.eye(n_classes)[y_true]
    class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """
    Train ViT with dynamic speckle weighting, early stopping on val accuracy,
    then evaluate the best checkpoint on the test split.
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # Training-time augmentations + dynamic speckle preprocessing.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        FastSpeckleMap(min_val=0.0, max_val=0.2),
        transforms.Normalize([0.5, 0.0], [0.5, 0.5]),
    ])

    # Validation/test transforms should be deterministic.
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        FastSpeckleMap(min_val=0.0, max_val=0.2),
        transforms.Normalize([0.5, 0.0], [0.5, 0.5]),
    ])

    # ImageFolder expects subfolders per class.
    train_loader = DataLoader(
        datasets.ImageFolder("OCT2017-Copy/train", transform=train_transform),
        batch_size=64,
        shuffle=True
    )
    val_loader = DataLoader(
        datasets.ImageFolder("OCT2017-Copy/val", transform=test_transform),
        batch_size=64,
        shuffle=False
    )
    test_loader = DataLoader(
        datasets.ImageFolder("OCT2017-Copy/test", transform=test_transform),
        batch_size=64,
        shuffle=False
    )

    model = SpeckleAwareViT(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )

    best_val_acc = 0.0
    patience = 5
    wait = 0

    for epoch in range(15):
        model.train()
        start_time = time.time()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/15")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=float(loss.item()))

        scheduler.step()

        epoch_time = time.time() - start_time
        epoch_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                val_loss += criterion(outputs, labels).item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1}/15 - Time: {epoch_time:.2f}s, "
            f"Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Early stopping + checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), "sav8.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # Test evaluation using the best checkpoint.
    model.load_state_dict(torch.load("sav8.pt", map_location=device))
    model.eval()

    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())
            y_scores.extend(F.softmax(outputs, dim=1).cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    print("Test Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["CNV", "DME", "DRUSEN", "NORMAL"]))

    plot_confusion_matrix(
        confusion_matrix(y_true, y_pred),
        ["CNV", "DME", "DRUSEN", "NORMAL"]
    )
    plot_roc_curve(y_true, y_scores, 4)


if __name__ == "__main__":
    main()
