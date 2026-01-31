"""
Speckle-Aware ViT with CNN Frontend (OCT2017-Copy) - Training + Evaluation

What changed in this version:
1) Added a small CNN frontend (CNNFrontend) to learn local texture features before the transformer.
2) The CNN maps 2-channel speckle input to a 3-channel feature image so the pretrained ViT can be used
   without modifying its internal patch projection layer.
3) Switched to torchvision's weights API (ViT_B_16_Weights.DEFAULT) for cleaner, version-stable loading.
4) Checkpoint name updated to "sa_dvt_cnn_frontend.pt" to distinguish it from earlier runs.

Expected folder structure (ImageFolder):
  OCT2017-Copy/train/<class>/*
  OCT2017-Copy/val/<class>/*
  OCT2017-Copy/test/<class>/*

Notes:
- The CNN frontend downsamples then upsamples back to 224x224 to match ViT input size.
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
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights


class FastSpeckleMap:
    """
    Speckle-aware preprocessing that returns a 2-channel tensor.

    Channel 0: grayscale intensity
    Channel 1: speckle residual (intensity - GaussianBlur(intensity))

    Output shape: (2, H, W) in float32, typically in [-1, 1] after Normalize().
    """
    def __call__(self, img):
        # Force grayscale so intensity statistics are consistent across inputs.
        img = img.convert("L")

        # Convert PIL -> Tensor in [0, 1], shape: (1, H, W)
        img_tensor = transforms.ToTensor()(img)

        # Low-pass estimate via Gaussian blur.
        blurred = transforms.GaussianBlur(5)(img_tensor)

        # High-frequency residual captures speckle-like components.
        speckle = img_tensor - blurred

        # Concatenate to produce a 2-channel tensor.
        return torch.cat([img_tensor, speckle], dim=0)


class CNNFrontend(nn.Module):
    """
    Lightweight CNN frontend that maps 2-channel speckle input -> 3 channels.

    Motivation:
    - Pretrained ViT-B/16 expects 3-channel images.
    - Instead of changing ViT's patch projection to accept 2 channels, we learn a small CNN
      that produces a 3-channel feature image compatible with the pretrained ViT.

    Design:
    - Two downsampling stages (MaxPool2d) then a learned projection to out_ch,
      followed by upsampling back to 224x224.
    """
    def __init__(self, in_ch=2, out_ch=3):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1: 2 -> 32 channels, downsample by 2
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 -> 64 channels, downsample by 2 again
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 -> out_ch channels (3 by default)
            nn.Conv2d(64, out_ch, kernel_size=3, padding=1),

            # Restore spatial size back to 224x224 (input to ViT)
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 2, 224, 224)
        Returns:
            Tensor of shape (B, 3, 224, 224)
        """
        return self.conv(x)


class SpeckleAwareViT(nn.Module):
    """
    End-to-end model: (FastSpeckleMap output) -> CNNFrontend -> pretrained ViT-B/16 -> classifier head.

    - ViT weights are loaded with torchvision's weights API.
    - The ViT classifier head is replaced to match the 4 OCT classes.
    """
    def __init__(self, num_classes=4):
        super().__init__()

        # CNN maps 2-channel speckle tensor into 3 channels expected by pretrained ViT.
        self.cnn_frontend = CNNFrontend(in_ch=2, out_ch=3)

        # Pretrained ViT backbone (ImageNet initialization).
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        # Replace the final classification head.
        self.vit.heads.head = nn.Linear(768, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 2, 224, 224)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        x = self.cnn_frontend(x)
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
    Train with early stopping on the validation set, save best weights,
    and report test metrics (accuracy, classification report, confusion matrix, ROC curves).
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # Training-time augmentation plus speckle-aware preprocessing.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        FastSpeckleMap(),
        transforms.Normalize([0.5, 0.0], [0.5, 0.5]),
    ])

    # Deterministic transforms for validation/testing.
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        FastSpeckleMap(),
        transforms.Normalize([0.5, 0.0], [0.5, 0.5]),
    ])

    # Dataloaders (ImageFolder infers class indices from folder names).
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

    # Cross-entropy with label smoothing helps reduce overconfidence.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Cosine warm restarts periodically increases LR after annealing.
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

        # Validation pass
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

        # Early stopping + checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), "sa_dvt_cnn_frontend.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # Final evaluation using the best checkpoint
    model.load_state_dict(torch.load("sa_dvt_cnn_frontend.pt", map_location=device))
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
