"""
Speckle-Aware ViT (ViT-B/16) Fine-Tuning on OCT2017-Copy (2-channel speckle input).

What changed in this version:
1) Uses torchvision.models.vision_transformer.vit_b_16 directly (instead of the broader torchvision.models API).
2) Simplifies the model: removes the custom gated encoder and reuses the stock ViT encoder.
3) Keeps the speckle-aware 2-channel preprocessing, but updates dataset paths to "OCT2017-Copy/<split>".
4) Trains for up to 15 epochs (with early stopping) and saves the best checkpoint as "sav7.pt".

Notes:
- This script expects ImageFolder structure:
    OCT2017-Copy/train/<class>/*
    OCT2017-Copy/val/<class>/*
    OCT2017-Copy/test/<class>/*
- torch.compile is not used here (maximum compatibility across MPS/CUDA/CPU).
- No secrets or API keys should ever be committed to GitHub.
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
    Speckle-aware transform that returns a 2-channel tensor.

    Steps:
    1) Convert PIL image to grayscale.
    2) Convert to tensor in [0, 1] with shape (1, H, W).
    3) Apply a Gaussian blur to estimate low-frequency content.
    4) Compute speckle residual = original - blurred.
    5) Concatenate [original, residual] -> (2, H, W).
    """
    def __call__(self, img):
        img = img.convert("L")
        img_tensor = transforms.ToTensor()(img)          # (1, H, W)
        blurred = transforms.GaussianBlur(5)(img_tensor) # (1, H, W)
        speckle = img_tensor - blurred                   # (1, H, W)
        return torch.cat([img_tensor, speckle], dim=0)   # (2, H, W)


class SpeckleAwareViT(nn.Module):
    """
    ViT-B/16 fine-tuner adapted for 2-channel speckle-aware input.

    Modifications from the default ViT-B/16:
    - conv_proj changed from 3 input channels (RGB) to 2 channels (intensity + speckle residual).
    - classification head replaced for 4 OCT classes.
    """
    def __init__(self, num_classes=4):
        super().__init__()

        # Create pretrained ViT-B/16 (ImageNet initialization).
        # Note: In newer torchvision versions, the preferred arg is `weights=...` instead of `pretrained=True`.
        self.vit = vit_b_16(pretrained=True)

        # Patch projection expects (B, C, 224, 224). We change C from 3 -> 2.
        self.vit.conv_proj = nn.Conv2d(2, 768, kernel_size=16, stride=16)

        # Replace classifier head for OCT classes.
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
    Train/validate ViT-B/16 with speckle-aware 2-channel inputs, then evaluate on the test set.
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # Training-time augmentation to improve robustness.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        FastSpeckleMap(),
        transforms.Normalize([0.5, 0.0], [0.5, 0.5]),
    ])

    # Validation/test transforms should be deterministic.
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        FastSpeckleMap(),
        transforms.Normalize([0.5, 0.0], [0.5, 0.5]),
    ])

    # ImageFolder expects: root/<class_name>/*.png|jpg|...
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

    # Cross-entropy with label smoothing to reduce overconfidence.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW is a strong default for ViT fine-tuning.
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Cosine warm restarts: periodically raises LR to encourage exploration.
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

        # Validation loop
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
            torch.save(model.state_dict(), "sav7.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # Final evaluation using the best checkpoint
    model.load_state_dict(torch.load("sav7.pt", map_location=device))
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
