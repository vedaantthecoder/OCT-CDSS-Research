"""
Speckle-Aware ViT with Robust Speckle Map (bandpass + local normalization + percentile mask)

What changed from the previous version:
1) Replaced simple Gaussian residual weighting with a *robust speckle map generator*:
   - Bandpass filtering (low_sigma vs high_sigma) to isolate speckle-like frequencies.
   - Local normalization (windowed mean/std) to stabilize contrast across the scan.
   - Percentile-based thresholding to create a binary speckle mask (more robust than global scaling).
2) Added signal masking to avoid labeling background/low-intensity regions as speckle.
3) The second channel is now a *binary speckle map* (0/1) rather than a continuous residual.
4) The transform stores `last_speckle_prop` (speckle proportion in signal area) for optional debugging/analysis.
5) Training loop and ViT patch projection stay the same (2-channel conv_proj into ViT-B/16).

Tuning tips (quick):
- If speckle map is too sparse: lower `percentile` (e.g., 95) or lower `min_abs_thresh`.
- If speckle map is too dense/noisy: raise `percentile` (e.g., 99) or raise `signal_thresh`.
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
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter


class RobustSpeckleMap:
    """
    Robust speckle map generator that produces a 2-channel tensor:
      Channel 0: normalized grayscale OCT intensity
      Channel 1: binary speckle mask (robustly estimated)

    Pipeline:
    1) Resize + grayscale -> img_np in [0, 1]
    2) Bandpass: Gaussian(low_sigma) - Gaussian(high_sigma)
    3) Local normalization: (bandpass - local_mean) / (local_std + eps)
    4) Dynamic threshold: keep only top percentile values
    5) Enforce a minimum absolute threshold to avoid near-zero noise
    6) Remove low-signal regions using `signal_thresh`
    7) Return [img_tensor, speckle_tensor] as (2, H, W)

    Side output:
    - last_speckle_prop: ratio of speckle pixels to valid signal pixels (for debugging)
    """
    def __init__(
        self,
        low_sigma=1.0,
        high_sigma=5.0,
        norm_win=15,
        percentile=95,
        min_abs_thresh=0.5,
        signal_thresh=0.1
    ):
        self.low_sigma = float(low_sigma)
        self.high_sigma = float(high_sigma)
        self.norm_win = int(norm_win)
        self.percentile = float(percentile)
        self.min_abs_thresh = float(min_abs_thresh)
        self.signal_thresh = float(signal_thresh)
        self.last_speckle_prop = 0.0

    def __call__(self, img):
        # Ensure grayscale and consistent spatial size for the transformer.
        img = img.convert("L").resize((224, 224))

        # Convert to numpy float in [0, 1], shape: (H, W)
        img_np = (np.array(img).astype(np.float32) / 255.0)

        # --- Bandpass filtering ---
        lowpass = gaussian_filter(img_np, sigma=self.low_sigma)
        highpass = gaussian_filter(img_np, sigma=self.high_sigma)
        bandpass = lowpass - highpass

        # --- Local normalization ---
        local_mean = uniform_filter(bandpass, size=self.norm_win)
        local_var = uniform_filter((bandpass - local_mean) ** 2, size=self.norm_win)
        local_std = np.sqrt(local_var)
        bandpass_norm = (bandpass - local_mean) / (local_std + 1e-8)

        # --- Dynamic, percentile-based thresholding ---
        # Keep the strongest responses relative to this image's distribution.
        dynamic_thresh = np.percentile(bandpass_norm, self.percentile)

        # Require both:
        # 1) above percentile threshold
        # 2) above absolute minimum (prevents weak fluctuations from passing)
        speckle_mask = (bandpass_norm > dynamic_thresh) & (bandpass_norm > self.min_abs_thresh)

        # --- Mask out low-signal/background regions ---
        signal_mask = img_np > self.signal_thresh
        speckle_map = (speckle_mask & signal_mask).astype(np.float32)

        # Cache speckle proportion in valid signal region (optional debug metric).
        denom = float(signal_mask.sum()) + 1e-8
        self.last_speckle_prop = float(speckle_map.sum() / denom)

        # Convert to torch tensors: (1, H, W)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()
        speckle_tensor = torch.from_numpy(speckle_map).unsqueeze(0).float()

        # Return 2-channel tensor: (2, H, W)
        return torch.cat([img_tensor, speckle_tensor], dim=0)


class SpeckleAwareViT(nn.Module):
    """
    ViT-B/16 adapted to accept 2-channel input (intensity + speckle mask).

    - Replace patch projection (conv_proj) to accept 2 channels.
    - Replace classifier head for 4 OCT classes.
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.conv_proj = nn.Conv2d(2, 768, kernel_size=16, stride=16)
        self.vit.heads.head = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.vit(x)


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix with cell count annotations."""
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
    """Plot one-vs-rest ROC curves for each class."""
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
    Train SpeckleAwareViT on OCT2017-Copy with robust speckle-mask second channel,
    save the best checkpoint by validation accuracy, then evaluate on the test split.
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # Instantiate one shared transform object so you can inspect last_speckle_prop if desired.
    speckle_map_transform = RobustSpeckleMap(
        low_sigma=1.0,
        high_sigma=5.0,
        norm_win=15,
        percentile=97,      # Higher -> fewer speckle pixels (more selective)
        min_abs_thresh=0.5,  # Higher -> suppress weaker responses
        signal_thresh=0.1    # Higher -> ignore darker regions more aggressively
    )

    # Training-time augmentation + robust speckle map generation.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        speckle_map_transform,
        transforms.Normalize([0.5, 0.0], [0.5, 0.5]),
    ])

    # Validation/test transforms should remain deterministic.
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        speckle_map_transform,
        transforms.Normalize([0.5, 0.0], [0.5, 0.5]),
    ])

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
            torch.save(model.state_dict(), "newmapsadvt.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # Final Evaluation
    model.load_state_dict(torch.load("newmapsadvt.pt", map_location=device))
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
