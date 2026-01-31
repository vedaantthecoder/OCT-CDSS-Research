"""
SA-DVT (ViT-B/16) with Robust Speckle Map + Edge-Removed Dynamic Gating (single-channel input)

What changed from the last version:
1) NEW edge-aware speckle suppression:
   - Added Canny edge detection + morphological dilation to identify thick structural edges.
   - Removed speckle detections on/near edges to reduce "false speckle" from layer boundaries and strong reflectors.
2) NEW dynamic gating by speckle density:
   - Compute speckle_percent (percent of pixels marked as speckle after edge removal).
   - Blend original OCT intensity and speckle map using a piecewise rule:
     • <= 5.5% speckle: use 100% OCT (ignore speckle map)
     • >= 13.5% speckle: 50% OCT + 50% speckle
     • between: linearly interpolate
3) Input to ViT is now SINGLE-CHANNEL:
   - RobustSpeckleMap returns a blended 1-channel tensor instead of a 2-channel tensor.
   - ViT patch projection changed from Conv2d(2, 768, ...) to Conv2d(1, 768, ...).
4) More permissive speckle thresholding:
   - percentile lowered to 85 and min_abs_thresh lowered to 0.2, then edge-removal + gating control density.

Notes:
- This version trades explicit "two-stream" input (img + speckle mask) for a fused representation that adapts to speckle content.
- If training becomes unstable, consider tightening thresholds (raise percentile or min_abs_thresh) or reducing the max speckle blend.
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
from skimage import feature, morphology


class RobustSpeckleMap:
    """
    Robust speckle map generator with edge removal + dynamic gating.

    Input:
      - PIL image (any mode) from ImageFolder.
    Output:
      - Torch tensor, shape (1, 224, 224), float32.

    Steps:
    1) Convert to grayscale + resize to 224x224.
    2) Bandpass filter using two Gaussians (low_sigma, high_sigma).
    3) Locally normalize bandpass response using windowed mean/std.
    4) Threshold with percentile + absolute minimum threshold to create a speckle mask.
    5) Detect strong edges via Canny, dilate edges, and remove speckle detections near edges.
    6) Compute speckle density (%) and blend:
         blended = weight_oct * img + weight_speckle * speckle_map_no_edges
       using a piecewise mapping based on speckle density.
    """
    def __init__(
        self,
        low_sigma=1.0,
        high_sigma=5.0,
        norm_win=15,
        percentile=85,
        min_abs_thresh=0.2,
        edge_sigma=2,
        edge_disk_radius=4
    ):
        self.low_sigma = float(low_sigma)
        self.high_sigma = float(high_sigma)
        self.norm_win = int(norm_win)
        self.percentile = float(percentile)
        self.min_abs_thresh = float(min_abs_thresh)
        self.edge_sigma = float(edge_sigma)
        self.edge_disk_radius = int(edge_disk_radius)

    def __call__(self, img):
        # Standardize input
        img = img.convert("L").resize((224, 224))
        img_np = np.array(img).astype(np.float32) / 255.0  # (H, W) in [0, 1]

        # ---- Bandpass filtering (speckle-like frequency emphasis) ----
        lowpass = gaussian_filter(img_np, sigma=self.low_sigma)
        highpass = gaussian_filter(img_np, sigma=self.high_sigma)
        bandpass = lowpass - highpass

        # ---- Local normalization with safe std (avoid div-by-0) ----
        local_mean = uniform_filter(bandpass, size=self.norm_win)
        local_std = np.sqrt(uniform_filter((bandpass - local_mean) ** 2, size=self.norm_win))
        safe_std = np.where(local_std < 1e-6, 1e-6, local_std)
        bandpass_norm = (bandpass - local_mean) / safe_std

        # ---- Dynamic thresholding (percentile over valid values) ----
        valid_norm = bandpass_norm[~np.isnan(bandpass_norm)]
        dynamic_thresh = np.percentile(valid_norm, self.percentile)
        speckle_mask = (bandpass_norm > dynamic_thresh) & (bandpass_norm > self.min_abs_thresh)

        # ---- Edge detection + removal to avoid structural boundaries being labeled speckle ----
        edges = feature.canny(img_np, sigma=self.edge_sigma)
        thick_edges = morphology.dilation(edges, morphology.disk(self.edge_disk_radius))

        speckle_map = speckle_mask.astype(np.float32)
        speckle_map_no_edges = speckle_map * (~thick_edges).astype(np.float32)

        # ---- Dynamic gating based on speckle density (%) ----
        speckle_percent = (np.count_nonzero(speckle_map_no_edges) / speckle_map_no_edges.size) * 100.0

        # Piecewise schedule:
        #  - low speckle -> rely on OCT intensity
        #  - high speckle -> blend in speckle map
        if speckle_percent <= 5.5:
            weight_oct = 1.0
            weight_speckle = 0.0
        elif speckle_percent >= 13.5:
            weight_oct = 0.5
            weight_speckle = 0.5
        else:
            # Linear interpolation between (5.5 -> 0% speckle contribution)
            # and (13.5 -> 50% speckle contribution)
            alpha = (speckle_percent - 5.5) / (13.5 - 5.5)
            weight_oct = 1.0 - 0.5 * alpha
            weight_speckle = 0.5 * alpha

        blended = (weight_oct * img_np) + (weight_speckle * speckle_map_no_edges)

        # Return 1-channel tensor for downstream Normalize([0.5],[0.5])
        img_tensor = torch.from_numpy(blended).unsqueeze(0).float()
        return img_tensor


class SpeckleAwareViT(nn.Module):
    """
    ViT-B/16 adapted for SINGLE-CHANNEL input (after dynamic gating fusion).

    Changes vs standard ViT:
      - conv_proj input channels: 1 instead of 3
      - classifier head output: num_classes
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.conv_proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
        self.vit.heads.head = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.vit(x)


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix with annotations."""
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
    """Train + validate with early stopping, then evaluate on test set."""
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    speckle_map_transform = RobustSpeckleMap(
        low_sigma=1.0,
        high_sigma=5.0,
        norm_win=15,
        percentile=85,
        min_abs_thresh=0.2,
        edge_sigma=2,
        edge_disk_radius=4
    )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        speckle_map_transform,
        transforms.Normalize([0.5], [0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        speckle_map_transform,
        transforms.Normalize([0.5], [0.5])
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
            torch.save(model.state_dict(), "sadvtv10.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # Final Evaluation
    model.load_state_dict(torch.load("sadvtv10.pt", map_location=device))
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
