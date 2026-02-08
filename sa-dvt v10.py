"""
SA-DVT (Swin-T) + Robust Speckle Map (edge-removed) + Learnable Gating + Focal Loss

What changed from the last version (sadvtv10: ViT-B/16 + edge-removed dynamic gating with fixed piecewise weights):
1) Backbone swap: ViT → Swin Transformer (Swin-T).
   - Uses hierarchical windowed attention better suited for local retinal structure.
   - Replaces Swin’s first conv to accept 1-channel input and adds a custom classification head.

2) Loss upgrade: CrossEntropy(label_smoothing=0.1) → FocalLoss(gamma=2.0, label_smoothing=0.15).
   - Down-weights “easy” examples and focuses learning on hard / minority / confusing samples.
   - Optional alpha weighting is supported (not enabled by default).

3) Gating upgrade: fixed piecewise blend → LearnableGating MLP.
   - Instead of hardcoded rules based on speckle_percent, a small MLP predicts [w_oct, w_speckle].
   - IMPORTANT: as written, the gate is set to eval() and not trained (so weights are effectively random).
     If you want “learnable” to mean trained, you must move the gate into the model so its parameters
     are optimized with the rest of the network.

4) Normalization is now dataset-specific (placeholders).
   - dataset_mean / dataset_std must be computed from YOUR processed images (after speckle transform)
     for best stability.

5) Reproducibility:
   - Added seed control (torch + numpy) and saves checkpoint as sadvt_best_seed{seed}.pt

Notes / gotchas:
- RobustSpeckleMap currently constructs a LearnableGating inside the transform and freezes it (eval mode).
  That means the gating is NOT actually learned unless you refactor it into the model.
- The transform returns a single blended 1-channel tensor, so Swin’s input conv is set to 1 channel.
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
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage import feature, morphology

# ---- Focal Loss Implementation ----
class FocalLoss(nn.Module):
    """
    Focal Loss with optional class weighting (alpha) and label smoothing.

    - gamma: focusing parameter. Higher => more focus on hard examples.
    - alpha: optional tensor of shape [num_classes] for class weighting.
    - label_smoothing: passed into cross_entropy (PyTorch >= 1.10).
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean", label_smoothing=0.15):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha  # torch.Tensor or None
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

    def forward(self, input, target):
        # Per-sample CE loss so we can focal-weight each example
        ce_loss = F.cross_entropy(
            input, target,
            reduction="none",
            label_smoothing=self.label_smoothing
        )

        # pt = probability of the true class (after CE)
        pt = torch.exp(-ce_loss)

        # focal scaling
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        # optional per-class weighting
        if self.alpha is not None:
            # alpha[target] for each example
            alpha_t = self.alpha.gather(0, target)
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---- Learnable Gating Mechanism ----
class LearnableGating(nn.Module):
    """
    Tiny MLP that maps a scalar speckle_percent (normalized to [0,1]) to
    two blending weights [w_oct, w_speckle] that sum to 1.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, speckle_percent):
        # speckle_percent: shape [B, 1]
        return self.fc(speckle_percent)


# ---- Robust Speckle Map Generator ----
class RobustSpeckleMap:
    """
    Produces a single blended 1-channel image tensor using:
      - bandpass filtering
      - local normalization
      - percentile thresholding
      - edge removal (Canny + dilation)
      - gating weights from LearnableGating(speckle_percent)

    IMPORTANT:
    - gate is NOT trained here (eval mode, used under no_grad()).
      To truly learn gating, integrate LearnableGating into the model.
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

        # Gating network (currently frozen / not trained)
        self.gate = LearnableGating()
        self.gate.eval()

    def __call__(self, img):
        # Standardize image
        img = img.convert("L").resize((224, 224))
        img_np = np.array(img).astype(np.float32) / 255.0  # (H, W)

        # ---- Bandpass filtering ----
        lowpass = gaussian_filter(img_np, sigma=self.low_sigma)
        highpass = gaussian_filter(img_np, sigma=self.high_sigma)
        bandpass = lowpass - highpass

        # ---- Local normalization with safe std ----
        local_mean = uniform_filter(bandpass, size=self.norm_win)
        local_std = np.sqrt(uniform_filter((bandpass - local_mean) ** 2, size=self.norm_win))
        safe_std = np.where(local_std < 1e-6, 1e-6, local_std)
        bandpass_norm = (bandpass - local_mean) / safe_std

        # ---- Percentile thresholding ----
        valid_norm = bandpass_norm[~np.isnan(bandpass_norm)]
        dynamic_thresh = np.percentile(valid_norm, self.percentile)

        speckle_mask = (bandpass_norm > dynamic_thresh) & (bandpass_norm > self.min_abs_thresh)

        # ---- Edge detection + removal ----
        edges = feature.canny(img_np, sigma=self.edge_sigma)
        thick_edges = morphology.dilation(edges, morphology.disk(self.edge_disk_radius))

        speckle_map = speckle_mask.astype(np.float32)
        speckle_map_no_edges = speckle_map * (~thick_edges).astype(np.float32)

        # ---- Learnable gating (currently inference-only) ----
        speckle_percent = (np.count_nonzero(speckle_map_no_edges) / speckle_map_no_edges.size) * 100.0
        speckle_percent_tensor = torch.tensor([[speckle_percent / 100.0]], dtype=torch.float32)

        with torch.no_grad():
            weights = self.gate(speckle_percent_tensor).cpu().numpy()[0]

        weight_oct, weight_speckle = float(weights[0]), float(weights[1])

        # Blend original OCT intensity with speckle map
        blended = (weight_oct * img_np) + (weight_speckle * speckle_map_no_edges)

        # Return 1-channel tensor (1, H, W)
        img_tensor = torch.from_numpy(blended).unsqueeze(0).float()
        return img_tensor


# ---- Swin Transformer Backbone ----
from torchvision.models import swin_t, Swin_T_Weights

class SpeckleAwareSwin(nn.Module):
    """
    Swin-T backbone adapted for 1-channel input (after gating fusion).

    - Replace first patch embed conv to accept 1 channel.
    - Remove default head and attach our own classifier head.
    """
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.DEFAULT)

        # Swin patch embedding conv: change input channels from 3 -> 1
        self.backbone.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)

        # Remove default classifier to get embeddings
        self.backbone.head = nn.Identity()

        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(768, num_classes)  # Swin-T outputs 768-d features

    def forward(self, x):
        x = self.backbone(x)     # [B, 768]
        x = self.dropout(x)
        x = self.head(x)         # [B, num_classes]
        return x


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix with integer annotations."""
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


def main(seed=42):
    """
    Train Swin-T classifier with robust speckle mapping and focal loss.
    Saves best model by validation accuracy and evaluates on the test set.
    """
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # NOTE: Replace these with your true dataset stats (after speckle_map_transform)
    dataset_mean = 0.38
    dataset_std = 0.21

    speckle_map_transform = RobustSpeckleMap(
        low_sigma=1.0,
        high_sigma=5.0,
        norm_win=15,
        percentile=85,
        min_abs_thresh=0.2,
        edge_sigma=2,
        edge_disk_radius=4
    )

    # Training transform:
    # - You can add augmentation BEFORE speckle_map_transform, but be careful:
    #   strong geometric aug can change edge density and speckle_percent distribution.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        speckle_map_transform,
        transforms.Normalize([dataset_mean], [dataset_std])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        speckle_map_transform,
        transforms.Normalize([dataset_mean], [dataset_std])
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

    model = SpeckleAwareSwin(num_classes=4, dropout=0.4).to(device)

    # Focal loss emphasizes hard examples; label smoothing helps prevent overconfidence
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.15)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
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

            total_loss += float(loss.item()) * images.size(0)
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

                val_loss += float(criterion(outputs, labels).item()) * images.size(0)
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
            torch.save(model.state_dict(), f"sadvt_best_seed{seed}.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # Final Evaluation
    ckpt_path = f"sadvt_best_seed{seed}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
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
