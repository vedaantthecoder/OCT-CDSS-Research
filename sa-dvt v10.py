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
        valid_norm = bandpass_norm[~np.isnan(bandpass_n_]()
