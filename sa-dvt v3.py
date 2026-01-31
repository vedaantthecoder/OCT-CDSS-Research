"""
Speckle-Aware ViT (SA-DVT-style) training + evaluation for OCT2017.

What changed in this version:
1) Speckle preprocessing is faster: uses GaussianBlur residual inside FastSpeckleMap instead of conv2d box blur.
2) torch.compile is now conditional: compiled on CUDA/CPU, skipped on MPS for better compatibility.
3) Training is set to 10 epochs (even though some progress-bar/print strings still say /30, see notes below).
4) The rest of the training stack stays focused on stability: AdamW, cosine warm restarts, label smoothing, and gradient clipping.

Notes you may want to fix:
- The loop runs `for epoch in range(10)` but tqdm/print uses `/30`. Update those strings to `/10` if you want consistency.
- CosineAnnealingWarmRestarts is often stepped per-iteration or with epoch+fraction. This script steps once per epoch, which is valid but behaves differently.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time


class FastSpeckleMap:
    """
    Fast speckle-aware 2-channel input construction.

    Input: a PIL image
    Output: a 2-channel torch tensor of shape (2, H, W):
      - channel 0: grayscale image intensity
      - channel 1: speckle residual = intensity - GaussianBlur(intensity)

    Using a Gaussian blur is typically faster than a custom conv2d kernel for preprocessing,
    and it highlights high-frequency OCT texture similar to speckle.
    """
    def __call__(self, img):
        # Ensure grayscale and convert to tensor in [0, 1]
        img = img.convert("L")
        img_tensor = transforms.ToTensor()(img)  # (1, H, W)

        # Low-frequency estimate (local mean) and residual (high-frequency detail)
        blurred = transforms.GaussianBlur(5)(img_tensor)
        speckle = img_tensor - blurred

        # Return (2, H, W)
        return torch.cat([img_tensor, speckle], dim=0)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder block with:
    - Pre-norm multi-head self-attention
    - Pre-norm MLP
    - Optional token gating

    Token gating:
    - Learns a per-token scalar gate in [0, 1].
    - Forces the class token (token index 0) to have gate=1.0 so it is never attenuated.
    """
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim, use_gating=True):
        super().__init__()
        self.use_gating = use_gating

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        if use_gating:
            self.gate_fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (B, T, D)
        y = self.norm1(x)

        if self.use_gating:
            # gate: (B, T, 1)
            gate = torch.sigmoid(self.gate_fc(y)).clone()
            gate[:, 0, :] = 1.0  # keep class token fully active
            y = y * gate

        # Residual attention
        x = x + self.attn(y, y, y)[0]

        # Residual MLP
        x = x + self.mlp(self.norm2(x))
        return x


class SpeckleAwareViT(nn.Module):
    """
    Speckle-aware Vision Transformer classifier.

    Expects a 2-channel input produced by FastSpeckleMap:
      - channel 0: grayscale intensity
      - channel 1: speckle residual

    Steps:
    1) Patch embedding via Conv2d (kernel=stride=patch_size)
    2) Prepend a learnable class token and add positional embeddings
    3) Transformer encoder stack
    4) Classification head applied to the class token output
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_chans=2,
        num_classes=4,
        embed_dim=512,
        depth=10,
        num_heads=8
    ):
        super().__init__()

        # Patch embedding: (B, C, H, W) -> (B, D, H/ps, W/ps)
        self.patch_embed = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Number of patch tokens
        num_patches = (image_size // patch_size) ** 2

        # Learnable class token and positional embeddings
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        # Encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_hidden_dim=embed_dim * 4)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patchify: (B, 2, 224, 224) -> (B, T, D)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Add class token and positional encoding
        class_token = self.class_token.expand(x.size(0), -1, -1)  # (B, 1, D)
        x = torch.cat([class_token, x], dim=1) + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder stack
        for block in self.blocks:
            x = block(x)

        # Classify using the class token
        x = self.norm(x)
        return self.head(x[:, 0])


def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix heatmap with numeric labels.

    cm: confusion matrix (num_classes x num_classes)
    class_names: list of class labels in index order
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Annotate each cell with its count
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
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

    y_true: integer labels, shape (N,)
    y_scores: probability scores (softmax), shape (N, n_classes)
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
    Train and evaluate SpeckleAwareViT on OCT2017.

    Device preference order:
    - MPS (Apple Silicon)
    - CUDA (NVIDIA)
    - CPU
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # Training transforms:
    # - geometric augmentation
    # - speckle residual channel via FastSpeckleMap
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        FastSpeckleMap(),
        transforms.Normalize([0.5, 0.0], [0.5, 0.5]),
    ])

    # Validation/test transforms should be deterministic
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        FastSpeckleMap(),
        transforms.Normalize([0.5, 0.0], [0.5, 0.5]),
    ])

    # Dataloaders (ImageFolder expects subfolders named by class)
    train_loader = DataLoader(
        datasets.ImageFolder("OCT2017/train", transform=train_transform),
        batch_size=64,
        shuffle=True
    )
    val_loader = DataLoader(
        datasets.ImageFolder("OCT2017/val", transform=test_transform),
        batch_size=64,
        shuffle=False
    )
    test_loader = DataLoader(
        datasets.ImageFolder("OCT2017/test", transform=test_transform),
        batch_size=64,
        shuffle=False
    )

    model = SpeckleAwareViT().to(device)

    # torch.compile can boost performance on CUDA/CPU, but can be unstable on MPS.
    if not torch.backends.mps.is_available():
        model = torch.compile(model)
    else:
        print("Skipping torch.compile() on MPS for compatibility.")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Warm restarts periodically raise the learning rate to help escape shallow minima.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )

    # Label smoothing reduces overconfidence and can improve generalization.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    patience = 5
    wait = 0

    # Training loop (note: loop is 10 epochs; update display strings if desired)
    for epoch in range(10):
        model.train()
        start_time = time.time()
        total_loss = 0.0

        # The string says /30, but the loop runs 10 epochs.
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/30")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            # Clip gradients for stability with transformer-style models.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=float(loss.item()))

        # Scheduler step per epoch (simple usage pattern)
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
            f"Epoch {epoch + 1}/30 - Time: {epoch_time:.2f}s, "
            f"Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Early stopping + best checkpoint saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), "best_sadvt.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # Final evaluation on test set using the best checkpoint
    model.load_state_dict(torch.load("best_sadvt.pt"))
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
