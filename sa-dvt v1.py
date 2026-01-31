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


# -----------------------------
# Data transforms (speckle-aware)
# -----------------------------

class SpeckleNoise(object):
    """
    Multiplicative speckle-like noise augmentation.

    Given an image tensor in [0, 1], inject noise as:
      img + img * (N(0, std))

    This approximates multiplicative noise seen in OCT, and helps improve robustness.
    """
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, img_tensor):
        noise = torch.randn_like(img_tensor) * self.std
        return torch.clamp(img_tensor + img_tensor * noise, 0.0, 1.0)


class SpeckleMap(object):
    """
    Create a 2-channel input:
      channel 0: original image
      channel 1: speckle residual = original - local mean (box blur)

    This is a lightweight way to expose high-frequency speckle/texture information
    to the model while preserving the original intensity.
    """
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

        # Box-blur kernel used for local averaging.
        # Shape must be (out_channels, in_channels, kH, kW) for conv2d.
        self.registered_kernel = torch.ones((1, 1, kernel_size, kernel_size)) / (kernel_size ** 2)

    def __call__(self, img_tensor):
        # img_tensor is expected to be (1, H, W) after Grayscale + ToTensor
        pad = self.kernel_size // 2

        # conv2d expects a batch dimension: (N, C, H, W)
        blurred = F.conv2d(
            img_tensor.unsqueeze(0),           # (1, 1, H, W)
            self.registered_kernel,            # (1, 1, k, k)
            padding=pad
        ).squeeze(0)                           # back to (1, H, W)

        speckle_residual = img_tensor - blurred

        # Return 2-channel tensor: (2, H, W)
        return torch.cat((img_tensor, speckle_residual), dim=0)


# -----------------------------
# SA-DVT style model (ViT encoder with optional gating)
# -----------------------------

class TransformerEncoderLayer(nn.Module):
    """
    One transformer encoder block with:
    - Pre-norm self-attention
    - Pre-norm MLP
    - Optional "gating" that increases the influence of certain tokens

    Gating details:
    - A scalar gate per token is learned from the normalized token embeddings.
    - The class token (index 0) is forced to gate=1.0 so it is not attenuated.
    """
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim, use_gating=True):
        super().__init__()
        self.use_gating = use_gating

        # Multi-head self-attention (batch_first=True means input is (B, T, D))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Feed-forward MLP block
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

        # LayerNorms for pre-norm architecture
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Token-wise gating head (learns importance per token)
        if use_gating:
            self.gate_fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape: (B, T, D)
        y = self.norm1(x)

        # Optional gating: scales each token embedding before attention
        if self.use_gating:
            gate = torch.sigmoid(self.gate_fc(y))  # (B, T, 1)
            gate = gate.clone()                    # avoid in-place ops that can break autograd
            gate[:, 0, :] = 1.0                    # ensure the class token is not gated down
            y = y * gate

        # Residual self-attention
        x = x + self.attn(y, y, y)[0]

        # Residual MLP
        x = x + self.mlp(self.norm2(x))
        return x


class SpeckleAwareViT(nn.Module):
    """
    A compact Vision Transformer that consumes 2-channel speckle-aware inputs:
      - channel 0: grayscale image
      - channel 1: speckle residual

    Pipeline:
    1) Patch embedding via Conv2d(stride=patch_size)
    2) Add class token + positional embedding
    3) Transformer encoder stack
    4) Classifier head on the class token
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_chans=2,
        num_classes=4,
        embed_dim=256,
        depth=8,
        num_heads=8
    ):
        super().__init__()

        # Converts (B, C, H, W) into patch embeddings (B, D, H/ps, W/ps)
        self.patch_embed = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Number of patches in the grid
        num_patches = (image_size // patch_size) ** 2

        # Learnable class token and positional embeddings
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(0.1)

        # Transformer encoder blocks
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x shape: (B, 2, 224, 224)

        # Patchify -> flatten spatial grid -> tokens
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, T, D)

        # Prepend class token
        class_tokens = self.class_token.expand(x.size(0), -1, -1)  # (B, 1, D)
        x = torch.cat((class_tokens, x), dim=1)                    # (B, T+1, D)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder
        for layer in self.layers:
            x = layer(x)

        # Classifier head on the class token
        cls = self.norm(x)[:, 0]  # (B, D)
        return self.fc(cls)       # (B, num_classes)


# -----------------------------
# Evaluation utilities
# -----------------------------

def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix heatmap.

    cm: confusion matrix of shape (num_classes, num_classes)
    class_names: list of class labels in the same order as cm indices
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Label ticks
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_scores, n_classes):
    """
    Plot one-vs-rest ROC curves for a multi-class classifier.

    y_true: integer labels, shape (N,)
    y_scores: probability scores, shape (N, n_classes)
    """
    y_true_bin = np.eye(n_classes)[y_true]  # one-hot encoding

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main training + evaluation
# -----------------------------

def main():
    """
    Train a speckle-aware ViT on OCT2017 (4-class classification) and evaluate on a held-out test split.

    Dataset structure expected (ImageFolder):
      OCT2017/train/<class_name>/*.png
      OCT2017/val/<class_name>/*.png
      OCT2017/test/<class_name>/*.png

    Important:
    - This code uses MPS if available; otherwise CPU.
    - If you have CUDA, you can swap the device logic accordingly.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Training-time augmentation: includes random horizontal flip + speckle noise
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        SpeckleNoise(0.1),
        SpeckleMap(),
        # Normalize 2-channel tensor: [image, residual]
        transforms.Normalize([0.5, 0.0], [0.5, 0.5]),
    ])

    # Validation/test transforms: deterministic preprocessing, no random aug, no noise injection
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        SpeckleMap(),
        transforms.Normalize([0.5, 0.0], [0.5, 0.5]),
    ])

    # DataLoaders
    train_loader = DataLoader(
        datasets.ImageFolder("OCT2017/train", transform=train_transform),
        batch_size=32,
        shuffle=True,
    )
    val_loader = DataLoader(
        datasets.ImageFolder("OCT2017/val", transform=test_transform),
        batch_size=32,
        shuffle=False,
    )
    test_loader = DataLoader(
        datasets.ImageFolder("OCT2017/test", transform=test_transform),
        batch_size=32,
        shuffle=False,
    )

    # Model, optimizer, loss
    model = SpeckleAwareViT().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        model.train()
        start_time = time.time()

        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/10")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            # Track epoch loss (sum of losses weighted by batch size)
            total_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=float(loss.item()))

        epoch_time = time.time() - start_time
        epoch_loss = total_loss / len(train_loader.dataset)

        # Validation step
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
            f"Epoch {epoch + 1}/10 - Time: {epoch_time:.2f}s, "
            f"Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    # Test evaluation
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
    print(classification_report(y_true, y_pred))

    # Confusion matrix + ROC curves
    class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]
    plot_confusion_matrix(confusion_matrix(y_true, y_pred), class_names)
    plot_roc_curve(y_true, y_scores, n_classes=4)


if __name__ == "__main__":
    main()
