"""
Speckle-Aware Vision Transformer with pretrained ViT-B/16 backbone + gated encoder (SA-DVT-style).

What changed in this version:
1) Pretrained backbone: uses torchvision's ViT-B/16 ImageNet weights for initialization.
2) 2-channel input support: replaces the original ViT patch projection with a new Conv2d that accepts
   (intensity, speckle residual) channels while keeping ViT-B/16 embedding dimension.
3) Custom gated transformer: swaps the stock ViT encoder for a custom SAEncoder that applies token gating
   before attention in every layer while preserving the class token.
4) Classifier head: uses a new linear head for 4-class OCT classification.

- This script expects an ImageFolder dataset layout at OCT2017/train, OCT2017/val, OCT2017/test.
- torch.compile is skipped on MPS for compatibility.
- This is a training and evaluation script; it saves the best checkpoint by validation accuracy.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time


class FastSpeckleMap:
    """
    Fast speckle-aware preprocessing that outputs a 2-channel tensor.

    Given a PIL image:
    1) Convert to grayscale (L)
    2) Convert to tensor in [0, 1] with shape (1, H, W)
    3) Compute a low-frequency estimate using Gaussian blur
    4) Compute speckle residual = original - blurred
    5) Return a 2-channel tensor: [original, residual] with shape (2, H, W)
    """
    def __call__(self, img):
        img = img.convert("L")
        img_tensor = transforms.ToTensor()(img)          # (1, H, W)
        blurred = transforms.GaussianBlur(5)(img_tensor) # (1, H, W)
        speckle = img_tensor - blurred                   # (1, H, W)
        return torch.cat([img_tensor, speckle], dim=0)   # (2, H, W)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with token gating.

    Structure:
    - Pre-norm
    - Token gating (learned per-token scalar in [0, 1])
      The class token gate is forced to 1.0 to avoid suppressing it.
    - Multi-head self-attention + residual
    - MLP + residual
    """
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Produces a single gate scalar per token
        self.gate_fc = nn.Linear(embed_dim, 1)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, T, D)
        y = self.norm1(x)

        # gate: (B, T, 1), applied multiplicatively to token features
        gate = torch.sigmoid(self.gate_fc(y)).clone()
        gate[:, 0, :] = 1.0  # keep class token fully active
        y = y * gate

        # Attention block with residual connection
        x = x + self.attn(y, y, y)[0]

        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class SAEncoder(nn.Module):
    """
    Custom encoder stack made of gated TransformerEncoderLayer blocks.

    This mirrors a ViT encoder conceptually, but explicitly inserts token gating
    before attention in every layer.
    """
    def __init__(self, embed_dim, depth, num_heads):
        super().__init__()
        self.dropout = nn.Dropout(0.1)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_hidden_dim=embed_dim * 4)
            for _ in range(depth)
        ])

        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, T, D)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return self.ln(x)


class SpeckleAwareViT(nn.Module):
    """
    Speckle-aware ViT classifier built on top of pretrained ViT-B/16.

    Key idea:
    - Use ImageNet-pretrained ViT-B/16 weights for token embeddings and positional embeddings.
    - Replace the patch projection so it accepts 2 channels (intensity + speckle residual).
    - Replace the encoder with a custom gated encoder (SAEncoder).
    - Train a fresh classification head for OCT classes.
    """
    def __init__(self, num_classes=4, patch_size=16, image_size=224):
        super().__init__()

        # Load a pretrained ViT-B/16 to reuse its embedding dimension and token/position parameters
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # Patch projection modified for 2-channel input, matching ViT hidden dimension
        self.conv_proj = nn.Conv2d(
            2,
            vit.conv_proj.out_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Reuse class token and positional embeddings from the pretrained ViT
        # (These are learnable parameters and will be fine-tuned unless you freeze them.)
        self.cls_token = vit.class_token
        self.pos_embed = vit.encoder.pos_embedding

        # Custom gated encoder (depth/heads chosen to match ViT-B/16 defaults)
        self.encoder = SAEncoder(embed_dim=vit.hidden_dim, depth=12, num_heads=12)

        # New classifier head for OCT2017 4-class output
        self.head = nn.Linear(vit.hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, 2, 224, 224)

        # Patchify: (B, 2, 224, 224) -> (B, D, 14, 14)
        x = self.conv_proj(x)

        # Flatten patches: (B, D, 14, 14) -> (B, N, D) where N=196
        x = x.flatten(2).transpose(1, 2)

        B, N, D = x.shape

        # Prepend class token: (B, 1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Add positional embeddings (expects sequence length N+1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed

        # Gated transformer encoder
        x = self.encoder(x)

        # Classify from class token output
        return self.head(x[:, 0])


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
    Train the pretrained-gated speckle-aware ViT and evaluate on the test set.
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # Training transforms include geometric augmentation
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

    # ImageFolder expects OCT2017/<split>/<class>/*
    train_loader = DataLoader(
        datasets.ImageFolder("OCT2017/train", transform=train_transform),
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        datasets.ImageFolder("OCT2017/val", transform=test_transform),
        batch_size=32,
        shuffle=False
    )
    test_loader = DataLoader(
        datasets.ImageFolder("OCT2017/test", transform=test_transform),
        batch_size=32,
        shuffle=False
    )

    model = SpeckleAwareViT().to(device)

    # Compile can accelerate CUDA/CPU, but can be problematic on MPS.
    if not torch.backends.mps.is_available():
        model = torch.compile(model)
    else:
        print("Skipping torch.compile() on MPS for compatibility.")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Cosine warm restarts periodically reset LR to improve exploration
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )

    # Label smoothing reduces overconfidence, often helping generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    patience = 5
    wait = 0

    for epoch in range(10):
        model.train()
        start_time = time.time()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/10")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            # Clip gradients to reduce instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=float(loss.item()))

        # Step the scheduler after each epoch
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
            f"Epoch {epoch + 1}/10 - Time: {epoch_time:.2f}s, "
            f"Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Early stopping and checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), "best_sadvt_pretrainedv6.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # Final evaluation on test set using best checkpoint
    model.load_state_dict(torch.load("best_sadvt_pretrainedv6.pt", map_location=device))
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
