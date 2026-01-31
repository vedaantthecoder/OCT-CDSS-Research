# Extracts SA-DVT embeddings from OCT images, assigns unsupervised severity levels
# via distance-to-centroid, and visualizes class/severity structure using t-SNE.

import os, shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights
from sklearn.manifold import TSNE
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage import feature, morphology
import matplotlib.cm as cm

# ------------------------------------------------------------
# Learnable gating for speckle/image fusion
# ------------------------------------------------------------
class LearnableGating(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)

# ------------------------------------------------------------
# Robust speckle map construction
# ------------------------------------------------------------
class RobustSpeckleMap:
    def __init__(self):
        self.low_sigma, self.high_sigma = 1.0, 5.0
        self.norm_win, self.percentile = 15, 85
        self.min_abs_thresh, self.edge_sigma = 0.2, 2
        self.edge_disk_radius = 4

        self.gate = LearnableGating()
        self.gate.eval()

    def __call__(self, img):
        img = img.convert("L").resize((224, 224))
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Bandpass filtering
        bandpass = (
            gaussian_filter(img_np, self.low_sigma) -
            gaussian_filter(img_np, self.high_sigma)
        )

        # Local normalization
        mu = uniform_filter(bandpass, self.norm_win)
        sigma = np.sqrt(uniform_filter((bandpass - mu) ** 2, self.norm_win))
        sigma = np.clip(sigma, 1e-6, None)
        norm = (bandpass - mu) / sigma

        # Speckle candidate mask
        thresh = np.percentile(norm[~np.isnan(norm)], self.percentile)
        mask = (norm > thresh) & (norm > self.min_abs_thresh)

        # Edge suppression
        edges = feature.canny(img_np, sigma=self.edge_sigma)
        mask &= ~morphology.dilation(edges, morphology.disk(self.edge_disk_radius))

        # Adaptive blending
        speckle_ratio = np.count_nonzero(mask) / mask.size
        w = self.gate(torch.tensor([[speckle_ratio]])).detach().numpy()[0]
        fused = w[0] * img_np + w[1] * mask.astype(np.float32)

        return torch.from_numpy(fused).unsqueeze(0)

# ------------------------------------------------------------
# Speckle-Aware Swin Transformer (embedding extractor)
# ------------------------------------------------------------
class SpeckleAwareSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone.features[0][0] = nn.Conv2d(1, 96, 4, 4)
        self.backbone.head = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

# ------------------------------------------------------------
# Model initialization
# ------------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = SpeckleAwareSwin().to(device)
model.load_state_dict(
    torch.load("sadvt_best_seed42.pt", map_location=device),
    strict=False
)
model.eval()

# ------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------
base_dir = "OCT2017-Copy/test"
output_dir = "SeveritySorted"

mean, std = 0.38, 0.21
classes = ["CNV", "DME", "DRUSEN", "NORMAL"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(1),
    RobustSpeckleMap(),
    transforms.Normalize([mean], [std])
])

features, labels, paths = [], [], []

# ------------------------------------------------------------
# Embedding extraction
# ------------------------------------------------------------
for label in classes:
    folder = os.path.join(base_dir, label)
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                img = Image.open(os.path.join(folder, fname))
                x = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(x).squeeze().cpu().numpy()
                features.append(emb)
                labels.append(label)
                paths.append(os.path.join(folder, fname))
            except:
                pass

features = np.asarray(features)
labels = np.asarray(labels)
paths = np.asarray(paths)

# ------------------------------------------------------------
# Severity assignment via distance-to-centroid
# ------------------------------------------------------------
severity_map = []
cmap = {"CNV": cm.Blues, "DME": cm.Greens, "DRUSEN": cm.Oranges}
alpha = {"early": 0.3, "mild-low": 0.5, "mild-high": 0.7, "severe": 1.0}

for cls in ["CNV", "DME", "DRUSEN"]:
    idx = labels == cls
    feats = features[idx]
    cls_paths = paths[idx]

    center = feats.mean(axis=0)
    dists = np.linalg.norm(feats - center, axis=1)
    q1, q2, q3 = np.percentile(dists, [25, 50, 75])

    for i, d in enumerate(dists):
        sev = (
            "early" if d < q1 else
            "mild-low" if d < q2 else
            "mild-high" if d < q3 else
            "severe"
        )
        dst = os.path.join(output_dir, cls, sev)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(cls_paths[i], os.path.join(dst, os.path.basename(cls_paths[i])))
        severity_map.append(sev)

    # Save distance histogram
    plt.hist(dists, bins=30)
    plt.axvline(q1); plt.axvline(q2); plt.axvline(q3)
    plt.title(f"{cls} Severity Distribution")
    plt.savefig(f"{cls}_severity_histogram.png")
    plt.close()

severity_map += ["normal"] * np.sum(labels == "NORMAL")

# ------------------------------------------------------------
# t-SNE projection
# ------------------------------------------------------------
tsne = TSNE(n_components=2, perplexity=40, random_state=42)
tsne_xy = tsne.fit_transform(features)

# ------------------------------------------------------------
# Metadata export
# ------------------------------------------------------------
pd.DataFrame({
    "Image": [os.path.basename(p) for p in paths],
    "Disease": labels,
    "Severity": severity_map,
    "tSNE_X": tsne_xy[:, 0],
    "tSNE_Y": tsne_xy[:, 1]
}).to_csv("tsne_metadata.csv", index=False)

# ------------------------------------------------------------
# t-SNE visualization
# ------------------------------------------------------------
plt.figure(figsize=(10, 7))

for i, (x, y) in enumerate(tsne_xy):
    cls, sev = labels[i], severity_map[i]
    if cls == "NORMAL":
        plt.scatter(x, y, color="yellow", s=10)
    else:
        plt.scatter(x, y, color=cmap[cls](alpha[sev]), s=10)

plt.legend(handles=[
    Patch(color=cm.Blues(0.3), label="CNV"),
    Patch(color=cm.Greens(0.3), label="DME"),
    Patch(color=cm.Oranges(0.3), label="DRUSEN"),
    Patch(color="yellow", label="NORMAL")
])

plt.title("t-SNE Embedding Space with Unsupervised Severity")
plt.savefig("tsne_with_severity_and_normal.png")
plt.show()
