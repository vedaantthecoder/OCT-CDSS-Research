import os
import csv
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights
from sklearn.metrics import classification_report, accuracy_score
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage import feature, morphology

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
# classes: class folder names and class order used for logits/probabilities
# device: uses Apple Silicon MPS if available, otherwise CPU
# data_dir: expected structure -> OCT-2017/test/{CNV,DME,DRUSEN,NORMAL}/image_files
# csv_out: output file containing per-image predictions + severity labels + probabilities
# model_path: trained SA-DVT checkpoint
# mean/std: normalization constants used during training (must match training)
classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
data_dir = "OCT-2017/test"
csv_out = "sadvt_results.csv"
model_path = "sadvt_best_seed42.pt"
mean, std = 0.38, 0.21

# ------------------------------------------------------------
# Learnable gating module (scalar -> 2 blending weights)
# ------------------------------------------------------------
# Input: speckle_ratio in [0,1]
# Output: softmax weights [w_image, w_speckle] that sum to 1
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
# Robust speckle map construction + gated fusion
# ------------------------------------------------------------
# Produces a single-channel tensor that blends:
# - the original OCT intensity image
# - a binary speckle mask derived from bandpass filtering + normalization
#
# Note: gate weights are initialized randomly unless trained weights are loaded.
class RobustSpeckleMap:
    def __init__(self):
        # Bandpass parameters
        self.low_sigma, self.high_sigma = 1.0, 5.0

        # Local normalization window + percentile threshold
        self.norm_win, self.percentile = 15, 85

        # Additional mask constraints
        self.min_abs_thresh, self.edge_sigma = 0.2, 2
        self.edge_disk_radius = 4

        # Gating network for adaptive blending
        self.gate = LearnableGating().eval()

    def __call__(self, img):
        # Standardize input format
        img = img.convert("L").resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0

        # Bandpass filtering (lowpass - highpass)
        bandpass = gaussian_filter(arr, self.low_sigma) - gaussian_filter(arr, self.high_sigma)

        # Local mean / std normalization
        local_mean = uniform_filter(bandpass, self.norm_win)
        local_std = np.sqrt(uniform_filter((bandpass - local_mean) ** 2, self.norm_win))
        denom = np.where(local_std < 1e-6, 1e-6, local_std)
        normed = (bandpass - local_mean) / denom

        # Speckle candidate mask (percentile + absolute threshold)
        speckle_mask = (normed > np.percentile(normed, self.percentile)) & (normed > self.min_abs_thresh)

        # Edge suppression to reduce false positives along boundaries
        edge = morphology.dilation(
            feature.canny(arr, self.edge_sigma),
            morphology.disk(self.edge_disk_radius)
        )
        speckle_map = speckle_mask * (~edge)

        # Speckle ratio used as gating input
        percent = np.count_nonzero(speckle_map) / speckle_map.size

        # Adaptive blend weights: [w_image, w_speckle]
        w = self.gate(torch.tensor([[percent]], dtype=torch.float32)).detach().numpy()[0]

        # Fuse original intensity and speckle map
        blended = w[0] * arr + w[1] * speckle_map.astype(np.float32)

        # Return tensor shaped [1, H, W] for transforms.Normalize
        return torch.from_numpy(blended).unsqueeze(0).float()

# ------------------------------------------------------------
# SA-DVT backbone (Swin-T) + classification head
# ------------------------------------------------------------
# backbone: Swin-T with patch embed modified for 1-channel input
# head: linear classifier mapping 768-d embedding to 4 logits
# forward returns:
# - logits: for class prediction
# - feat: 768-d embedding (used later for SEPIA-style severity assignment)
class SpeckleAwareSwin(nn.Module):
    def __init__(self):
        super().__init__()

        # Swin-T backbone (torchvision)
        self.backbone = swin_t(weights=Swin_T_Weights.DEFAULT)

        # Replace first conv to accept grayscale (1-channel) OCT
        self.backbone.features[0][0] = nn.Conv2d(1, 96, 4, 4)

        # Remove the built-in Swin classifier head to expose embeddings
        self.backbone.head = nn.Identity()

        # Classification head for 4 OCT disease classes
        self.head = nn.Linear(768, 4)

    def forward(self, x):
        feat = self.backbone(x)       # [B, 768]
        logits = self.head(feat)      # [B, 4]
        return logits, feat

# ------------------------------------------------------------
# Initialize model + transforms
# ------------------------------------------------------------
model = SpeckleAwareSwin().to(device)

# strict=False allows partial loading if checkpoint keys differ
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

# Speckle-aware preprocessing transform pipeline
speckle = RobustSpeckleMap()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    speckle,                             # returns tensor [1,224,224]
    transforms.Normalize([mean], [std])   # normalize single-channel tensor
])

# ------------------------------------------------------------
# Inference: predict class + collect embeddings
# ------------------------------------------------------------
# features_by_class stores embeddings grouped by predicted class
# records stores per-image output for CSV + evaluation
features_by_class = {c: [] for c in classes}
records = []

print("[INFO] Running predictions...")
for cls in classes:
    folder = os.path.join(data_dir, cls)
    for fname in tqdm(os.listdir(folder), desc=f"Processing {cls}"):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue

        path = os.path.join(folder, fname)
        try:
            # Preprocess and batchify: [1,1,224,224]
            img = transform(Image.open(path)).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, feat = model(img)

            # Convert logits -> probabilities
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

            # Predicted label
            pred = classes[np.argmax(probs)]

            # Store embedding under predicted class for SEPIA severity statistics
            feat_np = feat.squeeze().cpu().numpy()
            features_by_class[pred].append(feat_np)

            # Save record: name, predicted, embedding, actual, probs
            records.append([fname, pred, feat_np, cls] + list(probs))
        except:
            # Skips unreadable/corrupt images or runtime errors
            continue

# ------------------------------------------------------------
# Compute per-class embedding centers (predicted-class centroids)
# ------------------------------------------------------------
# centers are used for distance-to-center severity scoring
centers = {
    cls: np.mean(features_by_class[cls], axis=0) if features_by_class[cls] else np.zeros(768)
    for cls in classes
}

# ------------------------------------------------------------
# SEPIA-style severity assignment (quartiles of distance-to-center)
# ------------------------------------------------------------
# For each predicted disease class:
# - compute distance of embedding to class centroid
# - compute quartiles of distances among that predicted class
# - map into {early, mild-low, mild-high, severe}
def assign_severity(cls, feat):
    if cls == "NORMAL":
        return "normal"

    # Distance of this example to its predicted-class centroid
    d = np.linalg.norm(feat - centers[cls])

    # Reference distribution: distances of all predicted-as-cls embeddings
    dists = [np.linalg.norm(f - centers[cls]) for f in features_by_class[cls]]

    # Quartile thresholds
    q1, q2, q3 = np.percentile(dists, [25, 50, 75])

    if d < q1:
        return "early"
    elif d < q2:
        return "mild-low"
    elif d < q3:
        return "mild-high"
    else:
        return "severe"

# ------------------------------------------------------------
# Write per-image outputs to CSV
# ------------------------------------------------------------
with open(csv_out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Image Name",
        "Predicted Class",
        "Predicted Severity",
        "Actual Class",
        "Confidence CNV",
        "Confidence DME",
        "Confidence DRUSEN",
        "Confidence NORMAL"
    ])

    for r in records:
        fname, pred_class, feat, actual_class = r[:4]
        probs = r[4:]

        severity = assign_severity(pred_class, feat)

        writer.writerow([fname, pred_class, severity, actual_class] +
                        [f"{p:.4f}" for p in probs])

# ------------------------------------------------------------
# Classification evaluation (based on folder ground-truth labels)
# ------------------------------------------------------------
y_true = [r[3] for r in records]
y_pred = [r[1] for r in records]

print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=classes))
print(f"\nCSV saved at: {csv_out}")
