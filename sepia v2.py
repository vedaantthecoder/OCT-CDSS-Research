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
# Extra imports for SEPIA evaluation
# ------------------------------------------------------------
# spearmanr / kendalltau: rank correlations for monotonic severity validation
# defaultdict: bin counting
# warnings: suppress noisy numerical warnings during correlation tests
from scipy.stats import spearmanr, kendalltau
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
# classes: class folder names and class index order for probabilities
# device: uses MPS if available, otherwise CPU
# data_dir: expected structure -> OCT2017/test/{CNV,DME,DRUSEN,NORMAL}/...
# csv_out: per-image predictions + severity + deviation outputs
# sepia_summary_csv: analysis-friendly CSV for SEPIA evaluation results
# biomarkers_csv: optional image-level biomarkers (example: crt_um)
# model_path: trained SA-DVT checkpoint
# mean/std: normalization constants (must match training)
classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
data_dir = "OCT2017/test"
csv_out = "sadvt_results.csv"
sepia_summary_csv = "sepia_summary.csv"
biomarkers_csv = "biomarkers.csv"
model_path = "sadvt_best_seed42.pt"
mean, std = 0.38, 0.21

# ------------------------------------------------------------
# Learnable gating module (scalar -> 2 weights)
# ------------------------------------------------------------
# Input: speckle_ratio in [0,1]
# Output: [w_image, w_speckle] softmax weights
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
# Speckle map generator + gated fusion
# ------------------------------------------------------------
# Steps:
# 1) bandpass filter to emphasize speckle-like texture
# 2) local normalization to stabilize contrast
# 3) percentile threshold to produce a speckle candidate mask
# 4) edge suppression to avoid anatomical boundaries
# 5) gated blend of original image and speckle mask
#
# Note: if gating weights are not loaded from training, they are random.
class RobustSpeckleMap:
    def __init__(self):
        self.low_sigma, self.high_sigma = 1.0, 5.0
        self.norm_win, self.percentile = 15, 85
        self.min_abs_thresh, self.edge_sigma = 0.2, 2
        self.edge_disk_radius = 4
        self.gate = LearnableGating().eval()

    def __call__(self, img):
        img = img.convert("L").resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0

        # Bandpass filtering
        bandpass = gaussian_filter(arr, self.low_sigma) - gaussian_filter(arr, self.high_sigma)

        # Local normalization
        local_mean = uniform_filter(bandpass, self.norm_win)
        local_std = np.sqrt(uniform_filter((bandpass - local_mean) ** 2, self.norm_win))
        denom = np.where(local_std < 1e-6, 1e-6, local_std)
        normed = (bandpass - local_mean) / denom

        # Speckle candidate mask
        speckle_mask = (normed > np.percentile(normed, self.percentile)) & (normed > self.min_abs_thresh)

        # Edge suppression
        edge = morphology.dilation(
            feature.canny(arr, self.edge_sigma),
            morphology.disk(self.edge_disk_radius)
        )
        speckle_map = speckle_mask * (~edge)

        # Speckle density drives adaptive fusion
        percent = np.count_nonzero(speckle_map) / speckle_map.size
        w = self.gate(torch.tensor([[percent]], dtype=torch.float32)).detach().numpy()[0]

        # Fused input channel
        blended = w[0] * arr + w[1] * speckle_map.astype(np.float32)
        return torch.from_numpy(blended).unsqueeze(0).float()

# ------------------------------------------------------------
# SA-DVT model wrapper (Swin-T backbone + linear classifier)
# ------------------------------------------------------------
# backbone: Swin-T with patch embed modified for 1-channel OCT
# head: maps 768-d embedding -> 4 class logits
# forward returns (logits, embedding) for classification + SEPIA scoring
class SpeckleAwareSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone.features[0][0] = nn.Conv2d(1, 96, 4, 4)
        self.backbone.head = nn.Identity()
        self.head = nn.Linear(768, 4)

    def forward(self, x):
        emb = self.backbone(x)     # [B, 768]
        logits = self.head(emb)    # [B, 4]
        return logits, emb

# ------------------------------------------------------------
# Initialize model and preprocessing pipeline
# ------------------------------------------------------------
model = SpeckleAwareSwin().to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

speckle = RobustSpeckleMap()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    speckle,                           # returns tensor [1,224,224]
    transforms.Normalize([mean], [std])
])

# ------------------------------------------------------------
# Inference: predict class and store embeddings
# ------------------------------------------------------------
# features_by_class: embeddings grouped by predicted class (used for centroids)
# records: rows used for CSV export and classification metrics
features_by_class = {c: [] for c in classes}
records = []  # [fname, pred_class, feat_vec(np.ndarray), actual_class, p_cnv, p_dme, p_drusen, p_normal]

print("[INFO] Running predictions...")
for cls in classes:
    folder = os.path.join(data_dir, cls)
    for fname in tqdm(os.listdir(folder), desc=f"Processing {cls}"):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue

        path = os.path.join(folder, fname)
        try:
            img = transform(Image.open(path)).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, feat = model(img)

            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred = classes[int(np.argmax(probs))]

            feat_np = feat.squeeze().cpu().numpy()
            features_by_class[pred].append(feat_np)
            records.append([fname, pred, feat_np, cls] + list(probs))
        except Exception:
            continue

# ------------------------------------------------------------
# Compute embedding centers and NORMAL reference center
# ------------------------------------------------------------
# centers: centroid per predicted class
# normal_center: used as reference for SEPIA distance-to-normal scoring
centers = {
    cls: np.mean(features_by_class[cls], axis=0) if features_by_class[cls] else np.zeros(768)
    for cls in classes
}
normal_center = centers["NORMAL"]

# ------------------------------------------------------------
# Build a disease distance distribution from NORMAL center
# ------------------------------------------------------------
# all_disease_dists: distances of predicted disease embeddings to the NORMAL center
# mean_normal_dist: scale factor used for percent deviation
# q1/q2/q3: quartile cutoffs for severity binning
all_disease_dists = []
for cls in classes:
    if cls == "NORMAL":
        continue
    for f in features_by_class[cls]:
        all_disease_dists.append(np.linalg.norm(f - normal_center))

mean_normal_dist = np.mean(all_disease_dists) if all_disease_dists else 1e-6
q1, q2, q3 = np.percentile(all_disease_dists, [25, 50, 75]) if len(all_disease_dists) > 0 else (0.25, 0.5, 0.75)

# ------------------------------------------------------------
# SEPIA scoring utilities
# ------------------------------------------------------------
# SEV_TO_NUM: ordinal mapping for rank correlation tests
SEV_ORDER = ["early", "mild-low", "mild-high", "severe"]
SEV_TO_NUM = {"normal": 0, "early": 1, "mild-low": 2, "mild-high": 3, "severe": 4}

def assign_severity_and_deviation_from_normal(cls, feat):
    # Distance from NORMAL center is the SEPIA severity proxy
    d = np.linalg.norm(feat - normal_center)

    if cls == "NORMAL":
        return "normal", 0.0, d

    # Percent deviation relative to mean disease distance-to-normal
    percent_deviation = (d / mean_normal_dist) * 100 if mean_normal_dist > 0 else 0.0

    # Quartile-based severity bins
    if d < q1:
        sev = "early"
    elif d < q2:
        sev = "mild-low"
    elif d < q3:
        sev = "mild-high"
    else:
        sev = "severe"

    return sev, percent_deviation, d

# ------------------------------------------------------------
# Write prediction CSV and collect SEPIA evaluation rows
# ------------------------------------------------------------
sepia_rows = []
with open(csv_out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Image Name",
        "Predicted Class",
        "Predicted Severity",
        "Percent Deviation from Normal",
        "Actual Class",
        "Confidence CNV",
        "Confidence DME",
        "Confidence DRUSEN",
        "Confidence NORMAL"
    ])

    for r in records:
        fname, pred_class, feat, actual_class = r[:4]
        probs = np.array(r[4:], dtype=float)

        severity, deviation, dist = assign_severity_and_deviation_from_normal(pred_class, feat)

        writer.writerow([
            fname,
            pred_class,
            severity,
            f"{deviation:.2f}",
            actual_class,
            f"{probs[classes.index('CNV')]:.4f}",
            f"{probs[classes.index('DME')]:.4f}",
            f"{probs[classes.index('DRUSEN')]:.4f}",
            f"{probs[classes.index('NORMAL')]:.4f}"
        ])

        # Row format used for SEPIA evaluation and correlation checks
        sepia_rows.append({
            "image": fname,
            "pred_class": pred_class,
            "actual_class": actual_class,
            "severity": severity,
            "severity_num": SEV_TO_NUM[severity],
            "distance": float(dist),
            "percent_dev": float(deviation),
            "p_cnv": float(probs[classes.index("CNV")]),
            "p_dme": float(probs[classes.index("DME")]),
            "p_drusen": float(probs[classes.index("DRUSEN")]),
            "p_normal": float(probs[classes.index("NORMAL")])
        })

# ------------------------------------------------------------
# SA-DVT classification metrics (from folder labels)
# ------------------------------------------------------------
y_true = [r[3] for r in records]
y_pred = [r[1] for r in records]
print("\n[SA-DVT] Accuracy:", accuracy_score(y_true, y_pred))
print("\n[SA-DVT] Classification Report:\n", classification_report(y_true, y_pred, target_names=classes))
print(f"\n[INFO] Predictions CSV saved at: {csv_out}")

# ============================================================
# SEPIA evaluation
# ============================================================
print("\n[SEPIA] Starting evaluation of unsupervised severity scoring...")

# ------------------------------------------------------------
# Save analysis-friendly SEPIA CSV
# ------------------------------------------------------------
with open(sepia_summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "image", "pred_class", "actual_class", "severity", "severity_num", "distance",
        "percent_dev", "p_cnv", "p_dme", "p_drusen", "p_normal"
    ])
    for row in sepia_rows:
        w.writerow([row[k] for k in [
            "image", "pred_class", "actual_class", "severity", "severity_num",
            "distance", "percent_dev", "p_cnv", "p_dme", "p_drusen", "p_normal"
        ]])

print(f"[SEPIA] Summary CSV saved at: {sepia_summary_csv}")

# ------------------------------------------------------------
# 1) Monotonicity sanity check: distance should increase with severity bins
# ------------------------------------------------------------
bin_stats = []
for sev in SEV_ORDER:
    dists = [r["distance"] for r in sepia_rows if r["severity"] == sev]
    if len(dists) > 0:
        bin_stats.append((sev, len(dists), np.median(dists), np.mean(dists)))
    else:
        bin_stats.append((sev, 0, float("nan"), float("nan")))

print("\n[SEPIA] Distance by severity bin (count, median, mean):")
for sev, n, med, mean_ in bin_stats:
    print(f"  {sev:9s}  n={n:4d}  median={med:.4f}  mean={mean_:.4f}")

# ------------------------------------------------------------
# 2) Rank correlations between severity and confidence signals
# ------------------------------------------------------------
# disease_only filters out NORMAL predictions to avoid trivial correlations
disease_only = [r for r in sepia_rows if r["pred_class"] != "NORMAL"]

if len(disease_only) >= 3:
    sev_nums = np.array([r["severity_num"] for r in disease_only], dtype=float)

    p_normal = np.array([r["p_normal"] for r in disease_only], dtype=float)
    one_minus_p_normal = 1.0 - p_normal

    # Confidence for the predicted class
    p_pred = np.array([
        r["p_cnv"] if r["pred_class"] == "CNV" else
        r["p_dme"] if r["pred_class"] == "DME" else
        r["p_drusen"] if r["pred_class"] == "DRUSEN" else
        r["p_normal"]
        for r in disease_only
    ], dtype=float)

    # Severity vs (1 - p_normal): expected positive if severity aligns with "less normal"
    rho1, p1 = spearmanr(sev_nums, one_minus_p_normal)
    tau1, pt1 = kendalltau(sev_nums, one_minus_p_normal, variant="b")
    print(f"\n[SEPIA] Spearman(severity, 1 - p_NORMAL) = {rho1:.3f} (p={p1:.2e})")
    print(f"[SEPIA] Kendall τ_b(severity, 1 - p_NORMAL) = {tau1:.3f} (p={pt1:.2e})")

    # Severity vs predicted-class confidence: expected positive if confidence tracks severity
    rho2, p2 = spearmanr(sev_nums, p_pred)
    tau2, pt2 = kendalltau(sev_nums, p_pred, variant="b")
    print(f"[SEPIA] Spearman(severity, p_predclass) = {rho2:.3f} (p={p2:.2e})")
    print(f"[SEPIA] Kendall τ_b(severity, p_predclass) = {tau2:.3f} (p={pt2:.2e})")
else:
    print("[SEPIA] Not enough disease-predicted cases to compute correlations.")

# ------------------------------------------------------------
# 3) Severity distribution for ground-truth NORMAL false positives
# ------------------------------------------------------------
# Measures whether misclassified NORMAL cases cluster as "early" or skew severe.
normal_fp_bins = defaultdict(int)
normal_fp_total = 0

for r in sepia_rows:
    if r["actual_class"] == "NORMAL" and r["pred_class"] != "NORMAL":
        normal_fp_bins[r["severity"]] += 1
        normal_fp_total += 1

if normal_fp_total > 0:
    print("\n[SEPIA] Severity distribution for actual NORMAL misclassified as disease:")
    for sev in SEV_ORDER:
        print(f"  {sev:9s}: {normal_fp_bins[sev]:4d}")
else:
    print("\n[SEPIA] No actual NORMAL were misclassified as disease.")

# ------------------------------------------------------------
# 4) Optional biomarker correlation (example: central retinal thickness)
# ------------------------------------------------------------
# Expected biomarkers CSV format example:
# image, crt_um, has_intraretinal_fluid, has_subretinal_fluid
if os.path.exists(biomarkers_csv):
    try:
        import pandas as pd
        bio = pd.read_csv(biomarkers_csv)

        # Map model outputs by image filename for joining
        by_name = {r["image"]: r for r in sepia_rows}

        sev_list, crt_list = [], []
        for _, row in bio.iterrows():
            name = str(row.get("image"))
            if name in by_name and "crt_um" in row:
                sev_list.append(by_name[name]["severity_num"])
                crt_list.append(float(row["crt_um"]))

        if len(sev_list) >= 3:
            rho, p = spearmanr(sev_list, crt_list)
            tau, pt = kendalltau(sev_list, crt_list, variant="b")
            print(f"\n[SEPIA] Correlation with CRT (µm): Spearman={rho:.3f} (p={p:.2e}), Kendall τ_b={tau:.3f} (p={pt:.2e})")
        else:
            print("\n[SEPIA] Biomarker file found but not enough matched rows for correlation.")
    except Exception as e:
        print(f"\n[SEPIA] Biomarker correlation skipped due to error: {e}")

print("\n[SEPIA] Evaluation complete.")
