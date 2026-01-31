# Swin-UNet for retinal OCT layer segmentation
# Encoder: Swin-T (window-based self-attention)
# Decoder: U-Net style hierarchical upsampling with skip connections
# Output: per-pixel segmentation map (8 classes)

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import swin_tiny_patch4_window7_224

# ------------------------------------------------------------
# Basic convolutional refinement block
# ------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# ------------------------------------------------------------
# Decoder upsampling block with skip fusion
# ------------------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        # Upsample decoder feature map
        x = self.up(x)

        # Spatial alignment for Swin stage outputs
        if x.size() != skip.size():
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        # Channel-wise concatenation with encoder skip
        return self.conv(torch.cat([x, skip], dim=1))

# ------------------------------------------------------------
# Swin-UNet model definition
# ------------------------------------------------------------
class SwinUnet(nn.Module):
    def __init__(self, img_size=(224, 224), in_chans=1, num_classes=8):
        super().__init__()

        # Swin Transformer encoder
        self.encoder = swin_tiny_patch4_window7_224(pretrained=True)

        # Adapt patch embedding for single-channel OCT input
        self.encoder.patch_embed.proj = nn.Conv2d(
            in_chans, 96, kernel_size=4, stride=4
        )

        # Decoder stages (mirror Swin hierarchy)
        self.decode4 = UpBlock(768, 384, 384)
        self.decode3 = UpBlock(384, 192, 192)
        self.decode2 = UpBlock(192, 96, 96)

        # Final upsampling to full resolution
        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(96, 64, kernel_size=2, stride=2),
            ConvBlock(64, 64),
        )

        # Segmentation head
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Patch embedding (B, C, H/4, W/4)
        x = self.encoder.patch_embed(x)

        # Swin encoder stages (NHWC format)
        x1 = self.encoder.layers[0](x)   # 96 channels
        x2 = self.encoder.layers[1](x1)  # 192 channels
        x3 = self.encoder.layers[2](x2)  # 384 channels
        x4 = self.encoder.layers[3](x3)  # 768 channels

        # Convert NHWC → NCHW for CNN decoder
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        x3 = x3.permute(0, 3, 1, 2)
        x4 = x4.permute(0, 3, 1, 2)

        # Decoder with skip connections
        d4 = self.decode4(x4, x3)
        d3 = self.decode3(d4, x2)
        d2 = self.decode2(d3, x1)
        d1 = self.decode1(d2)

        # Segmentation logits
        out = self.head(d1)

        # Restore original spatial resolution
        out = F.interpolate(
            out, size=(224, 224), mode="bilinear", align_corners=False
        )

        return out
