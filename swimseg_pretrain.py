"""
swimseg_pretrain.py — Pretrain EfficientNetB2 on SwimSeg cloud segmentation
=============================================================================
Why: SwimSeg contains 1,013 Singapore sky images with binary cloud/sky masks.
     Training on cloud segmentation teaches the CNN encoder what clouds look
     like before it sees Himawari satellite images. This improves the encoder's
     spatial understanding and speeds up fusion model convergence.

What gets saved:
     swimseg_encoder.pt  — backbone weights only (loaded by model.py)

Architecture:
     EfficientNetB2 (features_only) → lightweight FPN decoder → binary mask

Usage:
     python swimseg_pretrain.py
     python swimseg_pretrain.py --swimseg-dir ./swimseg --epochs 50 --batch-size 16
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMG_SIZE = 224


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SwimSegDataset(Dataset):
    """
    Loads SwimSeg sky images and binary cloud segmentation masks.

    Split strategy: by date (not random) to avoid leakage — images from the
    same shooting session are kept together in either train or val.
    """

    def __init__(self, records: pd.DataFrame, swimseg_dir: str, augment: bool = False):
        self.records     = records.reset_index(drop=True)
        self.img_dir     = os.path.join(swimseg_dir, "images")
        self.gt_dir      = os.path.join(swimseg_dir, "GTmaps")
        self.augment     = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row   = self.records.iloc[idx]
        num   = int(row["Number"])
        fname = f"{num:04d}.png"

        # Load image
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1)  # (3, H, W)

        # Load mask — binary: cloud=1, sky=0
        mask = Image.open(os.path.join(self.gt_dir, f"{num:04d}_GT.png")).convert("L")
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        mask_t = torch.from_numpy((np.array(mask) > 128).astype(np.float32))  # (H, W)

        # Augmentations (training only)
        if self.augment:
            # Horizontal flip
            if torch.rand(1) > 0.5:
                img_t  = torch.flip(img_t,  dims=[2])
                mask_t = torch.flip(mask_t, dims=[1])

            # Vertical flip
            if torch.rand(1) > 0.5:
                img_t  = torch.flip(img_t,  dims=[1])
                mask_t = torch.flip(mask_t, dims=[0])

            # Color jitter (image only, not mask)
            if torch.rand(1) > 0.5:
                brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
                img_t = torch.clamp(img_t * brightness, 0, 1)

            if torch.rand(1) > 0.5:
                contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
                mean = img_t.mean(dim=[1, 2], keepdim=True)
                img_t = torch.clamp(mean + contrast * (img_t - mean), 0, 1)

        # Normalize with ImageNet stats (same as fusion model)
        img_t = (img_t - IMG_MEAN) / IMG_STD

        # Auxiliary target: cloud coverage ratio (for multi-task learning)
        cloud_ratio = mask_t.mean()

        return img_t, mask_t.unsqueeze(0), cloud_ratio  # (3,H,W), (1,H,W), scalar


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class FPNDecoder(nn.Module):
    """
    Lightweight Feature Pyramid Network decoder.
    Takes all 5 EfficientNetB2 feature maps, fuses them, outputs a
    full-resolution binary segmentation map.

    Feature map channels (EfficientNetB2, features_only):
        stage0:  24 ch,  112×112
        stage1:  48 ch,   56×56
        stage2: 120 ch,   28×28
        stage3: 208 ch,   14×14
        stage4: 352 ch,    7×7
    """
    STAGE_CHANNELS = [24, 48, 120, 208, 352]
    HIDDEN         = 128

    def __init__(self):
        super().__init__()
        # Lateral 1×1 convs to unify channels
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, self.HIDDEN, 1) for c in self.STAGE_CHANNELS
        ])
        # Top-down smooth convs
        self.smooths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.HIDDEN, self.HIDDEN, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.HIDDEN),
                nn.ReLU(inplace=True),
            )
            for _ in self.STAGE_CHANNELS
        ])
        # Final segmentation head: upsample to full resolution → 1 channel
        self.head = nn.Sequential(
            nn.Conv2d(self.HIDDEN, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        # Auxiliary regression head: cloud coverage ratio
        self.aux_pool = nn.AdaptiveAvgPool2d(1)
        self.aux_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(352, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        """
        features: list of 5 tensors (stage0 … stage4), stage4 is deepest.
        """
        # Top-down FPN pass
        out = self.laterals[4](features[4])
        for i in range(3, -1, -1):
            out = F.interpolate(out, size=features[i].shape[-2:], mode="nearest")
            out = out + self.laterals[i](features[i])
            out = self.smooths[i](out)

        # Upsample to IMG_SIZE × IMG_SIZE
        out = F.interpolate(out, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                            align_corners=False)
        seg_logit = self.head(out)  # (B, 1, H, W)

        # Auxiliary: cloud ratio from deepest feature map
        cloud_ratio = self.aux_head(self.aux_pool(features[4]))  # (B, 1)

        return seg_logit, cloud_ratio


class SwimSegModel(nn.Module):
    """EfficientNetB2 encoder + FPN decoder for cloud segmentation."""

    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "efficientnet_b2", pretrained=True, features_only=True
        )
        self.decoder = FPNDecoder()

    def forward(self, x):
        features = self.encoder(x)   # list of 5 feature maps
        seg_logit, cloud_ratio = self.decoder(features)
        return seg_logit, cloud_ratio


# ══════════════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════════════

class DiceBCELoss(nn.Module):
    """Combined BCE + Dice loss for binary segmentation."""

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.bce         = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        B = targets.shape[0]
        probs_f   = probs.view(B, -1)
        targets_f = targets.view(B, -1)
        intersection = (probs_f * targets_f).sum(1)
        dice_loss = 1 - (2 * intersection + 1) / (probs_f.sum(1) + targets_f.sum(1) + 1)
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def iou_score(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    B = preds.shape[0]
    p = preds.view(B, -1);  t = targets.view(B, -1)
    inter = (p * t).sum(1);  union = (p + t - p * t).sum(1)
    return (inter / (union + 1e-6)).mean().item()


def make_dataloaders(swimseg_dir, batch_size, val_frac=0.2):
    """Split by date — keep entire shooting sessions in one split."""
    meta = pd.read_csv(os.path.join(swimseg_dir, "metadata.csv"))
    dates = sorted(meta["Date"].unique())

    n_val   = max(1, int(len(dates) * val_frac))
    val_dates  = set(dates[-n_val:])   # hold out most recent dates as val
    train_dates = set(dates[:-n_val])

    train_rec = meta[meta["Date"].isin(train_dates)]
    val_rec   = meta[meta["Date"].isin(val_dates)]

    print(f"  Train: {len(train_rec):,} images  ({len(train_dates)} dates)")
    print(f"  Val  : {len(val_rec):,} images  ({len(val_dates)} dates)")

    train_ds = SwimSegDataset(train_rec, swimseg_dir, augment=True)
    val_ds   = SwimSegDataset(val_rec,   swimseg_dir, augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)
    return train_dl, val_dl


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(swimseg_dir="./swimseg", epochs=50, batch_size=16, lr=3e-4,
          output_path="./swimseg_encoder.pt"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  SwimSeg Cloud Segmentation Pretraining")
    print(f"{'='*65}")
    print(f"  SwimSeg dir : {os.path.abspath(swimseg_dir)}")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {epochs}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Output      : {output_path}")
    print(f"{'='*65}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Building dataloaders (split by date)...")
    train_dl, val_dl = make_dataloaders(swimseg_dir, batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = SwimSegModel().to(device)
    seg_loss  = DiceBCELoss(bce_weight=0.5)
    aux_loss  = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_iou = 0.0

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val IoU':>8}  {'LR':>10}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for imgs, masks, cloud_ratios in train_dl:
            imgs         = imgs.to(device)
            masks        = masks.to(device)
            cloud_ratios = cloud_ratios.float().unsqueeze(1).to(device)

            seg_logit, pred_ratio = model(imgs)

            loss = seg_loss(seg_logit, masks) + 0.2 * aux_loss(pred_ratio, cloud_ratios)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_losses, val_ious = [], []
        with torch.no_grad():
            for imgs, masks, cloud_ratios in val_dl:
                imgs         = imgs.to(device)
                masks        = masks.to(device)
                cloud_ratios = cloud_ratios.float().unsqueeze(1).to(device)

                seg_logit, pred_ratio = model(imgs)
                loss = seg_loss(seg_logit, masks) + 0.2 * aux_loss(pred_ratio, cloud_ratios)

                val_losses.append(loss.item())
                val_ious.append(iou_score(seg_logit, masks))

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        val_iou    = np.mean(val_ious)
        cur_lr     = scheduler.get_last_lr()[0]

        print(f"{epoch:>6}  {train_loss:>11.4f}  {val_loss:>9.4f}  {val_iou:>8.4f}  {cur_lr:>10.2e}")

        # ── Save best encoder ─────────────────────────────────────────────
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.encoder.state_dict(), output_path)
            print(f"         ✓ Best encoder saved (IoU={val_iou:.4f})")

    print(f"\n{'='*65}")
    print(f"  Pretraining complete!")
    print(f"  Best validation IoU : {best_val_iou:.4f}")
    print(f"  Encoder weights     : {os.path.abspath(output_path)}")
    print(f"\n  Next step: pass --pretrained-encoder {output_path}")
    print(f"  to your fusion model training script.")
    print(f"{'='*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain EfficientNetB2 on SwimSeg cloud segmentation"
    )
    parser.add_argument("--swimseg-dir",  default="./swimseg",
                        help="Path to swimseg folder  (default: ./swimseg)")
    parser.add_argument("--epochs",       type=int, default=50,
                        help="Training epochs  (default: 50)")
    parser.add_argument("--batch-size",   type=int, default=16,
                        help="Batch size  (default: 16)")
    parser.add_argument("--lr",           type=float, default=3e-4,
                        help="Learning rate  (default: 3e-4)")
    parser.add_argument("--output",       default="./swimseg_encoder.pt",
                        help="Where to save encoder weights  (default: ./swimseg_encoder.pt)")
    args = parser.parse_args()

    train(
        swimseg_dir  = args.swimseg_dir,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        output_path  = args.output,
    )
