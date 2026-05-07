# model.py
from transformers import ViTModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        mask = (target.sum(dim=(1, 2, 3)) > 0)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred = pred[mask]
        target = target[mask]
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)


class XrayViT(nn.Module):
    def __init__(self):
        super().__init__()

        # ViT-Base/16
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

        # Classification Head (768)
        self.cls_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

        # Segmentation Head (768)
        self.seg_head = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, pixel_values):
        out = self.vit(pixel_values=pixel_values)

        # Classification
        cls_token = out.last_hidden_state[:, 0, :]
        cls_prob = self.cls_head(cls_token)              # [B, 1]

        # Segmentation
        patch_tokens = out.last_hidden_state[:, 1:, :]  # [B, 196, 768]
        B = patch_tokens.shape[0]
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, 768, 14, 14)
        seg_map = self.seg_head(patch_tokens)
        seg_map = F.interpolate(seg_map, size=(224, 224), mode='bilinear', align_corners=False)

        return cls_prob, seg_map