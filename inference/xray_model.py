import json

import torch
import torch.nn as nn
from torchvision import models


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        scale = self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class ChestModel(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        base = models.densenet169(weights='IMAGENET1K_V1')
        self.features = base.features
        self.attention = ChannelAttention(1664)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1664, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes))

    def forward(self, x):
        features = self.features(x)
        features = self.attention(features)
        return self.classifier(features)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_xray_model(pth_path, fusion_config_path, device="cpu"):
    with open(fusion_config_path, "r") as f:
        fusion_cfg = json.load(f)
    disease_classes = fusion_cfg["xray_diseases"]

    model = ChestModel(num_classes=len(disease_classes))
    checkpoint = torch.load(pth_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, disease_classes


@torch.no_grad()
def predict(model, disease_classes, image_tensor, device="cpu"):
    model.eval()
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
    return {disease: float(prob) for disease, prob in zip(disease_classes, probs)}


@torch.no_grad()
def predict_raw(model, image_tensor, device="cpu"):
    """Single forward pass returning a numpy array of probabilities, without forcing eval mode.
    Used by MC Dropout uncertainty sampling, which needs Dropout layers left in train mode."""
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    return torch.sigmoid(logits).squeeze(0).cpu().numpy()
