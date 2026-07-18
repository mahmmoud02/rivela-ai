import json

import torch
import torch.nn as nn


class SymptomModel(nn.Module):
    def __init__(self, n_symptoms, n_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_symptoms, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes))

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_symptom_model(pth_path, fusion_config_path, device="cpu"):
    with open(fusion_config_path, "r") as f:
        fusion_cfg = json.load(f)
    symptom_columns = fusion_cfg["symptoms"]
    disease_classes = fusion_cfg["symptom_diseases"]

    model = SymptomModel(n_symptoms=len(symptom_columns), n_classes=len(disease_classes))
    checkpoint = torch.load(pth_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, symptom_columns, disease_classes


def _build_input(symptom_columns, symptom_dict, device="cpu"):
    values = [float(symptom_dict.get(col, 0)) for col in symptom_columns]
    return torch.tensor([values], dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model, symptom_columns, disease_classes, symptom_dict, device="cpu"):
    model.eval()
    x = _build_input(symptom_columns, symptom_dict, device)
    probs = model.predict_proba(x).squeeze(0).cpu().tolist()
    return {disease: float(prob) for disease, prob in zip(disease_classes, probs)}


@torch.no_grad()
def predict_raw(model, x_tensor):
    """Single forward pass returning a numpy array of probabilities, without forcing eval mode.
    Used by MC Dropout uncertainty sampling, which needs Dropout layers left in train mode."""
    return model.predict_proba(x_tensor).squeeze(0).cpu().numpy()
