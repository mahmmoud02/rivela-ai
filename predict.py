import json
import os

import torch
from PIL import Image

# Use all available CPU cores for PyTorch ops
torch.set_num_threads(os.cpu_count() or 4)

from inference.fusion import fuse, load_fusion_config
from inference.gradcam import GradCAM, get_target_layer, overlay_heatmap
from inference.preprocessing import preprocess_image
from inference.symptom_model import _build_input as build_symptom_input
from inference.symptom_model import load_symptom_model
from inference.symptom_model import predict as symptom_predict
from inference.symptom_model import predict_raw as symptom_predict_raw
from inference.uncertainty import mc_dropout_uncertainty
from inference.xray_model import load_xray_model
from inference.xray_model import predict as xray_predict
from inference.xray_model import predict_raw as xray_predict_raw

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "model_weights")
XRAY_WEIGHTS = os.path.join(WEIGHTS_DIR, "best_model_v4.pth")
SYMPTOM_WEIGHTS = os.path.join(WEIGHTS_DIR, "best_symptom_model.pth")
FUSION_CONFIG = os.path.join(WEIGHTS_DIR, "fusion_config.json")

_xray_model = None
_xray_diseases = None
_symptom_model = None
_symptom_columns = None
_symptom_diseases = None
_fusion_cfg = None
_gradcam = None


def get_models():
    global _xray_model, _xray_diseases, _symptom_model, _symptom_columns
    global _symptom_diseases, _fusion_cfg, _gradcam
    if _xray_model is None:
        _fusion_cfg = load_fusion_config(FUSION_CONFIG)
        _xray_model, _xray_diseases = load_xray_model(XRAY_WEIGHTS, FUSION_CONFIG)
        _symptom_model, _symptom_columns, _symptom_diseases = load_symptom_model(
            SYMPTOM_WEIGHTS, FUSION_CONFIG
        )
        _gradcam = GradCAM(_xray_model, get_target_layer(_xray_model))
    return _xray_model, _xray_diseases, _symptom_model, _symptom_columns, _symptom_diseases, _fusion_cfg


def get_symptom_list():
    fusion_cfg = load_fusion_config(FUSION_CONFIG)
    return fusion_cfg["symptoms"]


def run_prediction(modality, image_path=None, symptom_dict=None):
    """
    modality: "xray", "symptoms", or "both".
    Returns a dict with whichever of xray_probs / symptom_probs were computed,
    plus a fusion_result only when modality == "both".
    """
    xray_model, xray_diseases, symptom_model, symptom_columns, symptom_diseases, fusion_cfg = get_models()

    result = {
        "xray_probs": None,
        "symptom_probs": None,
        "fusion_result": None,
        "xray_uncertainty": None,
        "symptom_uncertainty": None,
    }

    if modality in ("xray", "both"):
        tensor = preprocess_image(image_path)
        # Run MC dropout with 10 samples — reuse the mean as the prediction
        # (avoids a redundant separate forward pass; 10 samples is sufficient for reliable uncertainty)
        result["xray_uncertainty"] = mc_dropout_uncertainty(
            xray_model, tensor,
            lambda t: torch.sigmoid(xray_model(t)).cpu().numpy(),
            xray_diseases,
            n_samples=10,
        )
        result["xray_probs"] = {d: v["mean"] for d, v in result["xray_uncertainty"].items()}

    if modality in ("symptoms", "both"):
        symptom_tensor = build_symptom_input(symptom_columns, symptom_dict or {})
        result["symptom_uncertainty"] = mc_dropout_uncertainty(
            symptom_model, symptom_tensor,
            lambda t: torch.softmax(symptom_model(t), dim=1).cpu().numpy(),
            symptom_diseases,
            n_samples=10,
        )
        result["symptom_probs"] = {d: v["mean"] for d, v in result["symptom_uncertainty"].items()}

    if modality == "both":
        result["fusion_result"] = fuse(result["xray_probs"], result["symptom_probs"], fusion_cfg)

    return result


def run_gradcam(image_path, target_disease):
    xray_model, xray_diseases, *_ = get_models()
    tensor = preprocess_image(image_path)
    target_idx = xray_diseases.index(target_disease)
    cam = _gradcam.generate(tensor, target_idx)

    original_image = Image.open(image_path).convert("RGB").resize((512, 512))
    heatmap_img, overlay_img = overlay_heatmap(original_image, cam)
    return original_image, heatmap_img, overlay_img


def compute_symptom_contributions(symptom_dict, top_disease, top_n=5):
    """Occlusion-based feature attribution: for each checked symptom, rerun the
    symptom model with that symptom turned off and measure how much the
    top disease's probability shifts. This is a lightweight approximation of
    SHAP-style attribution, not the actual shap library."""
    _, _, symptom_model, symptom_columns, symptom_diseases, _ = get_models()
    if not symptom_dict or top_disease not in symptom_diseases:
        return []

    checked = [name for name, value in symptom_dict.items() if value]
    if not checked:
        return []

    baseline = symptom_predict(symptom_model, symptom_columns, symptom_diseases, symptom_dict)[top_disease]

    contributions = []
    for symptom in checked:
        without_symptom = dict(symptom_dict)
        without_symptom[symptom] = 0
        without_prob = symptom_predict(symptom_model, symptom_columns, symptom_diseases, without_symptom)[
            top_disease
        ]
        contributions.append((symptom, baseline - without_prob))

    contributions.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return contributions[:top_n]


def serialize_result(result):
    row = {
        "xray_probs_json": json.dumps(result["xray_probs"]) if result["xray_probs"] else None,
        "symptom_probs_json": json.dumps(result["symptom_probs"]) if result["symptom_probs"] else None,
        "fused_result_json": json.dumps(result["fusion_result"]) if result["fusion_result"] else None,
        "xray_uncertainty_json": json.dumps(result["xray_uncertainty"]) if result["xray_uncertainty"] else None,
        "symptom_uncertainty_json": (
            json.dumps(result["symptom_uncertainty"]) if result["symptom_uncertainty"] else None
        ),
        "top_disease": None,
        "status": None,
        "conflict": 0,
    }
    if result["fusion_result"]:
        row["top_disease"] = result["fusion_result"]["top_disease"]
        row["status"] = result["fusion_result"]["status"]
        row["conflict"] = int(result["fusion_result"]["conflict"])
    elif result["xray_probs"]:
        row["top_disease"] = max(result["xray_probs"].items(), key=lambda kv: kv[1])[0]
    elif result["symptom_probs"]:
        row["top_disease"] = max(result["symptom_probs"].items(), key=lambda kv: kv[1])[0]
    return row
