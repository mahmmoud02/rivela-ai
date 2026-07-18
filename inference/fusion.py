import json


def load_fusion_config(path):
    with open(path, "r") as f:
        return json.load(f)


def _status_for(score, status_thresholds):
    if score >= status_thresholds["confirmed"]:
        return "confirmed"
    if score >= status_thresholds["possible"]:
        return "possible"
    return "uncertain"


def fuse(xray_probs, symptom_probs, fusion_cfg):
    diseases = fusion_cfg["diseases"]
    xray_weights = fusion_cfg["xray_weights"]
    symptom_weights = fusion_cfg["symptom_weights"]
    status_thresholds = fusion_cfg["status_thresholds"]

    fused_scores = {}
    for disease in diseases:
        xray_p = xray_probs.get(disease, 0.0)
        symptom_p = symptom_probs.get(disease, 0.0)
        fused_scores[disease] = (
            xray_weights.get(disease, 0.0) * xray_p
            + symptom_weights.get(disease, 0.0) * symptom_p
        )

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    top_disease, top_score = ranked[0]

    xray_top = max(xray_probs.items(), key=lambda item: item[1])[0]
    symptom_top = max(symptom_probs.items(), key=lambda item: item[1])[0]

    return {
        "fused_scores": fused_scores,
        "ranked": ranked,
        "top_disease": top_disease,
        "top_score": top_score,
        "status": _status_for(top_score, status_thresholds),
        "xray_top": xray_top,
        "symptom_top": symptom_top,
        "conflict": xray_top != symptom_top,
    }
