"""
Smoke test for the real trained ChestAI artifacts in model_weights/.
Run with: python test_real_models.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from inference.xray_model import load_xray_model, predict as xray_predict
from inference.symptom_model import load_symptom_model, predict as symptom_predict
from inference.preprocessing import preprocess_image
from inference.fusion import load_fusion_config, fuse

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "model_weights")
XRAY_WEIGHTS = os.path.join(WEIGHTS_DIR, "best_model_v4.pth")
SYMPTOM_WEIGHTS = os.path.join(WEIGHTS_DIR, "best_symptom_model.pth")
FUSION_CONFIG = os.path.join(WEIGHTS_DIR, "fusion_config.json")


def fail(stage, exc):
    print(f"\n[FAILED] {stage}")
    print(f"  {type(exc).__name__}: {exc}")
    sys.exit(1)


def main():
    print("1. Loading fusion config...")
    try:
        fusion_cfg = load_fusion_config(FUSION_CONFIG)
        print(f"   OK - {len(fusion_cfg['diseases'])} diseases, {len(fusion_cfg['symptoms'])} symptoms")
    except Exception as exc:
        fail("loading fusion_config.json", exc)

    print("2. Loading X-ray model (best_model_v4.pth)...")
    try:
        xray_model, xray_diseases = load_xray_model(XRAY_WEIGHTS, FUSION_CONFIG)
        print(f"   OK - {len(xray_diseases)} disease classes")
    except Exception as exc:
        fail(
            "loading best_model_v4.pth into ChestModel "
            "(if this is a state_dict mismatch, the architecture in "
            "inference/xray_model.py may not exactly match the trained checkpoint)",
            exc,
        )

    print("3. Loading symptom model (best_symptom_model.pth)...")
    try:
        symptom_model, symptom_columns, symptom_diseases = load_symptom_model(
            SYMPTOM_WEIGHTS, FUSION_CONFIG
        )
        print(f"   OK - {len(symptom_columns)} symptom inputs, {len(symptom_diseases)} disease classes")
    except Exception as exc:
        fail(
            "loading best_symptom_model.pth into SymptomModel "
            "(if this is a state_dict mismatch, check n_symptoms/n_classes "
            "against the real training notebook)",
            exc,
        )

    print("4. Running X-ray prediction on a synthetic test image...")
    try:
        fake_image = Image.fromarray(
            np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        )
        tensor = preprocess_image(fake_image)
        xray_probs = xray_predict(xray_model, xray_diseases, tensor)
        top = max(xray_probs.items(), key=lambda kv: kv[1])
        print(f"   OK - top prediction: {top[0]} ({top[1]:.3f})")
    except Exception as exc:
        fail("running inference through the X-ray model", exc)

    print("5. Running symptom prediction on a synthetic symptom set...")
    try:
        sample_symptoms = {"fever": 1, "dry_cough": 1, "shortness_of_breath": 1}
        symptom_probs = symptom_predict(symptom_model, symptom_columns, symptom_diseases, sample_symptoms)
        top = max(symptom_probs.items(), key=lambda kv: kv[1])
        print(f"   OK - top prediction: {top[0]} ({top[1]:.3f})")
    except Exception as exc:
        fail("running inference through the symptom model", exc)

    print("6. Fusing X-ray + symptom predictions...")
    try:
        result = fuse(xray_probs, symptom_probs, fusion_cfg)
        print(f"   OK - top_disease={result['top_disease']} "
              f"score={result['top_score']:.3f} status={result['status']} "
              f"conflict={result['conflict']}")
    except Exception as exc:
        fail("fusing predictions", exc)

    print("\nAll stages passed. Real model files load and run correctly.")


if __name__ == "__main__":
    main()
