import json
import os

from google import genai

from db import get_db

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are the Rivela AI reporting assistant. Rivela AI's own trained \
machine learning models (an X-ray model and a symptom model) have already produced \
the diagnostic numbers below for one specific patient case.

Your role is to help a doctor or medical student understand WHY the model reached \
its conclusions — not just what the numbers are, but what they mean clinically and \
radiographically.

Strict rules, do not break them:
1. Never state a probability or confidence score that is not in the data below. \
   All numbers must come from the structured data.
2. Never override or contradict the model's output. You explain it, not replace it.
3. Never give treatment advice or dosing.
4. When asked "why this disease and not that one", you SHOULD:
   - State the probability difference from the data.
   - Explain what the top disease typically looks like on a chest X-ray and why \
     the model's attention pattern (if available in the data) is consistent with it.
   - Explain what the alternative disease would typically look like radiographically \
     and why those features were NOT strongly present according to the model's output.
   This radiographic comparison uses your medical knowledge to contextualise the \
   model's numbers — you are narrating the model's decision, not making your own.
5. Write in plain, clear prose. No markdown formatting, no bullet points with *, \
   no asterisks for bold. Plain paragraphs only.

Here is the complete structured data for this case:

{case_data}
"""


_KEY_FILE = os.path.join(os.path.dirname(__file__), "instance", "gemini_key.txt")


def _load_api_key():
    # 1. Environment variable takes priority
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    # 2. Fall back to instance/gemini_key.txt
    try:
        with open(_KEY_FILE) as f:
            key = f.read().strip()
        if key and key != "PASTE_YOUR_GEMINI_API_KEY_HERE":
            return key
    except FileNotFoundError:
        pass
    raise RuntimeError(
        "Gemini API key not found. Either set the GEMINI_API_KEY environment variable, "
        "or paste your key into instance/gemini_key.txt"
    )


def _client():
    return genai.Client(api_key=_load_api_key())


def _build_case_data(record):
    # Derive heatmap region from stored appearance text context if available
    data = {
        "patient_name": record.get("patient_name"),
        "checkup_date": record.get("created_at", "")[:16],
        "top_disease": record.get("top_disease"),
        "status": record.get("status"),
        "symptoms_reported": {k: v for k, v in record["symptoms"].items() if v},
        "xray_probabilities": record["xray_probs"],
        "symptom_probabilities": record["symptom_probs"],
        "fusion_result": record["fusion_result"],
    }
    if record.get("xray_uncertainty"):
        top = record.get("top_disease")
        u = record["xray_uncertainty"].get(top) if top else None
        if u:
            data["xray_uncertainty"] = {
                "disease": top,
                "mean_probability": f"{u['mean']*100:.1f}%",
                "std_deviation": f"{u['std']*100:.1f}%",
                "level": "low" if u["std"] < 0.08 else "moderate" if u["std"] < 0.18 else "high",
            }
    if record.get("symptom_uncertainty"):
        top = record.get("top_disease")
        u = record["symptom_uncertainty"].get(top) if top else None
        if u:
            data["symptom_uncertainty"] = {
                "disease": top,
                "std_deviation": f"{u['std']*100:.1f}%",
                "level": "low" if u["std"] < 0.08 else "moderate" if u["std"] < 0.18 else "high",
            }
    if record.get("xray_appearance_text"):
        data["gradcam_analysis"] = record["xray_appearance_text"]
    return json.dumps(data, indent=2)


def get_chat_history(prediction_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, created_at FROM chat_messages WHERE prediction_id = ? ORDER BY id",
        (prediction_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _save_message(prediction_id, role, content):
    conn = get_db()
    conn.execute(
        "INSERT INTO chat_messages (prediction_id, role, content) VALUES (?, ?, ?)",
        (prediction_id, role, content),
    )
    conn.commit()
    conn.close()


def ask(record, prediction_id, question):
    history = get_chat_history(prediction_id)

    system_instruction = SYSTEM_PROMPT.format(case_data=_build_case_data(record))

    contents = []
    for msg in history:
        gemini_role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": question}]})

    client = _client()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config={"system_instruction": system_instruction},
    )
    answer = response.text

    _save_message(prediction_id, "user", question)
    _save_message(prediction_id, "assistant", answer)

    return answer


def generate_xray_ranking_reason(top_disease, top_conf, second_disease, second_conf, heatmap_region):
    """Explain why the X-ray model ranked top_disease above second_disease based on
    visual features in the heatmap region. Called only for X-ray-only predictions."""
    try:
        region_text = heatmap_region.replace("-", " ") if heatmap_region else "unspecified"
        prompt = (
            f"A chest X-ray model predicted {top_disease} with {top_conf * 100:.1f}% confidence "
            f"and {second_disease} with {second_conf * 100:.1f}% confidence. "
            f"Grad-CAM showed the model's attention was concentrated in the {region_text} region of the chest.\n\n"
            f"In 2 sentences, explain:\n"
            f"1. What visual X-ray features in the {region_text} region are more consistent with "
            f"{top_disease} than with {second_disease}.\n"
            f"2. Why those features caused the model to prefer {top_disease} over {second_disease}.\n\n"
            f"Rules: Do not diagnose. Do not recommend treatment. Only describe radiographic feature "
            f"differences between the two conditions in that chest region. "
            f"Write in plain prose, no bullet points, no markdown formatting."
        )
        client = _client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        return response.text.strip()
    except Exception:
        return None


def generate_xray_appearance(disease, confidence, heatmap_region):
    """Call Gemini once at prediction time to generate a prediction-specific
    description of what the model's attention pattern means clinically.
    Gemini only receives: disease name, confidence score, heatmap region.
    It never sees the actual X-ray image."""
    try:
        region_text = heatmap_region.replace("-", " ") if heatmap_region else "unspecified"

        prompt = (
            f"The Rivela AI X-ray model predicted {disease} with a confidence of "
            f"{confidence * 100:.1f}%. Grad-CAM saliency analysis showed the model's "
            f"attention was concentrated in the {region_text} region of the chest.\n\n"
            f"In 2-3 sentences, describe:\n"
            f"1. What anatomical structure is located in the {region_text} region of the chest.\n"
            f"2. Why this region is clinically relevant to {disease}.\n"
            f"3. Whether this attention pattern is consistent with typical radiographic "
            f"presentations of {disease}.\n\n"
            f"Rules: Do not diagnose. Do not say what treatment should be given. "
            f"Only describe the anatomical and clinical relevance of this attention pattern. "
            f"Write in plain prose, no bullet points, no markdown formatting."
        )

        client = _client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        return response.text.strip()
    except Exception:
        return None
