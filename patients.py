import json as _j

from db import get_db


def list_patients(user_id):
    conn = get_db()
    rows = conn.execute(
        """SELECT patients.*,
                  COUNT(predictions.id) AS checkup_count,
                  MAX(predictions.created_at) AS last_checkup,
                  (SELECT top_disease FROM predictions p2
                   WHERE p2.patient_id = patients.id
                   ORDER BY p2.created_at DESC LIMIT 1) AS last_disease,
                  (SELECT status FROM predictions p3
                   WHERE p3.patient_id = patients.id
                   ORDER BY p3.created_at DESC LIMIT 1) AS last_status,
                  (SELECT conflict FROM predictions p4
                   WHERE p4.patient_id = patients.id
                   ORDER BY p4.created_at DESC LIMIT 1) AS last_conflict
           FROM patients
           LEFT JOIN predictions ON predictions.patient_id = patients.id
           WHERE patients.created_by = ?
           GROUP BY patients.id
           ORDER BY COALESCE(MAX(predictions.created_at), '0000') DESC, patients.name""",
        (user_id,),
    ).fetchall()
    conn.close()
    return rows


def get_patient(patient_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    conn.close()
    return row


def create_patient(name, dob, sex, user_id):
    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO patients (name, dob, sex, created_by) VALUES (?, ?, ?, ?)",
        (name, dob, sex, user_id),
    )
    conn.commit()
    patient_id = cursor.lastrowid
    conn.close()
    return patient_id


def get_predictions_for_patient(patient_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE patient_id = ? ORDER BY created_at DESC",
        (patient_id,),
    ).fetchall()
    conn.close()
    return rows


def get_overview_stats(user_id):
    conn = get_db()
    patient_count = conn.execute(
        "SELECT COUNT(*) FROM patients WHERE created_by = ?", (user_id,)
    ).fetchone()[0]
    checkup_count = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    avg_conf_row = conn.execute(
        "SELECT AVG(json_extract(fused_result_json, '$.top_score')) FROM predictions "
        "WHERE fused_result_json IS NOT NULL AND user_id = ?",
        (user_id,),
    ).fetchone()
    conflicts_flagged = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE conflict = 1 AND user_id = ?", (user_id,)
    ).fetchone()[0]
    conn.close()
    avg_confidence = round((avg_conf_row[0] or 0) * 100) if avg_conf_row else 0
    return {
        "patients": patient_count,
        "checkups": checkup_count,
        "avg_confidence": avg_confidence,
        "conflicts_flagged": conflicts_flagged,
    }


def get_recent_activity(user_id, limit=5):
    conn = get_db()
    rows = conn.execute(
        """SELECT predictions.id, predictions.created_at, predictions.top_disease,
                  predictions.status, predictions.conflict,
                  patients.name AS patient_name, patients.id AS patient_id
           FROM predictions
           JOIN patients ON patients.id = predictions.patient_id
           WHERE predictions.user_id = ?
           ORDER BY predictions.created_at DESC
           LIMIT ?""",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return rows


def get_findings_this_week(user_id):
    conn = get_db()
    rows = conn.execute(
        """SELECT top_disease, COUNT(*) AS cnt
           FROM predictions
           WHERE user_id = ?
             AND top_disease IS NOT NULL
             AND DATE(created_at) >= DATE('now', '-7 days')
           GROUP BY top_disease
           ORDER BY cnt DESC
           LIMIT 5""",
        (user_id,),
    ).fetchall()
    conn.close()
    return [{"disease": r["top_disease"], "count": r["cnt"]} for r in rows]


def get_analytics_data(user_id):
    conn = get_db()

    disease_rows = conn.execute(
        "SELECT top_disease, COUNT(*) as cnt FROM predictions "
        "WHERE top_disease IS NOT NULL AND user_id = ? GROUP BY top_disease ORDER BY cnt DESC",
        (user_id,),
    ).fetchall()
    disease_labels = [r["top_disease"] for r in disease_rows]
    disease_counts = [r["cnt"] for r in disease_rows]

    status_rows = conn.execute(
        "SELECT json_extract(fused_result_json, '$.status') as st, COUNT(*) as cnt "
        "FROM predictions WHERE fused_result_json IS NOT NULL AND user_id = ? GROUP BY st",
        (user_id,),
    ).fetchall()
    status_map = {r["st"]: r["cnt"] for r in status_rows if r["st"]}

    fused = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE fused_result_json IS NOT NULL AND user_id = ?",
        (user_id,),
    ).fetchone()[0]
    xray_only = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE xray_probs_json IS NOT NULL "
        "AND (symptom_probs_json IS NULL OR symptom_probs_json = 'null') AND user_id = ?",
        (user_id,),
    ).fetchone()[0]
    symptoms_only = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE symptom_probs_json IS NOT NULL "
        "AND xray_image_path IS NULL AND user_id = ?",
        (user_id,),
    ).fetchone()[0]

    time_rows = conn.execute(
        "SELECT DATE(created_at) as day, COUNT(*) as cnt "
        "FROM predictions WHERE user_id = ? GROUP BY day ORDER BY day",
        (user_id,),
    ).fetchall()
    time_labels = [r["day"] for r in time_rows]
    time_counts = [r["cnt"] for r in time_rows]

    all_preds = conn.execute(
        "SELECT fused_result_json, xray_probs_json, symptom_probs_json, top_disease "
        "FROM predictions WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    conn.close()

    high, moderate, low = 0, 0, 0
    for p in all_preds:
        conf = None
        if p["fused_result_json"]:
            try: conf = _j.loads(p["fused_result_json"]).get("top_score")
            except: pass
        if conf is None and p["xray_probs_json"] and p["top_disease"]:
            try: conf = _j.loads(p["xray_probs_json"]).get(p["top_disease"])
            except: pass
        if conf is None and p["symptom_probs_json"] and p["top_disease"]:
            try: conf = _j.loads(p["symptom_probs_json"]).get(p["top_disease"])
            except: pass
        if conf is None:
            continue
        if conf >= 0.85: high += 1
        elif conf >= 0.60: moderate += 1
        else: low += 1

    return {
        "disease_labels": disease_labels,
        "disease_counts": disease_counts,
        "status_confirmed": status_map.get("confirmed", 0),
        "status_possible":  status_map.get("possible", 0),
        "status_uncertain": status_map.get("uncertain", 0),
        "modality_fused":    fused,
        "modality_xray":     xray_only,
        "modality_symptoms": symptoms_only,
        "time_labels": time_labels,
        "time_counts": time_counts,
        "conf_high":     high,
        "conf_moderate": moderate,
        "conf_low":      low,
    }
