import json
import os
import uuid

from flask import Flask, Response, abort, flash, g, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from auth import create_user, load_logged_in_user, login_required, role_required, verify_user
from chat import generate_xray_appearance, generate_xray_ranking_reason
from db import get_db
from patients import (
    create_patient,
    get_analytics_data,
    get_findings_this_week,
    get_overview_stats,
    get_patient,
    get_predictions_for_patient,
    get_recent_activity,
    list_patients,
)
from predict import compute_symptom_contributions, get_symptom_list, run_gradcam, run_prediction, serialize_result
from report import (DISCLAIMER, build_combined_narrative, build_imaging_narrative,
                    build_narrative, build_symptom_narrative, get_all_predictions_for_patient,
                    get_checked_symptoms,
                    get_prediction, render_pdf)

app = Flask(__name__)
app.secret_key = os.environ.get("CHESTAI_SECRET_KEY", "dev-secret-change-me")

# Auto-create the database schema and default admin account on first boot
if not os.path.exists(os.path.join(os.path.dirname(__file__), "instance", "chestai.db")):
    from db import init_db
    from auth import create_user
    init_db()
    create_user(
        os.environ.get("DEFAULT_ADMIN_USERNAME", "admin"),
        os.environ.get("DEFAULT_ADMIN_PASSWORD", "rivela2026"),
        "admin",
    )

UPLOAD_DIR = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.before_request
def before_request():
    load_logged_in_user()


@app.context_processor
def inject_sidebar():
    if g.user is None:
        return {}
    return {"sidebar_patients": list_patients(g.user["id"])}


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = verify_user(username, password)
        if user is None:
            flash("Invalid username or password.")
            return render_template("login.html")
        session.clear()
        session["user_id"] = user["id"]
        dest = "admin_panel" if user["role"] == "admin" else "dashboard"
        return redirect(url_for(dest))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── ADMIN ──────────────────────────────────────────────────────────────────

@app.route("/admin")
@role_required("admin")
def admin_panel():
    conn = get_db()
    users = conn.execute(
        "SELECT id, username, role, created_at FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return render_template("admin.html", users=users)


@app.route("/admin/users/add", methods=["POST"])
@role_required("admin")
def admin_add_user():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    role = request.form.get("role", "doctor")
    if not username or not password:
        flash("Username and password are required.")
        return redirect(url_for("admin_panel"))
    role = "doctor"  # admin creation is not permitted through the UI
    try:
        create_user(username, password, role)
        flash(f"User '{username}' added as {role}.")
    except Exception:
        flash(f"Username '{username}' already exists.")
    return redirect(url_for("admin_panel"))


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@role_required("admin")
def admin_delete_user(user_id):
    conn = get_db()
    target = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
    if not target or target["role"] == "admin":
        flash("Admin accounts cannot be deleted.")
        conn.close()
        return redirect(url_for("admin_panel"))
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    flash("User deleted.")
    return redirect(url_for("admin_panel"))




# ── DOCTOR ROUTES ───────────────────────────────────────────────────────────

@app.route("/")
@login_required
def dashboard():
    uid = g.user["id"]
    return render_template(
        "dashboard.html",
        stats=get_overview_stats(uid),
        recent_activity=get_recent_activity(uid),
        findings_this_week=get_findings_this_week(uid),
        active_nav="dashboard",
    )


@app.route("/analytics")
@login_required
def analytics():
    return render_template("analytics.html",
                           stats=get_overview_stats(g.user["id"]),
                           data=get_analytics_data(g.user["id"]),
                           active_nav="analytics")


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings_page():
    success = None
    error = None
    if request.method == "POST":
        action = request.form.get("action")
        conn = get_db()
        if action == "update_username":
            new_username = request.form.get("username", "").strip()
            if not new_username:
                error = "Username cannot be empty."
            elif new_username == g.user["username"]:
                error = "That is already your username."
            else:
                try:
                    conn.execute("UPDATE users SET username = ? WHERE id = ?", (new_username, g.user["id"]))
                    conn.commit()
                    g.user = conn.execute("SELECT * FROM users WHERE id = ?", (g.user["id"],)).fetchone()
                    success = "Username updated successfully."
                except Exception:
                    error = "Username already taken."
        elif action == "update_password":
            current_pw = request.form.get("current_password", "")
            new_pw = request.form.get("new_password", "")
            confirm_pw = request.form.get("confirm_password", "")
            from werkzeug.security import check_password_hash, generate_password_hash
            if not check_password_hash(g.user["password_hash"], current_pw):
                error = "Current password is incorrect."
            elif len(new_pw) < 6:
                error = "New password must be at least 6 characters."
            elif new_pw != confirm_pw:
                error = "New passwords do not match."
            else:
                conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                             (generate_password_hash(new_pw), g.user["id"]))
                conn.commit()
                g.user = conn.execute("SELECT * FROM users WHERE id = ?", (g.user["id"],)).fetchone()
                success = "Password updated successfully."
        conn.close()
    return render_template("settings.html", active_nav="settings", success=success, error=error)


@app.route("/patients/new", methods=["GET", "POST"])
@login_required
def new_patient():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            flash("Patient name is required.")
            return render_template("new_patient.html", active_nav="patients")
        patient_id = create_patient(
            name,
            request.form.get("dob", "").strip() or None,
            request.form.get("sex", "").strip() or None,
            g.user["id"],
        )
        return redirect(url_for("patient_info", patient_id=patient_id))
    return render_template("new_patient.html", active_nav="patients")


def _patient_context(patient_id):
    patient = get_patient(patient_id)
    if patient is None or patient["created_by"] != g.user["id"]:
        abort(404)
    predictions = get_predictions_for_patient(patient_id)
    latest = predictions[0] if predictions else None
    return patient, predictions, latest


@app.route("/patients/<int:patient_id>")
@login_required
def patient_info(patient_id):
    patient, predictions, latest = _patient_context(patient_id)
    return render_template(
        "patient_info.html",
        active_nav="patients",
        active_tab="info",
        patient=patient,
        checkup_count=len(predictions),
        latest=latest,
        latest_prediction_id=latest["id"] if latest else None,
    )


@app.route("/patients/<int:patient_id>/history")
@login_required
def patient_history(patient_id):
    patient, predictions, latest = _patient_context(patient_id)
    return render_template(
        "patient_history.html",
        active_nav="patients",
        active_tab="history",
        patient=patient,
        checkup_count=len(predictions),
        predictions=predictions,
        latest_prediction_id=latest["id"] if latest else None,
    )


@app.route("/patients/<int:patient_id>/checkups/new", methods=["GET", "POST"])
@login_required
def new_checkup(patient_id):
    patient, predictions, latest = _patient_context(patient_id)
    symptoms = get_symptom_list()

    if request.method == "POST":
        modality = request.form.get("modality", "both")
        if modality not in ("xray", "symptoms", "both"):
            modality = "both"

        saved_path = None
        xray_image_path = None
        heatmap_path = None
        overlay_path = None
        filename = None

        if modality in ("xray", "both"):
            image_file = request.files.get("xray_image")
            if image_file is None or image_file.filename == "":
                flash("Please upload an X-ray image, or switch to symptoms-only mode.")
                return render_template("new_checkup.html", patient=patient, symptoms=symptoms, active_nav="patients")

            filename = f"{uuid.uuid4().hex}_{secure_filename(image_file.filename)}"
            saved_path = os.path.join(UPLOAD_DIR, filename)
            image_file.save(saved_path)
            xray_image_path = f"uploads/{filename}"

        symptom_dict = {name: 1 for name in symptoms if request.form.get(name) == "on"}
        if modality == "symptoms" and not symptom_dict:
            flash("Please check at least one symptom, or switch to X-ray-only mode.")
            return render_template("new_checkup.html", patient=patient, symptoms=symptoms, active_nav="patients")

        result = run_prediction(
            modality,
            image_path=saved_path,
            symptom_dict=symptom_dict if modality in ("symptoms", "both") else None,
        )
        row = serialize_result(result)
        row["xray_image_path"] = xray_image_path
        row["symptoms_json"] = json.dumps(symptom_dict)

        if saved_path and row["top_disease"]:
            gradcam_target = (
                result["fusion_result"]["xray_top"] if result["fusion_result"] else row["top_disease"]
            )
            _, heatmap_img, overlay_img = run_gradcam(saved_path, gradcam_target)
            heatmap_filename = f"heatmap_{filename}"
            overlay_filename = f"overlay_{filename}"
            heatmap_img.save(os.path.join(UPLOAD_DIR, heatmap_filename))
            overlay_img.save(os.path.join(UPLOAD_DIR, overlay_filename))
            heatmap_path = f"uploads/{heatmap_filename}"
            overlay_path = f"uploads/{overlay_filename}"

        # Generate Gemini-based appearance description using disease + confidence + heatmap region
        xray_appearance_text = None
        if heatmap_path and row["top_disease"] and row["xray_probs_json"]:
            try:
                import json as _j
                xray_probs = _j.loads(row["xray_probs_json"])
                top_conf = xray_probs.get(row["top_disease"], 0)
                heatmap_full = os.path.join(app.static_folder, heatmap_path)
                # reuse the spatial analysis already in the route
                import numpy as np
                from PIL import Image as _PIL
                arr = np.array(_PIL.open(heatmap_full).convert("L")).astype(float)
                arr = arr / arr.max() if arr.max() > 0 else arr
                hot = np.where(arr >= arr.max() * 0.8)
                if len(hot[0]) > 0:
                    cy = hot[0].mean() / arr.shape[0]
                    cx = hot[1].mean() / arr.shape[1]
                    vert = "lower" if cy > 0.55 else "upper" if cy < 0.45 else "mid"
                    horiz = "right" if cx > 0.55 else "left" if cx < 0.45 else "central"
                    region = f"{vert}-{horiz}"
                else:
                    region = None
                xray_appearance_text = generate_xray_appearance(row["top_disease"], top_conf, region)
            except Exception:
                pass

        conn = get_db()
        cursor = conn.execute(
            """INSERT INTO predictions
               (user_id, patient_id, xray_image_path, heatmap_image_path, overlay_image_path,
                symptoms_json, xray_probs_json, symptom_probs_json, fused_result_json,
                xray_uncertainty_json, symptom_uncertainty_json, xray_appearance_text,
                top_disease, status, conflict)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                g.user["id"],
                patient_id,
                row["xray_image_path"],
                heatmap_path,
                overlay_path,
                row["symptoms_json"],
                row["xray_probs_json"],
                row["symptom_probs_json"],
                row["fused_result_json"],
                row["xray_uncertainty_json"],
                row["symptom_uncertainty_json"],
                xray_appearance_text,
                row["top_disease"],
                row["status"],
                row["conflict"],
            ),
        )
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return redirect(url_for("patient_analysis", patient_id=patient_id, prediction_id=prediction_id))

    return render_template("new_checkup.html", patient=patient, symptoms=symptoms, active_nav="patients")


@app.route("/patients/<int:patient_id>/checkups/<int:prediction_id>")
@login_required
def patient_analysis(patient_id, prediction_id):
    patient, predictions, latest = _patient_context(patient_id)
    record = get_prediction(prediction_id)
    if record is None or record["patient_id"] != patient_id:
        abort(404)

    import json as _json
    xray_thresholds = {}
    try:
        with open("model_weights/fusion_config.json") as f:
            xray_thresholds = _json.load(f).get("xray_thresholds", {})
    except Exception:
        pass

    # Only show X-ray bars that meet their per-disease detection threshold
    xray_probs_sorted = None
    if record["xray_probs"]:
        xray_probs_sorted = sorted(
            [(d, p) for d, p in record["xray_probs"].items() if p >= xray_thresholds.get(d, 0)],
            key=lambda kv: kv[1], reverse=True
        )

    symptom_probs_sorted = (
        sorted(record["symptom_probs"].items(), key=lambda kv: kv[1], reverse=True)
        if record["symptom_probs"]
        else None
    )
    symptom_contributions = compute_symptom_contributions(record["symptoms"], record["top_disease"])

    # Derive a display status for single-modality predictions (no fusion_result / no status)
    status_thresholds = {"confirmed": 0.70, "possible": 0.45}
    display_status = record["status"]
    display_confidence = None
    display_modality = None
    if not display_status and record["top_disease"]:
        if record["xray_probs"] and record["top_disease"] in record["xray_probs"]:
            conf = record["xray_probs"][record["top_disease"]]
            display_confidence = conf
            display_modality = "X-ray only"
            if conf >= status_thresholds["confirmed"]:
                display_status = "confirmed"
            elif conf >= status_thresholds["possible"]:
                display_status = "possible"
            else:
                display_status = "uncertain"
        elif record["symptom_probs"] and record["top_disease"] in record["symptom_probs"]:
            conf = record["symptom_probs"][record["top_disease"]]
            display_confidence = conf
            display_modality = "Symptoms only"
            if conf >= status_thresholds["confirmed"]:
                display_status = "confirmed"
            elif conf >= status_thresholds["possible"]:
                display_status = "possible"
            else:
                display_status = "uncertain"

    # Disease-specific descriptions of typical X-ray appearances
    XRAY_APPEARANCES = {
        "Pneumonia": (
            "Pneumonia typically manifests on chest radiograph as areas of increased "
            "pulmonary opacity (airspace consolidation), often lobar or segmental in distribution. "
            "The model's attention was expected to focus on regions of parenchymal opacification."
        ),
        "Cardiomegaly": (
            "Cardiomegaly is identified radiographically by an enlarged cardiac silhouette, "
            "typically defined by a cardiothoracic ratio exceeding 0.5 on a PA view. "
            "The model's attention was expected to centre on the cardiac shadow and its margins."
        ),
        "COVID-19": (
            "COVID-19 pneumonia characteristically presents as bilateral, peripheral ground-glass "
            "opacities, often lower-zone predominant in the early stages. "
            "The model's attention was expected to focus on peripheral lung zones bilaterally."
        ),
        "Consolidation": (
            "Consolidation appears as homogeneous increased density within the lung parenchyma, "
            "replacing normal aerated lung. Air bronchograms may be present. "
            "The model's attention was expected to focus on areas of increased opacity."
        ),
        "Edema": (
            "Pulmonary oedema typically presents with perihilar haze, bilateral basal opacification, "
            "Kerley B lines, and upper lobe venous diversion. "
            "The model's attention was expected to focus on perihilar and basal regions."
        ),
        "Emphysema": (
            "Emphysema is characterised radiographically by hyperinflation, flattening of the "
            "diaphragms, increased AP diameter, and attenuation of the pulmonary vasculature. "
            "The model's attention was expected to focus on lung volume and diaphragm contour."
        ),
        "Fibrosis": (
            "Pulmonary fibrosis presents as a reticular or reticulonodular interstitial pattern, "
            "typically basal and peripheral in distribution, with possible honeycombing in advanced cases. "
            "The model's attention was expected to focus on basal interstitial markings."
        ),
        "No Finding": (
            "A 'No Finding' classification indicates no significant radiographic abnormality was detected. "
            "The model found no dominant opacification, consolidation, or structural abnormality "
            "above its detection threshold."
        ),
        "Nodule": (
            "A pulmonary nodule appears as a discrete, rounded opacity less than 3 cm in diameter. "
            "The model's attention was expected to focus on a localised region of rounded opacity "
            "within the lung fields."
        ),
        "Pneumothorax": (
            "Pneumothorax is identified by the absence of lung markings at the periphery and a "
            "visible visceral pleural line. The model's attention was expected to focus on the "
            "peripheral lung margins where the pleural line would be visible."
        ),
        "Tuberculosis": (
            "Tuberculosis most commonly presents with upper-zone opacification, cavitation, "
            "and nodular or patchy consolidation. Lymphadenopathy may also be present. "
            "The model's attention was expected to focus on the upper lung zones."
        ),
    }

    # Analyse stored heatmap to describe where the model focused spatially
    def _heatmap_region(heatmap_path):
        if not heatmap_path:
            return None
        try:
            import numpy as np
            from PIL import Image as _PILImage
            full_path = os.path.join(app.static_folder, heatmap_path)
            img = _PILImage.open(full_path).convert("L")
            arr = np.array(img).astype(float)
            arr = arr / arr.max() if arr.max() > 0 else arr
            # Only consider high-activation pixels (top 20%)
            threshold = arr.max() * 0.8
            hot = np.where(arr >= threshold)
            if len(hot[0]) == 0:
                return None
            cy = hot[0].mean() / arr.shape[0]  # 0=top, 1=bottom
            cx = hot[1].mean() / arr.shape[1]  # 0=left, 1=right
            vertical = "lower" if cy > 0.55 else "upper" if cy < 0.45 else "mid"
            horizontal = "right" if cx > 0.55 else "left" if cx < 0.45 else "central"
            return f"{vertical}-{horizontal}"
        except Exception:
            return None

    heatmap_region = _heatmap_region(record.get("heatmap_image_path"))

    # Build a plain-English explanation of why the top finding was selected over the runner-up
    # When fusion ran, use the fused ranked list so the explanation matches the primary finding
    finding_explanation = None
    fr = record.get("fusion_result")
    if fr and fr.get("ranked"):
        all_above_threshold = [(d, s) for d, s in fr["ranked"] if s > 0]
    else:
        all_above_threshold = xray_probs_sorted or []
    if all_above_threshold and record["top_disease"]:
        top_d, top_p = all_above_threshold[0]

        # Part 1: Gemini-generated description if available and matches top_d, else fallback
        stored_text = record.get("xray_appearance_text")
        appearance = (stored_text if stored_text else None) or XRAY_APPEARANCES.get(top_d, "")

        # Part 2: where the heatmap focused
        if heatmap_region:
            region_parts = heatmap_region.split("-")
            vertical = region_parts[0] if len(region_parts) > 1 else ""
            horizontal = region_parts[1] if len(region_parts) > 1 else heatmap_region
            if horizontal == "central":
                region_text = f"Grad-CAM saliency analysis indicates the model's primary attention was concentrated in the {vertical} central region of the chest."
            else:
                region_text = f"Grad-CAM saliency analysis indicates the model's primary attention was concentrated in the {vertical} {horizontal} region of the chest."
        else:
            region_text = "Grad-CAM saliency mapping was applied to visualise the spatial regions driving this prediction — see the overlay image above."

        # Part 3: why it ranked above the runner-up
        if len(all_above_threshold) >= 2:
            second_d, second_p = all_above_threshold[1]
            margin = top_p - second_p
            ratio = top_p / second_p if second_p > 0 else None
            margin_desc = "substantially" if margin >= 0.40 else "notably" if margin >= 0.20 else "moderately"
            ratio_text = f" ({ratio:.1f}× higher probability)" if ratio and ratio >= 1.5 else ""
            ambiguity = (
                "This margin suggests the model has a clear preference for the primary diagnosis."
                if margin >= 0.30 else
                f"This relatively narrow margin ({margin * 100:.1f} percentage points) indicates "
                f"some ambiguity between {top_d} and {second_d}; the secondary finding warrants consideration in clinical context."
            )
            ranking_parts = [
                f"{top_d} was ranked as the primary finding with a probability of {top_p * 100:.1f}%, "
                f"which {margin_desc} exceeds the next closest finding, {second_d} ({second_p * 100:.1f}%){ratio_text}. "
                f"{ambiguity}"
            ]

            s_probs = record.get("symptom_probs") or {}
            has_symptoms = bool(record.get("symptoms") and any(record["symptoms"].values()))

            if has_symptoms and s_probs:
                # Symptom model comparison between the two diseases
                s_top = s_probs.get(top_d, 0)
                s_second = s_probs.get(second_d, 0)
                if s_top > s_second:
                    ranking_parts.append(
                        f"The symptom model also favoured {top_d} ({s_top * 100:.1f}%) over "
                        f"{second_d} ({s_second * 100:.1f}%), reinforcing the X-ray model's preference."
                    )
                elif s_second > s_top and s_second > 0.05:
                    ranking_parts.append(
                        f"However, the symptom model assigned a higher probability to {second_d} "
                        f"({s_second * 100:.1f}%) than to {top_d} ({s_top * 100:.1f}%), "
                        f"indicating some disagreement between modalities — the secondary finding "
                        f"warrants careful clinical consideration."
                    )

                # Top symptoms that contributed to the primary finding
                if symptom_contributions:
                    driving = [(s, d) for s, d in symptom_contributions if d > 0.02][:3]
                    if driving:
                        sym_names = ", ".join(s.replace("_", " ") for s, _ in driving)
                        ranking_parts.append(
                            f"The reported symptoms most strongly associated with {top_d} in this case "
                            f"were: {sym_names}."
                        )
            else:
                # X-ray only — use Gemini to explain visual feature difference
                visual_reason = generate_xray_ranking_reason(
                    top_d, top_p, second_d, second_p, heatmap_region
                )
                if visual_reason:
                    ranking_parts.append(visual_reason)

            ranking_text = " ".join(ranking_parts)
        else:
            ranking_text = (
                f"{top_d} is the only finding that exceeded its detection threshold ({top_p * 100:.1f}%). "
                f"All other disease probabilities fell below their respective per-disease thresholds."
            )

        finding_explanation = {
            "appearance": appearance,
            "region": region_text,
            "ranking": ranking_text,
        }

    return render_template(
        "patient_analysis.html",
        active_nav="patients",
        active_tab="analysis",
        patient=patient,
        checkup_count=len(predictions),
        latest_prediction_id=prediction_id,
        record=record,
        xray_probs_sorted=xray_probs_sorted,
        xray_thresholds=xray_thresholds,
        display_status=display_status,
        display_confidence=display_confidence,
        display_modality=display_modality,
        finding_explanation=finding_explanation,
        symptom_probs_sorted=symptom_probs_sorted,
        symptom_contributions=symptom_contributions,
        disclaimer=DISCLAIMER,
    )


def _get_confidence(p):
    if p["fused_result_json"]:
        try:
            return json.loads(p["fused_result_json"]).get("top_score")
        except Exception:
            pass
    if p["xray_probs_json"]:
        try:
            xray = json.loads(p["xray_probs_json"])
            return max(xray.values()) if xray else None
        except Exception:
            pass
    if p["symptom_probs_json"]:
        try:
            symp = json.loads(p["symptom_probs_json"])
            return max(symp.values()) if symp else None
        except Exception:
            pass
    return None


def _build_trend_points(predictions):
    points = []
    for p in reversed(predictions):
        second_disease = second_conf = None
        all_probs = {}
        conflict_detail = None
        try:
            if p["fused_result_json"]:
                fr = json.loads(p["fused_result_json"])
                ranked = [(d, s) for d, s in fr.get("ranked", []) if s > 0]
                if len(ranked) >= 2:
                    second_disease, second_conf = ranked[1]
                all_probs = dict(fr.get("ranked", []))
                xt, st = fr.get("xray_top"), fr.get("symptom_top")
                if p["conflict"] and xt and st:
                    conflict_detail = f"X-ray → {xt} · Symptoms → {st}"
            elif p["xray_probs_json"]:
                xray = json.loads(p["xray_probs_json"])
                sorted_probs = sorted(xray.items(), key=lambda kv: kv[1], reverse=True)
                if len(sorted_probs) >= 2:
                    second_disease, second_conf = sorted_probs[1]
                all_probs = xray
        except Exception:
            pass

        points.append({
            "prediction_id": p["id"],
            "date": p["created_at"][:10],
            "top_disease": p["top_disease"],
            "confidence": _get_confidence(p),
            "status": p["status"],
            "conflict": bool(p["conflict"]),
            "conflict_detail": conflict_detail,
            "xray_image_path": p["xray_image_path"],
            "heatmap_image_path": p["heatmap_image_path"],
            "overlay_image_path": p["overlay_image_path"],
            "second_disease": second_disease,
            "second_conf": second_conf,
            "all_probs": all_probs,
            "attention_weight": None,
            "heatmap_region": None,
        })
    return points


def _heatmap_weight(heatmap_path, static_folder):
    """Peak activation intensity (0-1) and spatial region from a saved heatmap image."""
    if not heatmap_path:
        return None, None
    try:
        import numpy as np
        from PIL import Image as _PIL
        raw = np.array(_PIL.open(os.path.join(static_folder, heatmap_path)).convert("L"))
        if raw.max() == 0:
            return 0.0, None
        weight = float(raw.max()) / 255.0
        arr = raw.astype(float) / raw.max()
        hot = np.where(arr >= 0.8)
        region = None
        if len(hot[0]) > 0:
            cy = hot[0].mean() / arr.shape[0]
            cx = hot[1].mean() / arr.shape[1]
            vert = "lower" if cy > 0.55 else "upper" if cy < 0.45 else "mid"
            horiz = "right" if cx > 0.55 else "left" if cx < 0.45 else "central"
            region = f"{vert}-{horiz}"
        return weight, region
    except Exception:
        return None, None


def _compute_trend_label(trend_points):
    if len(trend_points) < 2:
        return None, None
    first, last = trend_points[0], trend_points[-1]
    fd, ld = first["top_disease"], last["top_disease"]
    if fd and ld and fd != ld:
        return "Changed", f"{fd} → {ld}"
    d = ld or ""
    fc, lc = first["confidence"], last["confidence"]
    if fc is not None and lc is not None:
        delta = lc - fc
        if d == "No Finding":
            return "Stable", "Consistently clear — no significant findings"
        if delta > 0.05:
            return "Progressing", f"Confidence in {d} increasing — review recommended"
        if delta < -0.05:
            return "Improving", f"Confidence in {d} decreasing"
        return "Stable", f"{d} trend unchanged across checkups"
    return "Stable", "Insufficient data for trend"


def _build_follow_up_plan(trend_points):
    if len(trend_points) < 2:
        return []
    first, last = trend_points[0], trend_points[-1]
    plan = []

    if last.get("conflict"):
        plan.append({
            "title": "Investigate modality conflict",
            "detail": "The latest checkup shows disagreement between X-ray and symptom models — clinical correlation is recommended.",
        })

    if first["top_disease"] != last["top_disease"] and first["top_disease"] and last["top_disease"]:
        plan.append({
            "title": f"Assess new primary finding: {last['top_disease']}",
            "detail": f"Primary finding changed from {first['top_disease']} to {last['top_disease']} — warrants direct clinical evaluation.",
        })
    elif first["confidence"] and last["confidence"] and (last["confidence"] - first["confidence"]) > 0.05:
        d = last["top_disease"] or "finding"
        plan.append({
            "title": f"Clinical review of {d}",
            "detail": f"Model confidence increased from {first['confidence']*100:.1f}% to {last['confidence']*100:.1f}% — the AI models are increasingly certain of this finding.",
        })

    fw = first.get("attention_weight")
    lw = last.get("attention_weight")
    lr = (last.get("heatmap_region") or "").replace("-", " ")
    if fw is not None and lw is not None and (lw - fw) > 0.15 and lr:
        plan.append({
            "title": f"Grad-CAM focus intensified — {lr} region",
            "detail": f"Model attention increased significantly compared to baseline, now concentrated in the {lr} chest region.",
        })

    plan.append({
        "title": "Schedule next AI checkup",
        "detail": "Repeat imaging in 4–6 weeks to monitor progression and allow the models to update their assessment.",
    })

    return plan[:3]


@app.route("/patients/<int:patient_id>/longitudinal")
@login_required
def patient_longitudinal(patient_id):
    patient, predictions, latest = _patient_context(patient_id)
    trend_points = _build_trend_points(predictions)

    for pt in trend_points:
        w, r = _heatmap_weight(pt.get("heatmap_image_path"), app.static_folder)
        pt["attention_weight"] = w
        pt["heatmap_region"] = r

    xray_points = [p for p in trend_points if p["xray_image_path"]]
    compare_first = xray_points[0] if xray_points else None
    compare_latest = xray_points[-1] if len(xray_points) > 1 else None

    trend_label, trend_note = _compute_trend_label(trend_points)
    follow_up_plan = _build_follow_up_plan(trend_points)

    secondary_trend = None
    if len(trend_points) >= 2:
        lp, fp = trend_points[-1], trend_points[0]
        sec_d = lp.get("second_disease")
        sec_conf_last = lp.get("second_conf")
        if sec_d and sec_conf_last is not None:
            sec_conf_first = fp.get("all_probs", {}).get(sec_d)
            direction = None
            if sec_conf_first is not None:
                diff = sec_conf_last - sec_conf_first
                direction = "Rising" if diff > 0.03 else "Falling" if diff < -0.03 else "Stable"
            secondary_trend = {
                "disease": sec_d,
                "conf_first": sec_conf_first,
                "conf_last": sec_conf_last,
                "direction": direction,
            }

    return render_template(
        "patient_longitudinal.html",
        active_nav="patients",
        active_tab="longitudinal",
        patient=patient,
        checkup_count=len(predictions),
        latest_prediction_id=latest["id"] if latest else None,
        trend_points=trend_points,
        trend_json=json.dumps(trend_points),
        compare_first=compare_first,
        compare_latest=compare_latest,
        trend_label=trend_label,
        trend_note=trend_note,
        follow_up_plan=follow_up_plan,
        secondary_trend=secondary_trend,
    )


def _narrative_sections(record):
    return {
        "imaging": build_imaging_narrative(record),
        "symptoms": build_symptom_narrative(record),
        "combined": build_combined_narrative(record),
    }


@app.route("/patients/<int:patient_id>/report")
@login_required
def patient_report(patient_id):
    patient, predictions, latest = _patient_context(patient_id)
    all_records = get_all_predictions_for_patient(patient_id)
    return render_template(
        "patient_report.html",
        active_nav="patients",
        active_tab="report",
        patient=patient,
        checkup_count=len(predictions),
        latest_prediction_id=latest["id"] if latest else None,
        all_records=all_records,
    )



@app.route("/patients/<int:patient_id>/report/pdf")
@login_required
def patient_report_pdf(patient_id):
    patient, predictions, latest = _patient_context(patient_id)
    all_records = get_all_predictions_for_patient(patient_id)

    records_with_data = []
    for r in all_records:
        records_with_data.append({
            "record": r,
            "narrative": build_narrative(r),
            "checked_symptoms": get_checked_symptoms(r),
        })

    from datetime import datetime as _dt
    html_string = render_template(
        "report_pdf.html",
        patient=patient,
        records_with_data=records_with_data,
        disclaimer=DISCLAIMER,
        generated_date=_dt.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )
    pdf_bytes = render_pdf(html_string, app.static_folder)
    safe_name = (patient["name"] or "patient").replace(" ", "_")
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=rivela_{safe_name}_full_report.pdf"},
    )


if __name__ == "__main__":
    # use_reloader=False: PyTorch touches files inside its own package directory
    # at import time, which the watchdog reloader misreads as a source change and
    # restarts the server mid-request, killing in-flight predictions.
    app.run(debug=True, use_reloader=False)
