CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('doctor', 'admin')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    dob TEXT,
    sex TEXT,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    patient_id INTEGER REFERENCES patients(id),
    xray_image_path TEXT,
    heatmap_image_path TEXT,
    overlay_image_path TEXT,
    symptoms_json TEXT,
    xray_probs_json TEXT,
    symptom_probs_json TEXT,
    fused_result_json TEXT,
    xray_uncertainty_json TEXT,
    symptom_uncertainty_json TEXT,
    xray_appearance_text TEXT,
    top_disease TEXT,
    status TEXT,
    conflict INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);