import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset
import json
from datetime import datetime
from src.config import REFERENCE_PATH, CURRENT_PATH, REPORT_PATH, METRICS_JSON

TARGET_COLUMN = "label"                # nome della colonna target
PRED_COLUMN = "prediction"             # nome della colonna delle predizioni

# === 1. Carica dati ===
reference = pd.read_csv(REFERENCE_PATH)
current = pd.read_csv(CURRENT_PATH)

# assicurati che le colonne coincidano
common_cols = list(set(reference.columns) & set(current.columns))
reference = reference[common_cols]
current = current[common_cols]

# === 2. Crea il report Evidently ===
report = Report(metrics=[
    DataDriftPreset(),           # analisi del data drift
    ClassificationPreset()       # metriche di performance (accuracy, precision, recall, ecc.)
])

report.run(reference_data=reference, current_data=current)

# === 3. Salva il report HTML ===
report.save_html(REPORT_PATH)

# === 4. Esporta metriche principali per alert semplificato ===
result = report.as_dict()

# estrai metriche sintetiche
data_drift = result["metrics"][0]["result"]["dataset_drift"]
n_drifted_features = result["metrics"][0]["result"]["number_of_drifted_columns"]
accuracy_ref = result["metrics"][1]["result"]["reference_metrics"]["accuracy"]
accuracy_cur = result["metrics"][1]["result"]["current_metrics"]["accuracy"]

summary = {
    "timestamp": datetime.now().isoformat(),
    "data_drift_detected": data_drift,
    "n_drifted_features": n_drifted_features,
    "accuracy_reference": accuracy_ref,
    "accuracy_current": accuracy_cur,
    "accuracy_drop": round(accuracy_ref - accuracy_cur, 4)
}

with open(METRICS_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print("✅ Monitoring report salvato in:", REPORT_PATH)
print("📊 Metriche riassuntive:")
print(json.dumps(summary, indent=2))

# === 5. (Facoltativo) Semplice alert in console ===
if summary["data_drift_detected"] or summary["accuracy_drop"] > 0.05:
    print("⚠️  ATTENZIONE: rilevato drift o degrado delle performance!")
