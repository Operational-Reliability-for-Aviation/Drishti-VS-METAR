from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import re
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import os

app = Flask(__name__, static_folder=".")

MODEL_DIR = Path("models_12h")
SCALE = 1.018
OFFSET = 1


models = {}
for step in range(1, 25):
    path = MODEL_DIR / f"vis_t+{step}.pkl"
    if path.exists():
        models[step] = joblib.load(path)

print(f" Loaded {len(models)} models")

FEATURES = [
    "wind", "temp", "dew", "fog", "mist",
    "hour", "month", "night", "winter"
] + [f"vis_lag_{l}" for l in range(1, 13)]


def parse_metar(metar):
    out = {"vis": np.nan, "wind": 0, "temp": np.nan, "dew": np.nan, "fog": 0, "mist": 0}
    m = re.search(r"\b(\d{4})\b", metar)
    if m:
        out["vis"] = int(m.group(1))
    m = re.search(r"(\d{2,3})KT", metar)
    if m:
        out["wind"] = int(m.group(1))
    m = re.search(r"(M?\d{2})/(M?\d{2})", metar)
    if m:
        out["temp"] = int(m.group(1).replace("M", "-"))
        out["dew"]  = int(m.group(2).replace("M", "-"))
    out["fog"]  = int("FG" in metar)
    out["mist"] = int(("BR" in metar) or ("MIFG" in metar))
    return out


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not models:
        return jsonify({"error": "No models loaded. Make sure models_12h/ folder exists."}), 500

    data = request.json
    metars = data.get("metars", [])
    latest_ist = data.get("latest_ist")  

    if len(metars) < 12:
        return jsonify({"error": "Need exactly 12 METARs"}), 400

    try:
        latest_valid = datetime.fromisoformat(latest_ist)
    except Exception:
        return jsonify({"error": "Invalid datetime format"}), 400

    
    latest_parsed = parse_metar(metars[-1])

 
    vis_values = [parse_metar(m)["vis"] for m in metars[-12:]]

    lag_features = {}
    for l in range(1, 13):
        idx = -(l + 1) if l % 3 == 0 else -l
        try:
            val = vis_values[idx]
        except IndexError:
            val = np.nan
        lag_features[f"vis_lag_{l}"] = float(val * SCALE) if not (isinstance(val, float) and np.isnan(val)) else None

    hour  = latest_valid.hour
    month = latest_valid.month

    row = {
        "wind":   float(latest_parsed["wind"]),
        "temp":   float(latest_parsed["temp"]) if not np.isnan(latest_parsed["temp"]) else 25.0,
        "dew":    float(latest_parsed["dew"])  if not np.isnan(latest_parsed["dew"])  else 10.0,
        "fog":    float(latest_parsed["fog"]),
        "mist":   float(latest_parsed["mist"]),
        "hour":   float(hour),
        "month":  float(month),
        "night":  float(int(hour >= 18 or hour <= 6)),
        "winter": float(int(month in [11, 12, 1, 2])),
        **{k: float(v) if v is not None else 0.0 for k, v in lag_features.items()}
    }

    import pandas as pd
    X = pd.DataFrame([row])[FEATURES]

    forecast = []
    for step in range(1, 25):
        pred_vis = float(models[step].predict(X)[0])
        forecast_time = latest_valid + timedelta(hours=step)
        forecast.append({
            "step": step,
            "time": forecast_time.strftime("%Y-%m-%d %H:%M"),
            "visibility": round(pred_vis)
        })

   
    current = {
        "vis":   int(latest_parsed["vis"])  if not np.isnan(latest_parsed["vis"])  else None,
        "wind":  int(latest_parsed["wind"]),
        "temp":  int(latest_parsed["temp"]) if not np.isnan(latest_parsed["temp"]) else None,
        "dew":   int(latest_parsed["dew"])  if not np.isnan(latest_parsed["dew"])  else None,
        "fog":   int(latest_parsed["fog"]),
        "mist":  int(latest_parsed["mist"]),
    }

    return jsonify({"forecast": forecast, "current": current})


if __name__ == "__main__":
    print("\n VIDP Forecast server starting...")
    print(" Make sure models_12h/ is in the same folder as this file")
    print(" Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
