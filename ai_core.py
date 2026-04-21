"""
ai_core.py – AI Inference Engine
Tải model từ models/ và cung cấp hàm predict() cho server.
"""
import json
import os
import numpy as np
import joblib
from datetime import datetime

# ── Đường dẫn model ───────────────────────────────────────────────────────────
_BASE  = os.path.dirname(os.path.abspath(__file__))
_MDIR  = os.path.join(_BASE, "models")

# ── Load models (một lần khi import) ─────────────────────────────────────────
with open(os.path.join(_MDIR, "model_meta.json")) as f:
    META = json.load(f)

CLF       = joblib.load(os.path.join(_MDIR, META["models"]["rain_clf"]))
REG       = joblib.load(os.path.join(_MDIR, META["models"]["rain_reg"]))
WATER_MDL = joblib.load(os.path.join(_MDIR, META["models"]["water"]))
SCALER    = joblib.load(os.path.join(_MDIR, META["models"]["scaler"]))
SCALER_W  = joblib.load(os.path.join(_MDIR, META["models"]["scaler_water"]))

FEATURES     = META["features"]
WATER_FEAT   = META["water_features"]
RAIN_WARN_P  = META["rain_warning_prob"]   # 0.40
WATER_MIN    = META["water_clip_min"]       # 50
WATER_MAX    = META["water_clip_max"]       # 500


# ── Feature builder ───────────────────────────────────────────────────────────
def _build_features(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Chuyển dict dữ liệu đầu vào → 2 vector:
      X_full  : dùng cho clf + reg  (len = len(FEATURES))
      X_water : dùng cho water_model (len = len(WATER_FEAT))
    """
    now   = datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat()))
    hour  = now.hour
    month = now.month
    temp  = float(data["temperature_c"])
    hum   = float(data["humidity_rh"])

    # Cyclic time
    hour_sin  = np.sin(2 * np.pi * hour  / 24)
    hour_cos  = np.cos(2 * np.pi * hour  / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Heat index
    heat_index = temp + 0.33 * (
        hum / 100 * 6.105 * np.exp(17.27 * temp / (237.7 + temp))
    ) - 4

    # Lag/rolling: dùng giá trị hiện tại làm fallback
    def g(key, default=None):
        return float(data.get(key, default if default is not None else data.get(key.split("_")[0], 0)))

    row = {
        "temperature_c":        temp,
        "humidity_rh":          hum,
        "rain_prob_1h":         float(data.get("rain_prob_1h",  0)),
        "rain_prob_3h":         float(data.get("rain_prob_3h",  0)),
        "rain_prob_6h":         float(data.get("rain_prob_6h",  0)),
        "rain_prob_12h":        float(data.get("rain_prob_12h", 0)),
        "rain_forecast_1h_mm":  float(data.get("rain_forecast_1h_mm",  0)),
        "rain_forecast_3h_mm":  float(data.get("rain_forecast_3h_mm",  0)),
        "rain_forecast_6h_mm":  float(data.get("rain_forecast_6h_mm",  0)),
        "rain_forecast_12h_mm": float(data.get("rain_forecast_12h_mm", 0)),
        "temp_lag1":   float(data.get("temp_lag1",  temp)),
        "temp_lag3":   float(data.get("temp_lag3",  temp)),
        "hum_lag1":    float(data.get("hum_lag1",   hum)),
        "hum_lag3":    float(data.get("hum_lag3",   hum)),
        "rain_lag1":   float(data.get("rain_lag1",  0)),
        "rain_lag3":   float(data.get("rain_lag3",  0)),
        "rain_lag6":   float(data.get("rain_lag6",  0)),
        "temp_rolling3": float(data.get("temp_rolling3", temp)),
        "hum_rolling3":  float(data.get("hum_rolling3",  hum)),
        "rain_rolling6": float(data.get("rain_rolling6", 0)),
        "hour_sin":   hour_sin,  "hour_cos":   hour_cos,
        "month_sin":  month_sin, "month_cos":  month_cos,
        "heat_index": heat_index,
    }

    X_full  = np.array([[row[f] for f in FEATURES]])
    X_water = np.array([[row[f] for f in WATER_FEAT]])
    return X_full, X_water


# ── LCD formatter ─────────────────────────────────────────────────────────────
def _fmt_lcd(rain_prob: float, rain_mm: float,
             water_per_h: int, hours: int = 3) -> dict:
    water_for_period = water_per_h * hours
    # Round to nearest 50ml
    water_for_period = int(round(water_for_period / 50) * 50)

    line1 = f"Nuoc:{water_for_period}ml/{hours}h"
    if rain_prob >= RAIN_WARN_P:
        prob_pct = round(rain_prob * 100)
        line2 = f"AoMua!{prob_pct}% {rain_mm:.1f}mm"
    elif water_per_h >= 150:
        line2 = f"Nong! UV cao!"
    else:
        line2 = f"T/t:{round(rain_prob*100)}%mua OK"

    return {
        "line1": line1[:16],
        "line2": line2[:16],
    }


# ── Main predict function ─────────────────────────────────────────────────────
def predict(data: dict) -> dict:
    """
    Input : dict với các key theo schema bên dưới
    Output: dict kết quả AI đầy đủ gồm lcd text, water, rain
    
    Schema tối thiểu:
        temperature_c, humidity_rh
    Schema đầy đủ (nếu có):
        rain_prob_1h..12h, rain_forecast_*_mm,
        temp_lag1/3, hum_lag1/3, rain_lag1/3/6,
        temp_rolling3, hum_rolling3, rain_rolling6,
        timestamp (ISO string)
    """
    X_full, X_water = _build_features(data)

    X_full_s  = SCALER.transform(X_full)
    X_water_s = SCALER_W.transform(X_water)

    # 1. Rain classification
    rain_prob    = float(CLF.predict_proba(X_full_s)[0][1])
    rain_warning = rain_prob >= RAIN_WARN_P

    # 2. Rain amount (mm)
    rain_mm = float(np.clip(REG.predict(X_full_s)[0], 0, 200))

    # 3. Water per hour
    water_per_h = int(np.clip(WATER_MDL.predict(X_water_s)[0], WATER_MIN, WATER_MAX))
    # Round to 50ml
    water_per_h = int(round(water_per_h / 50) * 50)

    # 4. LCD text
    lcd = _fmt_lcd(rain_prob, rain_mm, water_per_h, hours=3)

    return {
        "rain_probability":  round(rain_prob, 4),
        "rain_warning":      rain_warning,
        "rain_predicted_mm": round(rain_mm, 2),
        "water_per_hour_ml": water_per_h,
        "water_3h_ml":       water_per_h * 3,
        "water_6h_ml":       water_per_h * 6,
        "lcd":               lcd,
        "model_version":     META["version"],
        "timestamp":         datetime.utcnow().isoformat(),
    }


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test = {
        "temperature_c": 33.5, "humidity_rh": 72.0,
        "rain_prob_1h": 0.65,  "rain_prob_3h": 0.55,
        "rain_prob_6h": 0.35,  "rain_prob_12h": 0.20,
        "rain_forecast_1h_mm": 2.5, "rain_forecast_3h_mm": 4.0,
        "rain_forecast_6h_mm": 1.0, "rain_forecast_12h_mm": 0.0,
        "timestamp": "2024-07-15T14:00:00",
    }
    result = predict(test)
    print(json.dumps(result, indent=2))
    print(f"\n📺 LCD:")
    print(f"  ┌{'─'*16}┐")
    print(f"  │{result['lcd']['line1']:<16}│")
    print(f"  │{result['lcd']['line2']:<16}│")
    print(f"  └{'─'*16}┘")
