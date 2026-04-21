"""
server.py – ESP32 Weather AI Server
Chạy liên tục trên Render. Luồng:
  Firebase RTDB /sensor/data (ESP32 ghi lên)
  → polling mỗi 30s
  → Open-Meteo dự báo thời tiết (không cần API key)
  → ai_core.predict()
  → ghi kết quả về Firebase RTDB /ai_result
  → ESP32 tự đọc /ai_result và hiển thị LCD
"""
import os, json, logging, threading, time
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db as firebase_db

from ai_core import predict as ai_predict

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("server")

# ─────────────────────────────────────────────────────────────────────────────
# ENV CONFIG
# ─────────────────────────────────────────────────────────────────────────────
_FB_CRED_JSON  = os.environ.get("FIREBASE_CRED_JSON", "")
_FB_DB_URL     = os.environ.get("FIREBASE_DB_URL", "https://your-project.firebaseio.com")
DEFAULT_LAT    = float(os.environ.get("DEFAULT_LAT", "21.0285"))
DEFAULT_LON    = float(os.environ.get("DEFAULT_LON", "105.8542"))
POLL_INTERVAL  = int(os.environ.get("POLL_INTERVAL", "30"))  # giây

# ─────────────────────────────────────────────────────────────────────────────
# Firebase init
# ─────────────────────────────────────────────────────────────────────────────
_fb_app = None

def _init_firebase():
    global _fb_app
    if not _FB_CRED_JSON:
        log.warning("FIREBASE_CRED_JSON not set – Firebase disabled")
        return
    try:
        cred_dict = json.loads(_FB_CRED_JSON)
        cred = credentials.Certificate(cred_dict)
        _fb_app = firebase_admin.initialize_app(cred, {"databaseURL": _FB_DB_URL})
        log.info("Firebase initialized OK")
    except Exception as e:
        log.error(f"Firebase init error: {e}")

_init_firebase()

def firebase_read(path: str):
    if not _fb_app:
        return None
    try:
        return firebase_db.reference(path).get()
    except Exception as e:
        log.warning(f"Firebase read error ({path}): {e}")
        return None

def firebase_write(path: str, data: dict):
    if not _fb_app:
        return
    try:
        firebase_db.reference(path).set(data)
    except Exception as e:
        log.warning(f"Firebase write error ({path}): {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Open-Meteo (không cần API key)
# ─────────────────────────────────────────────────────────────────────────────
_meteo_cache: dict = {}
_meteo_cache_ts: float = 0
METEO_CACHE_TTL = 900  # 15 phút

def get_meteo_forecast(lat=DEFAULT_LAT, lon=DEFAULT_LON) -> dict:
    global _meteo_cache, _meteo_cache_ts
    if time.time() - _meteo_cache_ts < METEO_CACHE_TTL and _meteo_cache:
        return _meteo_cache
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&hourly=precipitation_probability,precipitation"
            "&forecast_days=2"
            "&timezone=Asia%2FBangkok"
            "&timeformat=unixtime"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        raw = r.json()

        hourly    = raw["hourly"]
        times     = hourly["time"]
        prec_prob = hourly["precipitation_probability"]
        prec_mm   = hourly["precipitation"]

        now_ts  = time.time()
        cur_idx = 0
        for i, t in enumerate(times):
            if t <= now_ts:
                cur_idx = i

        result = {}
        for h in [1, 3, 6, 12]:
            idx = min(cur_idx + h, len(times) - 1)
            result[f"rain_prob_{h}h"]        = round(prec_prob[idx] / 100.0, 3)
            result[f"rain_forecast_{h}h_mm"] = round(prec_mm[idx], 2)

        _meteo_cache    = result
        _meteo_cache_ts = time.time()
        log.info(f"Open-Meteo OK: {result}")
        return result
    except Exception as e:
        log.error(f"Open-Meteo error: {e}")
        return _meteo_cache

# ─────────────────────────────────────────────────────────────────────────────
# History buffer cho lag features
# ─────────────────────────────────────────────────────────────────────────────
import collections
_history: collections.deque = collections.deque(maxlen=12)

def _build_lag_features() -> dict:
    h = list(_history)
    lags = {}
    if len(h) >= 1:
        lags.update(temp_lag1=h[-1]["temperature_c"], hum_lag1=h[-1]["humidity_rh"],
                    rain_lag1=h[-1].get("rain_actual", 0))
    if len(h) >= 3:
        lags.update(temp_lag3=h[-3]["temperature_c"], hum_lag3=h[-3]["humidity_rh"],
                    rain_lag3=h[-3].get("rain_actual", 0))
    if len(h) >= 6:
        lags["rain_lag6"] = sum(x.get("rain_actual", 0) for x in list(h)[-6:])
    if len(h) >= 3:
        last3 = list(h)[-3:]
        lags["temp_rolling3"] = sum(x["temperature_c"] for x in last3) / 3
        lags["hum_rolling3"]  = sum(x["humidity_rh"]   for x in last3) / 3
    if len(h) >= 6:
        lags["rain_rolling6"] = sum(x.get("rain_actual", 0) for x in list(h)[-6:])
    return lags

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline chính: đọc Firebase → AI → ghi lại Firebase
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline():
    """Đọc /sensor/data từ Firebase, chạy AI, ghi /ai_result"""

    # 1. Đọc dữ liệu cảm biến từ Firebase
    sensor = firebase_read("/sensor/data")
    if not sensor:
        log.warning("Chưa có dữ liệu cảm biến trong Firebase /sensor/data")
        return

    temp = float(sensor.get("temperature", 30))
    hum  = float(sensor.get("humidity", 60))
    log.info(f"Sensor from Firebase: T={temp}°C  H={hum}%")

    # 2. Lấy dự báo Open-Meteo
    meteo = get_meteo_forecast()

    # 3. Build input AI
    ai_input = {
        "temperature_c": temp,
        "humidity_rh":   hum,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        **meteo,
        **_build_lag_features(),
    }

    # 4. Chạy AI
    result = ai_predict(ai_input)

    # 5. Cập nhật history
    _history.append({
        "temperature_c": temp,
        "humidity_rh":   hum,
        "rain_actual":   result["rain_predicted_mm"],
    })

    # 6. Ghi kết quả về Firebase /ai_result
    firebase_write("/ai_result", {
        **result,
        "sensor": {"temperature": temp, "humidity": hum},
        "updated": datetime.utcnow().isoformat(),
    })

    log.info(
        f"AI done → rain={result['rain_probability']:.0%} "
        f"water={result['water_per_hour_ml']}ml/h "
        f"lcd1={result['lcd']['line1']} | lcd2={result['lcd']['line2']}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Background polling thread
# ─────────────────────────────────────────────────────────────────────────────
def _polling_loop():
    log.info(f"Polling Firebase mỗi {POLL_INTERVAL}s...")
    while True:
        try:
            run_pipeline()
        except Exception as e:
            log.error(f"Pipeline error: {e}")
        time.sleep(POLL_INTERVAL)

def start_polling():
    t = threading.Thread(target=_polling_loop, daemon=True)
    t.start()
    log.info("Polling thread started")

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service":  "ESP32 Weather AI Server",
        "flow":     "ESP32 → Firebase /sensor/data → Server → AI → Firebase /ai_result → ESP32",
        "poll_interval_s": POLL_INTERVAL,
        "endpoints": {
            "GET  /health":  "Status server + Firebase",
            "GET  /result":  "Kết quả AI mới nhất",
            "POST /trigger": "Chạy pipeline ngay lập tức (debug)",
        }
    })

@app.route("/health", methods=["GET"])
def health():
    sensor = firebase_read("/sensor/data")
    return jsonify({
        "status":          "ok",
        "firebase":        "enabled" if _fb_app else "disabled",
        "last_sensor":     sensor,
        "poll_interval_s": POLL_INTERVAL,
        "utc":             datetime.utcnow().isoformat(),
    })

@app.route("/result", methods=["GET"])
def get_result():
    """Xem kết quả AI mới nhất"""
    data = firebase_read("/ai_result")
    if data:
        return jsonify(data), 200
    return jsonify({"message": "Chưa có kết quả AI"}), 204

@app.route("/trigger", methods=["POST"])
def trigger():
    """Chạy pipeline ngay lập tức, không cần chờ poll"""
    try:
        run_pipeline()
        result = firebase_read("/ai_result")
        return jsonify({"status": "ok", "result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start_polling()
    port = int(os.environ.get("PORT", 5000))
    log.info(f"Flask on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
