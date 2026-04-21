"""
server.py – ESP32 Weather AI Server
Chạy liên tục trên Render. Luồng:
  Firebase RTDB (dự báo OWM) + MQTT (DHT22 từ ESP32)
  → ai_core.predict()
  → MQTT publish → ESP32 LCD
"""
import os, json, logging, threading, time
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify
import paho.mqtt.client as mqtt
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
# ENV CONFIG  (đặt trong Render Dashboard → Environment)
# ─────────────────────────────────────────────────────────────────────────────
MQTT_BROKER   = os.environ.get("MQTT_BROKER",   "broker.hivemq.com")
MQTT_PORT     = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USER     = os.environ.get("MQTT_USER",     "")
MQTT_PASS     = os.environ.get("MQTT_PASS",     "")

# Topics
TOPIC_SENSOR  = os.environ.get("TOPIC_SENSOR",  "esp32/dht22")    # ESP32 → Server
TOPIC_LCD     = os.environ.get("TOPIC_LCD",     "esp32/lcd")       # Server → ESP32
TOPIC_STATUS  = os.environ.get("TOPIC_STATUS",  "esp32/status")

OWM_API_KEY   = os.environ.get("OWM_API_KEY",   "")
DEFAULT_LAT   = float(os.environ.get("DEFAULT_LAT", "21.0285"))
DEFAULT_LON   = float(os.environ.get("DEFAULT_LON", "105.8542"))

# Firebase – set FIREBASE_CRED_JSON = JSON string của service account
_FB_CRED_JSON = os.environ.get("FIREBASE_CRED_JSON", "")
_FB_DB_URL    = os.environ.get("FIREBASE_DB_URL",  "https://your-project.firebaseio.com")

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
        log.info("Firebase initialized")
    except Exception as e:
        log.error(f"Firebase init error: {e}")

_init_firebase()

def firebase_read(path: str) -> dict | None:
    """Đọc node Firebase RTDB"""
    if not _fb_app:
        return None
    try:
        return firebase_db.reference(path).get()
    except Exception as e:
        log.warning(f"Firebase read error ({path}): {e}")
        return None

def firebase_write(path: str, data: dict):
    """Ghi kết quả AI vào Firebase"""
    if not _fb_app:
        return
    try:
        firebase_db.reference(path).set(data)
    except Exception as e:
        log.warning(f"Firebase write error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# OpenWeatherMap helper
# ─────────────────────────────────────────────────────────────────────────────
_owm_cache: dict = {}
_owm_cache_ts: float = 0
OWM_CACHE_TTL = 900  # 15 phút

def get_owm_forecast(lat=DEFAULT_LAT, lon=DEFAULT_LON) -> dict:
    global _owm_cache, _owm_cache_ts
    if time.time() - _owm_cache_ts < OWM_CACHE_TTL and _owm_cache:
        return _owm_cache

    if not OWM_API_KEY:
        log.warning("OWM_API_KEY not set")
        return {}

    try:
        url = (f"https://api.openweathermap.org/data/3.0/onecall"
               f"?lat={lat}&lon={lon}&exclude=minutely,daily,alerts"
               f"&appid={OWM_API_KEY}&units=metric")
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        raw = r.json()
        hourly = raw.get("hourly", [])
        result = {}
        for h in [1, 3, 6, 12]:
            if h < len(hourly):
                item = hourly[h]
                wid = item.get("weather", [{}])[0].get("id", 800)
                pop = item.get("pop", 0)
                result[f"rain_prob_{h}h"]         = round(pop, 3)
                result[f"rain_forecast_{h}h_mm"]  = round(
                    item.get("rain", {}).get("1h", 0), 2
                )
        _owm_cache    = result
        _owm_cache_ts = time.time()
        log.info(f"OWM fetched: {result}")
        return result
    except Exception as e:
        log.error(f"OWM error: {e}")
        return _owm_cache  # trả cache cũ nếu lỗi

# ─────────────────────────────────────────────────────────────────────────────
# History buffer (lag features)
# ─────────────────────────────────────────────────────────────────────────────
import collections
_history: collections.deque = collections.deque(maxlen=12)

def _build_lag_features() -> dict:
    lags = {}
    h = list(_history)
    # lag1
    if len(h) >= 1:
        lags["temp_lag1"] = h[-1]["temperature_c"]
        lags["hum_lag1"]  = h[-1]["humidity_rh"]
        lags["rain_lag1"] = h[-1].get("rain_actual", 0)
    # lag3
    if len(h) >= 3:
        lags["temp_lag3"] = h[-3]["temperature_c"]
        lags["hum_lag3"]  = h[-3]["humidity_rh"]
        lags["rain_lag3"] = h[-3].get("rain_actual", 0)
    # lag6
    if len(h) >= 6:
        lags["rain_lag6"] = sum(x.get("rain_actual", 0) for x in list(h)[-6:])
    # rolling
    if len(h) >= 3:
        last3 = list(h)[-3:]
        lags["temp_rolling3"] = sum(x["temperature_c"] for x in last3) / 3
        lags["hum_rolling3"]  = sum(x["humidity_rh"]   for x in last3) / 3
    if len(h) >= 6:
        lags["rain_rolling6"] = sum(x.get("rain_actual", 0) for x in list(h)[-6:])
    return lags

# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline: sensor_data → AI → MQTT publish
# ─────────────────────────────────────────────────────────────────────────────
def process_and_publish(sensor_payload: dict, mqtt_client=None):
    """
    sensor_payload: {"temperature": float, "humidity": float}
    """
    temp = float(sensor_payload.get("temperature", 30))
    hum  = float(sensor_payload.get("humidity", 60))

    # 1. Lấy dự báo OWM (cache 15 phút)
    owm = get_owm_forecast()

    # 2. Đọc thêm từ Firebase (nếu có dữ liệu bổ sung)
    fb_extra = firebase_read("/weather/owm") or {}

    # 3. Build input cho AI
    ai_input = {
        "temperature_c": temp,
        "humidity_rh":   hum,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        **owm,
        **fb_extra,         # Firebase ghi đè nếu có
        **_build_lag_features(),
    }

    # 4. Chạy AI
    result = ai_predict(ai_input)

    # 5. Ghi vào history
    _history.append({
        "temperature_c": temp,
        "humidity_rh":   hum,
        "rain_actual":   result["rain_predicted_mm"],
    })

    # 6. Ghi kết quả lên Firebase
    firebase_write("/ai_result", {
        **result,
        "sensor": {"temperature": temp, "humidity": hum},
        "updated": datetime.utcnow().isoformat(),
    })

    # 7. Publish MQTT về ESP32
    lcd_payload = json.dumps({
        "line1": result["lcd"]["line1"],
        "line2": result["lcd"]["line2"],
        "water_3h": result["water_3h_ml"],
        "rain_warn": result["rain_warning"],
        "rain_prob": round(result["rain_probability"] * 100),
    })
    if mqtt_client and mqtt_client.is_connected():
        mqtt_client.publish(TOPIC_LCD, lcd_payload, qos=1)
        log.info(f"MQTT → {TOPIC_LCD}: {lcd_payload}")
    else:
        log.warning("MQTT not connected – skipping publish")

    log.info(f"AI result: rain={result['rain_probability']:.2%} "
             f"water={result['water_per_hour_ml']}ml/h "
             f"lcd1={result['lcd']['line1']}")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MQTT client
# ─────────────────────────────────────────────────────────────────────────────
_mqtt_client: mqtt.Client = None

def _on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        log.info(f"MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(TOPIC_SENSOR, qos=1)
        log.info(f"Subscribed to {TOPIC_SENSOR}")
    else:
        log.error(f"MQTT connect failed rc={rc}")

def _on_message(client, userdata, msg):
    log.info(f"MQTT ← {msg.topic}: {msg.payload.decode()}")
    try:
        payload = json.loads(msg.payload.decode())
        process_and_publish(payload, client)
    except Exception as e:
        log.error(f"MQTT message error: {e}")

def _on_disconnect(client, userdata, rc, properties=None):
    log.warning(f"MQTT disconnected rc={rc} – reconnecting...")

def start_mqtt():
    global _mqtt_client
    client = mqtt.Client(
        client_id=f"render-server-{os.getpid()}",
        protocol=mqtt.MQTTv5,
    )
    client.on_connect    = _on_connect
    client.on_message    = _on_message
    client.on_disconnect = _on_disconnect
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.reconnect_delay_set(min_delay=2, max_delay=60)

    def _run():
        while True:
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                client.loop_forever()
            except Exception as e:
                log.error(f"MQTT connection error: {e} – retry in 10s")
                time.sleep(10)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    _mqtt_client = client
    return client

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "ESP32 Weather AI Server",
        "mqtt_broker": MQTT_BROKER,
        "topics": {
            "subscribe": TOPIC_SENSOR,
            "publish":   TOPIC_LCD,
        },
        "endpoints": {
            "POST /sensor": "Gửi dữ liệu thủ công (fallback HTTP)",
            "GET  /result": "Xem kết quả AI mới nhất",
            "GET  /health": "Health check",
        }
    })

@app.route("/health", methods=["GET"])
def health():
    mqtt_ok = _mqtt_client is not None and _mqtt_client.is_connected()
    return jsonify({
        "status":   "ok",
        "mqtt":     "connected" if mqtt_ok else "disconnected",
        "firebase": "enabled" if _fb_app else "disabled",
        "utc":      datetime.utcnow().isoformat(),
    })

@app.route("/sensor", methods=["POST"])
def http_sensor():
    """Fallback: ESP32 gửi HTTP thay MQTT"""
    body = request.get_json(silent=True) or {}
    if "temperature" not in body or "humidity" not in body:
        return jsonify({"error": "Cần temperature và humidity"}), 400
    try:
        result = process_and_publish(body, _mqtt_client)
        return jsonify({"status": "ok", "result": result}), 200
    except Exception as e:
        log.error(f"/sensor error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/result", methods=["GET"])
def get_result():
    """Xem kết quả AI mới nhất từ Firebase"""
    data = firebase_read("/ai_result")
    if data:
        return jsonify(data), 200
    return jsonify({"message": "No data yet"}), 204

@app.route("/mqtt/publish", methods=["POST"])
def manual_publish():
    """Debug: publish thủ công bất kỳ message lên MQTT"""
    body  = request.get_json(silent=True) or {}
    topic = body.get("topic", TOPIC_LCD)
    msg   = body.get("message", "{}")
    if _mqtt_client and _mqtt_client.is_connected():
        _mqtt_client.publish(topic, msg, qos=1)
        return jsonify({"status": "published", "topic": topic}), 200
    return jsonify({"error": "MQTT not connected"}), 503


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting MQTT listener...")
    start_mqtt()
    port = int(os.environ.get("PORT", 5000))
    log.info(f"Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
