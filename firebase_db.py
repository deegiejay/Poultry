"""
firebase_db.py  —  Shared Firebase Realtime Database Layer
============================================================
Used by BOTH:
  • cloud_ml_server.py  (writes predictions / alerts)
  • fletapp.py          (reads everything, pure UI)

Pure REST — no Firebase SDK. Works on Android APK, cloud server, desktop.

SETUP (5 min, one-time):
  1. https://console.firebase.google.com → New project "poultry-farm-ai"
  2. Build → Realtime Database → Create → Start in TEST mode
  3. Copy URL:  https://poultry-farm-ai-default-rtdb.firebaseio.com
  4. Set FIREBASE_URL below (same value in ALL files)

Firebase structure:
  /latest          → single node, overwritten by ESP every 2s  (live display)
  /readings        → append-only log, one child per ESP POST   (history + ML)
  /ml_result       → single node, overwritten by cloud ML      (current prediction)
  /forecast_7d     → single node, overwritten by cloud ML      (7-day table)
  /alerts          → append-only alert log
  /ml_status       → single node, cloud ML heartbeat
"""

import requests
import time
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

# ══════════════════════════════════════════════════════════════════════════════
# ▶▶  SET YOUR FIREBASE URL HERE  (same value in all 3 files)
# ══════════════════════════════════════════════════════════════════════════════
FIREBASE_URL = "https://poultry-ai-e901a-default-rtdb.firebaseio.com/"
# ══════════════════════════════════════════════════════════════════════════════

TIMEOUT   = 6
CACHE_TTL = 20   # seconds before re-fetching readings from Firebase

# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE CACHE  (APK uses this when network is unavailable)
# ─────────────────────────────────────────────────────────────────────────────
_cache: Dict[str, Any] = {
    "latest":       None,
    "readings":     [],
    "ml_result":    None,
    "forecast_7d":  None,
    "last_fetch":   0,
    "online":       False,
}
_lock = threading.Lock()


def _get(path: str, params: str = "") -> Optional[Any]:
    """GET from Firebase. Returns parsed JSON or None."""
    try:
        url = f"{FIREBASE_URL}/{path}.json{params}"
        r   = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            with _lock:
                _cache["online"] = True
            return r.json()
    except Exception:
        pass
    with _lock:
        _cache["online"] = False
    return None


def _put(path: str, payload: dict) -> bool:
    """PUT (overwrite) a node in Firebase."""
    try:
        url = f"{FIREBASE_URL}/{path}.json"
        r   = requests.put(url, json=payload, timeout=TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False


def _post(path: str, payload: dict) -> bool:
    """POST (append child) to Firebase."""
    try:
        url = f"{FIREBASE_URL}/{path}.json"
        r   = requests.post(url, json=payload, timeout=TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# READ  (used by APK + cloud server)
# ═════════════════════════════════════════════════════════════════════════════

def get_latest() -> Optional[Dict]:
    """Fetch /latest — single fast read for live dashboard."""
    data = _get("latest")
    if data:
        with _lock:
            _cache["latest"] = data
        return data
    with _lock:
        return _cache.get("latest")


def get_readings(limit: int = 1000) -> List[Dict]:
    """
    Fetch last N readings from /readings ordered by timestamp.
    Cached locally for CACHE_TTL seconds.
    """
    now = time.time()
    with _lock:
        cached   = _cache["readings"]
        last_f   = _cache["last_fetch"]
    if cached and (now - last_f) < CACHE_TTL:
        return cached

    raw = _get("readings", f'?orderBy="timestamp"&limitToLast={limit}')
    if raw and isinstance(raw, dict):
        readings = list(raw.values())
        readings.sort(key=lambda x: x.get("timestamp", 0))
        with _lock:
            _cache["readings"]   = readings
            _cache["last_fetch"] = now
        return readings
    with _lock:
        return _cache.get("readings", [])


def get_reading_count() -> int:
    return len(get_readings(limit=5000))


def get_ml_result() -> Optional[Dict]:
    """
    Fetch the latest ML prediction written by the cloud server.
    Returns dict with keys:
      feedKg, waterL, predDate, confidence, trend, trendIcon,
      anomaly, anomalyMsg, feedDelta, waterDelta,
      arimaFeed, arimaWater, modelRows, trainedAt
    Falls back to cache.
    """
    data = _get("ml_result")
    if data:
        with _lock:
            _cache["ml_result"] = data
        return data
    with _lock:
        return _cache.get("ml_result")


def get_forecast_7d() -> Optional[List[Dict]]:
    """Fetch the 7-day forecast table written by cloud server."""
    data = _get("forecast_7d")
    if data:
        if isinstance(data, dict):
            data = list(data.values())
        with _lock:
            _cache["forecast_7d"] = data
        return data
    with _lock:
        return _cache.get("forecast_7d")


def get_ml_status() -> Optional[Dict]:
    """Fetch cloud ML server heartbeat."""
    return _get("ml_status")


def get_alerts(limit: int = 20) -> List[Dict]:
    """Fetch recent alerts."""
    raw = _get("alerts", f"?orderBy=\"ts\"&limitToLast={limit}")
    if raw and isinstance(raw, dict):
        alerts = list(raw.values())
        alerts.sort(key=lambda x: x.get("ts", ""), reverse=True)
        return alerts
    return []


def get_cache_status() -> Dict:
    with _lock:
        return {
            "online":       _cache["online"],
            "cached_rows":  len(_cache["readings"]),
            "has_latest":   _cache["latest"] is not None,
            "has_ml":       _cache["ml_result"] is not None,
        }


# ═════════════════════════════════════════════════════════════════════════════
# WRITE  (used by cloud_ml_server.py only — APK never writes ML data)
# ═════════════════════════════════════════════════════════════════════════════

def write_ml_result(payload: dict) -> bool:
    """Cloud server overwrites /ml_result with latest prediction."""
    payload["writtenAt"] = datetime.utcnow().isoformat()
    return _put("ml_result", payload)


def write_forecast_7d(rows: list) -> bool:
    """Cloud server overwrites /forecast_7d with new 7-day table."""
    # Store as numbered dict so Firebase accepts it
    data = {str(i): row for i, row in enumerate(rows)}
    return _put("forecast_7d", data)


def write_ml_status(status: str, rows: int, error: str = "") -> bool:
    """Cloud server writes heartbeat to /ml_status."""
    return _put("ml_status", {
        "status":    status,
        "rows":      rows,
        "error":     error,
        "updatedAt": datetime.utcnow().isoformat(),
    })


def push_alert(alert_type: str, message: str, value: float = 0.0) -> bool:
    """Append an alert to /alerts."""
    return _post("alerts", {
        "ts":      datetime.utcnow().isoformat(),
        "type":    alert_type,
        "message": message,
        "value":   value,
    })


# ═════════════════════════════════════════════════════════════════════════════
# CONVERSION HELPER  (used by cloud server to build training DataFrame)
# ═════════════════════════════════════════════════════════════════════════════

def readings_to_df(readings: List[Dict]):
    """Convert Firebase readings list → pandas DataFrame for ML training."""
    import pandas as pd
    if not readings:
        return pd.DataFrame()
    rows = []
    for rec in readings:
        try:
            rows.append({
                "date":         pd.to_datetime(rec.get("ts") or rec.get("timestamp", 0)),
                "feed_kg":      float(rec.get("weight",      0.0)),
                "water_liters": float(rec.get("totalLiters", 0.0)),
                "flow":         float(rec.get("flow",        0.0)),
                "level":        str(rec.get("level",         "N/A")),
                "day_of_week":  int(rec.get("dayOfWeek",     0)),
                "month":        int(rec.get("month",         1)),
                "system":       1,
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    try:
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
    except Exception:
        pass
    return df