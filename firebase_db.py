"""
firebase_db.py  —  Shared Firebase Realtime Database Layer
============================================================
Used by BOTH:
  • cloud_ml_server.py  (writes predictions / alerts)
  • fletapp.py          (reads everything, pure UI)

Pure REST — no Firebase SDK. Works on Android APK, cloud server, desktop.

SETUP (5 min, one-time):
  1. https://console.firebase.google.com → New project
  2. Build → Realtime Database → Create → Start in TEST mode
  3. Copy URL and set FIREBASE_URL below (same in ALL files)

Firebase structure:
  /latest       → overwritten by ESP every 2s        (live display)
  /readings     → append-only log per ESP POST        (history + ML)
  /ml_result    → overwritten by cloud ML             (current prediction)
  /forecast_7d  → overwritten by cloud ML             (7-day table)
  /alerts       → append-only alert log
  /ml_status    → cloud ML heartbeat
"""

import requests
import time
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

# ══════════════════════════════════════════════════════════════════════════════
# ▶▶  YOUR FIREBASE URL — matches your ESP32 sketch and Render env var
# ══════════════════════════════════════════════════════════════════════════════
FIREBASE_URL = "https://poultry-ai-e901a-default-rtdb.firebaseio.com"
# ══════════════════════════════════════════════════════════════════════════════

TIMEOUT   = 6
CACHE_TTL = 20   # seconds before re-fetching readings

# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE CACHE
# ─────────────────────────────────────────────────────────────────────────────
_cache: Dict[str, Any] = {
    "latest":      None,
    "readings":    [],
    "ml_result":   None,
    "forecast_7d": None,
    "last_fetch":  0,
    "online":      False,
}
_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL REST HELPERS
# NOTE: All helpers read FIREBASE_URL at call time so that
#       cloud_ml_server.py can override it via:
#           import firebase_db as db
#           db.FIREBASE_URL = os.getenv("FIREBASE_URL")
#       and the change takes effect immediately.
# ─────────────────────────────────────────────────────────────────────────────

def _base() -> str:
    """Always returns the current module-level FIREBASE_URL."""
    return FIREBASE_URL.rstrip("/")


def _get(path: str, params: str = "") -> Optional[Any]:
    """GET from Firebase. Returns parsed JSON or None on error."""
    try:
        url = f"{_base()}/{path}.json{params}"
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
    """PUT (overwrite) a Firebase node."""
    try:
        r = requests.put(f"{_base()}/{path}.json", json=payload, timeout=TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False


def _post(path: str, payload: dict) -> bool:
    """POST (append child) to a Firebase node."""
    try:
        r = requests.post(f"{_base()}/{path}.json", json=payload, timeout=TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# READ  (APK + cloud server)
# ═════════════════════════════════════════════════════════════════════════════

def get_latest() -> Optional[Dict]:
    """Single fast read of /latest for live dashboard."""
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
    Cached for CACHE_TTL seconds.
    Firebase requires the 'timestamp' index to be defined in rules for
    orderBy queries — if not set yet, results still come back unordered
    and we sort client-side.
    """
    now = time.time()
    with _lock:
        cached = _cache["readings"]
        last_f = _cache["last_fetch"]
    if cached and (now - last_f) < CACHE_TTL:
        return cached

    # Try ordered query first
    raw = _get("readings", f'?orderBy="timestamp"&limitToLast={limit}')
    if raw is None:
        # Fallback: fetch without ordering (works even without index rules)
        raw = _get("readings")

    if raw and isinstance(raw, dict):
        readings = list(raw.values())
        # Client-side sort by timestamp (Unix epoch float/int)
        readings.sort(key=lambda x: float(x.get("timestamp", 0)))
        # Apply limit client-side
        if len(readings) > limit:
            readings = readings[-limit:]
        with _lock:
            _cache["readings"]   = readings
            _cache["last_fetch"] = now
        return readings

    with _lock:
        return _cache.get("readings", [])


def get_reading_count() -> int:
    """Count readings via cached list (Firebase has no COUNT)."""
    return len(get_readings(limit=5000))


def get_ml_result() -> Optional[Dict]:
    """
    Fetch latest ML prediction from /ml_result.
    Keys: feedKg, waterL, predDate, confidence, trend, trendIcon,
          anomaly, anomalyMsg, feedDelta, waterDelta,
          arimaFeed, arimaWater, modelRows, trainedAt,
          patSystem, patDay, patMonth
    Falls back to cache on network error.
    """
    data = _get("ml_result")
    if data:
        with _lock:
            _cache["ml_result"] = data
        return data
    with _lock:
        return _cache.get("ml_result")


def get_forecast_7d() -> Optional[List[Dict]]:
    """Fetch 7-day forecast list from /forecast_7d."""
    data = _get("forecast_7d")
    if data:
        # Firebase stores as numbered dict {"0":{...},"1":{...}}
        if isinstance(data, dict):
            data = [data[k] for k in sorted(data.keys(), key=lambda x: int(x))]
        with _lock:
            _cache["forecast_7d"] = data
        return data
    with _lock:
        return _cache.get("forecast_7d")


def get_ml_status() -> Optional[Dict]:
    """Fetch cloud ML heartbeat from /ml_status."""
    return _get("ml_status")


def get_alerts(limit: int = 20) -> List[Dict]:
    """Fetch recent alerts from /alerts."""
    raw = _get("alerts", f'?orderBy="ts"&limitToLast={limit}')
    if raw and isinstance(raw, dict):
        alerts = list(raw.values())
        alerts.sort(key=lambda x: x.get("ts", ""), reverse=True)
        return alerts
    return []


def get_cache_status() -> Dict:
    with _lock:
        return {
            "online":      _cache["online"],
            "cached_rows": len(_cache["readings"]),
            "has_latest":  _cache["latest"] is not None,
            "has_ml":      _cache["ml_result"] is not None,
        }


# ═════════════════════════════════════════════════════════════════════════════
# WRITE  (cloud_ml_server.py only — APK never writes ML data)
# ═════════════════════════════════════════════════════════════════════════════

def write_ml_result(payload: dict) -> bool:
    payload["writtenAt"] = datetime.utcnow().isoformat()
    return _put("ml_result", payload)


def write_forecast_7d(rows: list) -> bool:
    data = {str(i): row for i, row in enumerate(rows)}
    return _put("forecast_7d", data)


def write_ml_status(status: str, rows: int, error: str = "") -> bool:
    return _put("ml_status", {
        "status":    status,
        "rows":      rows,
        "error":     error,
        "updatedAt": datetime.utcnow().isoformat(),
    })


def push_alert(alert_type: str, message: str, value: float = 0.0) -> bool:
    return _post("alerts", {
        "ts":      datetime.utcnow().isoformat(),
        "type":    alert_type,
        "message": message,
        "value":   value,
    })


# ═════════════════════════════════════════════════════════════════════════════
# CONVERSION  (cloud server uses this for ML training)
# ═════════════════════════════════════════════════════════════════════════════

def readings_to_df(readings: List[Dict]):
    """
    Convert Firebase readings list → pandas DataFrame for ML training.

    ESP32 sends:
      timestamp  → Unix epoch seconds (integer from time(nullptr))
      ts         → ISO string "2026-04-20T12:34:56"
      weight     → feed_kg
      totalLiters→ water_liters
      flow, level, dayOfWeek, month

    FIX vs previous version:
      - Parse timestamp as Unix epoch (unit='s') first, fall back to ISO ts
      - Handles both int and float timestamps
    """
    import pandas as pd

    if not readings:
        return pd.DataFrame()

    rows = []
    for rec in readings:
        try:
            # Try Unix epoch timestamp first (what ESP sends via time(nullptr))
            ts_val = rec.get("timestamp")
            if ts_val and float(ts_val) > 1_000_000:          # looks like epoch
                date = pd.to_datetime(float(ts_val), unit="s")
            else:
                # Fallback: ISO string from "ts" field
                ts_iso = rec.get("ts", "")
                date = pd.to_datetime(ts_iso) if ts_iso else pd.Timestamp.now()

            rows.append({
                "date":         date,
                "feed_kg":      float(rec.get("weight",       0.0)),
                "water_liters": float(rec.get("totalLiters",  0.0)),
                "flow":         float(rec.get("flow",         0.0)),
                "level":        str(rec.get("level",          "0%")),
                "day_of_week":  int(rec.get("dayOfWeek",      0)),
                "month":        int(rec.get("month",          1)),
                "system":       1,
            })
        except Exception as e:
            print(f"[DB] skip row: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # Strip timezone if present (matplotlib + pandas tz-aware = crash)
    try:
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
    except Exception:
        pass

    return df