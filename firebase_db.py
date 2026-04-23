"""
firebase_db.py  —  Firebase Realtime Database Layer
=====================================================
Shared by cloud_ml_server.py and fletapp.py.

ROOT CAUSE FIX:
  Firebase orderBy="timestamp" requires an index rule. Without it,
  Firebase returns HTTP 400, our code got 0 rows, ML server stayed stuck.

  FIX 1: get_reading_count() now uses ?shallow=true (counts keys only,
          never times out, needs no index rule).
  FIX 2: get_readings() fetches without orderBy first, sorts client-side.
          Falls back gracefully. Never returns 0 when data exists.

ARCHIVE SYSTEM:
  /readings → active window (last ACTIVE_LIMIT rows)
  /archive  → cold storage (older rows moved here automatically)
  Cloud ML trains only on /readings (fast, bounded size).
  APK can show /archive for history if needed.

FIREBASE RULES (paste in Firebase Console → Rules):
{
  "rules": {
    ".read": true,
    ".write": true,
    "readings": {
      ".indexOn": ["timestamp"]
    },
    "archive": {
      ".indexOn": ["timestamp"]
    }
  }
}
"""

import requests
import time
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — set FIREBASE_URL to your project URL
# ══════════════════════════════════════════════════════════════════════════════
FIREBASE_URL  = "https://poultry-ai-e901a-default-rtdb.firebaseio.com"
ACTIVE_LIMIT  = 500     # max rows kept in /readings; older rows → /archive
ARCHIVE_BATCH = 100     # rows moved per archive sweep

TIMEOUT   = 8
CACHE_TTL = 15

# ──────────────────────────────────────────────────────────────────────────────
# OFFLINE CACHE
# ──────────────────────────────────────────────────────────────────────────────
_cache: Dict[str, Any] = {
    "latest":      None,
    "readings":    [],
    "ml_result":   None,
    "forecast_7d": None,
    "last_fetch":  0,
    "online":      False,
    "count":       0,
    "count_ts":    0,
}
_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS  (_base() reads FIREBASE_URL live — safe to patch from env)
# ──────────────────────────────────────────────────────────────────────────────

def _base() -> str:
    return FIREBASE_URL.rstrip("/")


def _get(path: str, params: str = "") -> Optional[Any]:
    try:
        r = requests.get(f"{_base()}/{path}.json{params}", timeout=TIMEOUT)
        if r.status_code == 200:
            with _lock:
                _cache["online"] = True
            return r.json()
        # Log non-200 for debugging
        print(f"[DB] GET {path} → HTTP {r.status_code}: {r.text[:120]}")
    except Exception as e:
        print(f"[DB] GET {path} error: {e}")
    with _lock:
        _cache["online"] = False
    return None


def _put(path: str, payload: dict) -> bool:
    try:
        r = requests.put(f"{_base()}/{path}.json", json=payload, timeout=TIMEOUT)
        if r.status_code != 200:
            print(f"[DB] PUT {path} → HTTP {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        print(f"[DB] PUT {path} error: {e}")
        return False


def _post(path: str, payload: dict) -> bool:
    try:
        r = requests.post(f"{_base()}/{path}.json", json=payload, timeout=TIMEOUT)
        return r.status_code == 200
    except Exception as e:
        print(f"[DB] POST {path} error: {e}")
        return False


def _delete(path: str) -> bool:
    try:
        r = requests.delete(f"{_base()}/{path}.json", timeout=TIMEOUT)
        return r.status_code == 200
    except Exception as e:
        print(f"[DB] DELETE {path} error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# COUNT  (uses shallow=true — fast, no index needed, no data downloaded)
# ══════════════════════════════════════════════════════════════════════════════

def get_reading_count() -> int:
    """
    Count /readings entries using shallow=true.
    Returns only the keys (no values), so it's fast and never times out.
    Does NOT require a Firebase index rule.
    Cached for 10 seconds.
    """
    now = time.time()
    with _lock:
        cached_count = _cache["count"]
        count_ts     = _cache["count_ts"]

    if cached_count > 0 and (now - count_ts) < 10:
        return cached_count

    raw = _get("readings", "?shallow=true")
    if raw and isinstance(raw, dict):
        count = len(raw)
        with _lock:
            _cache["count"]    = count
            _cache["count_ts"] = now
        return count

    # Fallback: use cached readings list length
    with _lock:
        return len(_cache.get("readings", []))


# ══════════════════════════════════════════════════════════════════════════════
# READ
# ══════════════════════════════════════════════════════════════════════════════

def get_latest() -> Optional[Dict]:
    """Read /latest — single node, always fast."""
    data = _get("latest")
    if data:
        with _lock:
            _cache["latest"] = data
        return data
    with _lock:
        return _cache.get("latest")


def get_readings(limit: int = 500) -> List[Dict]:
    """
    Fetch last N readings from /readings.

    Strategy (most reliable first):
      1. orderBy="timestamp"&limitToLast=N  (requires index rule — fast)
      2. Full fetch + client-side sort + slice  (always works, no index needed)
      3. Return cache if both fail

    Client-side sort always applied so order is correct regardless.
    """
    now = time.time()
    with _lock:
        cached = _cache["readings"]
        last_f = _cache["last_fetch"]
    if cached and (now - last_f) < CACHE_TTL:
        return cached

    readings = None

    # ── Strategy 1: ordered query (needs Firebase index rule) ─────────────────
    raw = _get("readings", f'?orderBy="timestamp"&limitToLast={limit}')
    if raw and isinstance(raw, dict):
        readings = list(raw.values())

    # ── Strategy 2: full fetch (always works) ─────────────────────────────────
    if readings is None:
        print("[DB] Falling back to full /readings fetch (no index rule set)")
        raw = _get("readings")
        if raw and isinstance(raw, dict):
            readings = list(raw.values())

    if readings:
        # Always sort client-side — covers both strategies
        try:
            readings.sort(key=lambda x: float(x.get("timestamp", 0)))
        except Exception:
            pass
        # Apply limit
        if len(readings) > limit:
            readings = readings[-limit:]
        with _lock:
            _cache["readings"]   = readings
            _cache["last_fetch"] = now
        return readings

    # ── Strategy 3: return cache ───────────────────────────────────────────────
    with _lock:
        return _cache.get("readings", [])


def get_ml_result() -> Optional[Dict]:
    data = _get("ml_result")
    if data:
        with _lock:
            _cache["ml_result"] = data
        return data
    with _lock:
        return _cache.get("ml_result")


def get_forecast_7d() -> Optional[List[Dict]]:
    data = _get("forecast_7d")
    if data:
        if isinstance(data, dict):
            data = [data[k] for k in sorted(data.keys(), key=lambda x: int(x))]
        with _lock:
            _cache["forecast_7d"] = data
        return data
    with _lock:
        return _cache.get("forecast_7d")


def get_ml_status() -> Optional[Dict]:
    return _get("ml_status")


def get_cache_status() -> Dict:
    with _lock:
        return {
            "online":      _cache["online"],
            "cached_rows": len(_cache["readings"]),
            "has_latest":  _cache["latest"] is not None,
            "has_ml":      _cache["ml_result"] is not None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ARCHIVE  (/readings → /archive when count > ACTIVE_LIMIT)
# ══════════════════════════════════════════════════════════════════════════════

def archive_old_readings() -> int:
    """
    Move oldest rows from /readings to /archive when count > ACTIVE_LIMIT.
    Returns number of rows archived.
    Called by cloud ML server after each training cycle.

    Strategy:
      1. Get all /readings keys + timestamps via shallow + small fetch
      2. Find oldest ARCHIVE_BATCH keys
      3. Write them to /archive (POST each)
      4. Delete them from /readings
    """
    count = get_reading_count()
    if count <= ACTIVE_LIMIT:
        return 0

    excess = count - ACTIVE_LIMIT
    to_archive = min(excess + ARCHIVE_BATCH, excess * 2)  # archive a bit extra
    print(f"[ARCHIVE] {count} readings > {ACTIVE_LIMIT} limit. Archiving {to_archive} oldest…")

    # Fetch all keys with timestamps (use limitToFirst to get oldest)
    raw = _get("readings", f'?orderBy="timestamp"&limitToFirst={to_archive}')
    if not raw or not isinstance(raw, dict):
        # Fallback: fetch full and slice
        raw = _get("readings")
        if not raw or not isinstance(raw, dict):
            print("[ARCHIVE] Could not fetch readings for archiving")
            return 0
        # Sort and take oldest
        items = sorted(raw.items(), key=lambda kv: float(kv[1].get("timestamp", 0)))
        raw = dict(items[:to_archive])

    archived = 0
    for key, val in raw.items():
        if _post("archive", val):          # POST to /archive (new key)
            if _delete(f"readings/{key}"): # DELETE from /readings
                archived += 1

    # Invalidate cache
    with _lock:
        _cache["readings"]   = []
        _cache["last_fetch"] = 0
        _cache["count"]      = 0

    print(f"[ARCHIVE] Done — archived {archived} rows")
    return archived


# ══════════════════════════════════════════════════════════════════════════════
# WRITE  (cloud server only)
# ══════════════════════════════════════════════════════════════════════════════

def write_ml_result(payload: dict) -> bool:
    payload["writtenAt"] = datetime.utcnow().isoformat()
    ok = _put("ml_result", payload)
    if not ok:
        print("[DB] ⚠️  write_ml_result FAILED")
    return ok


def write_forecast_7d(rows: list) -> bool:
    data = {str(i): row for i, row in enumerate(rows)}
    ok = _put("forecast_7d", data)
    if not ok:
        print("[DB] ⚠️  write_forecast_7d FAILED")
    return ok


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


# ══════════════════════════════════════════════════════════════════════════════
# DATAFRAME CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def readings_to_df(readings: List[Dict]):
    """
    Convert Firebase readings list → pandas DataFrame for ML training.
    Handles both Unix epoch timestamp and ISO string ts.
    """
    import pandas as pd

    if not readings:
        return pd.DataFrame()

    rows = []
    for rec in readings:
        try:
            ts_val = rec.get("timestamp")
            if ts_val and float(ts_val) > 1_000_000:
                date = pd.to_datetime(float(ts_val), unit="s")
            else:
                ts_iso = rec.get("ts", "")
                date = pd.to_datetime(ts_iso) if ts_iso else pd.Timestamp.now()

            rows.append({
                "date":         date,
                "feed_kg":      float(rec.get("weight",      0.0)),
                "water_liters": float(rec.get("totalLiters", 0.0)),
                "flow":         float(rec.get("flow",        0.0)),
                "level":        str(rec.get("level",         "0%")),
                "day_of_week":  int(rec.get("dayOfWeek",     0)),
                "month":        int(rec.get("month",         1)),
                "system":       1,
            })
        except Exception as e:
            print(f"[DB] skip row: {e}")
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