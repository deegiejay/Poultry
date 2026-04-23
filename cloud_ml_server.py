"""
cloud_ml_server.py  —  Cloud ML Engine
========================================
CRITICAL FIX: When Render runs `uvicorn cloud_ml_server:app`, the
`if __name__ == "__main__"` block NEVER runs, so the training thread
was never started. Fixed by using FastAPI's `lifespan` context manager
which runs on startup regardless of how uvicorn is invoked.

ARCHIVE SYSTEM:
  After each training cycle, /readings is checked. If count > ACTIVE_LIMIT,
  oldest rows are moved to /archive automatically. ML trains on bounded
  active window only — fast and scalable forever.

RENDER DEPLOY:
  Start command: uvicorn cloud_ml_server:app --host 0.0.0.0 --port $PORT
  Env var: FIREBASE_URL = https://poultry-ai-e901a-default-rtdb.firebaseio.com
"""

import os
import sys
import time
import threading
import traceback
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# ─── Patch Firebase URL from env BEFORE any DB call ──────────────────────────
import firebase_db as db

_env_url = os.getenv("FIREBASE_URL", "").strip()
if _env_url:
    db.FIREBASE_URL = _env_url
    print(f"[CONFIG] Firebase URL from env: {db.FIREBASE_URL}")
else:
    print(f"[CONFIG] Firebase URL from module: {db.FIREBASE_URL}")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
RETRAIN_EVERY  = 50     # retrain when N new rows arrive since last train
MIN_ROWS       = 20     # minimum rows before first train
ROLLING_WINDOW = 500    # use last N rows for training (matches ACTIVE_LIMIT)
TRAIN_INTERVAL = 90     # also retrain every N seconds (shorter = faster first train)
ANOMALY_Z      = 2.5

FEATURES = [
    "water_liters", "system", "day_of_week", "month",
    "hour", "lag1_feed", "lag1_water", "roll3_feed",
]

# ═════════════════════════════════════════════════════════════════════════════
# STATE
# ═════════════════════════════════════════════════════════════════════════════
state: Dict[str, Any] = {
    "trained_rows": 0,
    "training":     False,
    "status":       "starting",
    "error":        "",
}
_lock  = threading.Lock()
_event = threading.Event()


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def detect_anomaly(df: pd.DataFrame) -> Dict:
    result = {"anomaly": False, "message": "", "value": 0.0}
    if len(df) < 10:
        return result
    for col, label in [("feed_kg", "Feed/Weight"), ("water_liters", "Water")]:
        vals = df[col].tail(50).dropna()
        if len(vals) < 5:
            continue
        z = abs((vals.iloc[-1] - vals.mean()) / (vals.std() + 1e-9))
        if z > ANOMALY_Z:
            result.update(anomaly=True,
                          message=f"⚠️ {label} is {z:.1f}σ from normal ({vals.iloc[-1]:.3f})",
                          value=float(vals.iloc[-1]))
            return result
    return result


def analyze_trend(df: pd.DataFrame) -> Dict:
    result = {"trend": "stable", "icon": "✅", "feedDelta": 0.0, "waterDelta": 0.0}
    if len(df) < 40:
        return result
    rec  = df.tail(20)
    prev = df.iloc[-40:-20]
    fd = (rec["feed_kg"].mean()      - prev["feed_kg"].mean())      / (prev["feed_kg"].mean()      + 1e-9) * 100
    wd = (rec["water_liters"].mean() - prev["water_liters"].mean()) / (prev["water_liters"].mean() + 1e-9) * 100
    result.update(feedDelta=round(fd, 1), waterDelta=round(wd, 1))
    if detect_anomaly(df)["anomaly"]:
        result.update(trend="warning",    icon="🚨")
    elif abs(fd) < 5 and abs(wd) < 5:
        result.update(trend="stable",     icon="✅")
    elif fd > 5 or wd > 5:
        result.update(trend="increasing", icon="📈")
    else:
        result.update(trend="decreasing", icon="📉")
    return result


def calc_confidence(df: pd.DataFrame, rows: int) -> float:
    row_score = min(1.0, max(0.0, (rows - MIN_ROWS) / max(1, 500 - MIN_ROWS)))
    try:
        cv = (df["feed_kg"].std()      / (df["feed_kg"].mean()      + 1e-9) +
              df["water_liters"].std() / (df["water_liters"].mean() + 1e-9)) / 2
        var_score = max(0.0, 1.0 - cv)
    except Exception:
        var_score = 0.5
    return round(min(1.0, max(0.05, row_score * 0.6 + var_score * 0.4)), 2)


def confidence_label(c: float) -> str:
    if c >= 0.80: return "High"
    if c >= 0.55: return "Medium"
    if c >= 0.30: return "Low"
    return "Very Low"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single source of truth for feature engineering."""
    df = df.copy()
    df["hour"]       = pd.to_datetime(df["date"]).dt.hour
    df["lag1_feed"]  = df["feed_kg"].shift(1).fillna(df["feed_kg"].mean())
    df["lag1_water"] = df["water_liters"].shift(1).fillna(df["water_liters"].mean())
    df["roll3_feed"] = df["feed_kg"].rolling(3, min_periods=1).mean()
    return df


def build_predict_row(last_row: pd.Series, next_date: pd.Timestamp,
                      recent_df: pd.DataFrame) -> pd.DataFrame:
    """Build one prediction input row. Uses same FEATURES as training."""
    return pd.DataFrame([{
        "water_liters": float(last_row["water_liters"]),
        "system":       int(last_row.get("system", 1)),
        "day_of_week":  next_date.weekday(),
        "month":        next_date.month,
        "hour":         0,
        "lag1_feed":    float(last_row["feed_kg"]),
        "lag1_water":   float(last_row["water_liters"]),
        "roll3_feed":   float(recent_df["feed_kg"].tail(3).mean()),
    }])


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_once():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    HAS_ARIMA = False
    try:
        from statsmodels.tsa.arima.model import ARIMA as _A
        HAS_ARIMA = True
    except Exception:
        pass

    with _lock:
        state["training"] = True
        state["status"]   = "training"

    db.write_ml_status("training", state["trained_rows"])

    try:
        # 1. Load ─────────────────────────────────────────────────────────────
        readings = db.get_readings(limit=ROLLING_WINDOW)
        print(f"[ML] Fetched {len(readings)} readings from Firebase")

        if not readings:
            db.write_ml_status("waiting", 0, "No data in Firebase yet")
            with _lock:
                state.update(training=False, status="waiting")
            return

        df = db.readings_to_df(readings)
        print(f"[ML] DataFrame: {len(df)} rows, columns: {list(df.columns)}")

        if df.empty or len(df) < MIN_ROWS:
            msg = f"Need {MIN_ROWS} rows, have {len(df)}"
            db.write_ml_status("collecting", len(df), msg)
            with _lock:
                state.update(training=False, status="collecting",
                             trained_rows=len(df))
            return

        total_rows = len(df)

        # 2. Features ─────────────────────────────────────────────────────────
        df = add_features(df)
        X       = df[FEATURES].fillna(0)
        y_feed  = df["feed_kg"]
        y_water = df["water_liters"]

        # 3. Train ────────────────────────────────────────────────────────────
        def make_pipe():
            return Pipeline([
                ("sc", StandardScaler()),
                ("gb", GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, random_state=42)),
            ])

        m_feed  = make_pipe(); m_feed.fit(X, y_feed)
        m_water = make_pipe(); m_water.fit(X, y_water)
        print(f"[ML] Models trained on {total_rows} rows")

        # 4. Next-day prediction ──────────────────────────────────────────────
        last   = df.iloc[-1]
        nd     = pd.to_datetime(last["date"]) + timedelta(days=1)
        inp    = build_predict_row(last, nd, df)
        feed_v = round(float(m_feed.predict(inp)[0]),  3)
        water_v= round(float(m_water.predict(inp)[0]), 3)

        # 5. ARIMA ────────────────────────────────────────────────────────────
        arima_feed = arima_water = feed_v
        if HAS_ARIMA and total_rows >= 30:
            try:
                from statsmodels.tsa.arima.model import ARIMA
                af = ARIMA(df["feed_kg"].values,      order=(2, 1, 2)).fit()
                aw = ARIMA(df["water_liters"].values, order=(2, 1, 2)).fit()
                arima_feed  = round(float(af.forecast(1)[0]), 3)
                arima_water = round(float(aw.forecast(1)[0]), 3)
            except Exception as e:
                print(f"[ML] ARIMA skipped: {e}")

        # 6. Analysis ─────────────────────────────────────────────────────────
        conf  = calc_confidence(df, total_rows)
        trend = analyze_trend(df)
        anom  = detect_anomaly(df)

        # 7. 7-day forecast ───────────────────────────────────────────────────
        rows_7d = []
        tmp = df.copy()
        for _ in range(7):
            l    = tmp.iloc[-1]
            nd2  = pd.to_datetime(l["date"]) + timedelta(days=1)
            xi   = build_predict_row(l, nd2, tmp)
            fv   = float(m_feed.predict(xi)[0])
            wv   = float(m_water.predict(xi)[0])
            rows_7d.append({"date": str(nd2.date()),
                            "feed_kg": round(fv, 4),
                            "water_liters": round(wv, 4)})
            new_row = pd.DataFrame([{
                "date": nd2, "feed_kg": fv, "water_liters": wv,
                "system": 1, "day_of_week": nd2.weekday(), "month": nd2.month,
                "hour": 0, "lag1_feed": l["feed_kg"], "lag1_water": l["water_liters"],
                "roll3_feed": float(tmp["feed_kg"].tail(3).mean()),
                "flow": 0.0, "level": "0%",
            }])
            tmp = pd.concat([tmp, new_row], ignore_index=True)

        # 8. Patterns ─────────────────────────────────────────────────────────
        try:
            pat_sys   = df.groupby("system")[["feed_kg","water_liters"]].mean().to_dict()
            pat_day   = df.groupby("day_of_week")[["feed_kg","water_liters"]].mean().to_dict()
            pat_month = df.groupby("month")[["feed_kg","water_liters"]].mean().to_dict()
        except Exception:
            pat_sys = pat_day = pat_month = {}

        # 9. Write to Firebase ────────────────────────────────────────────────
        ml_result = {
            "feedKg":     feed_v,
            "waterL":     water_v,
            "predDate":   str(nd.date()),
            "arimaFeed":  arima_feed,
            "arimaWater": arima_water,
            "confidence": conf,
            "confLabel":  confidence_label(conf),
            "trend":      trend["trend"],
            "trendIcon":  trend["icon"],
            "feedDelta":  trend["feedDelta"],
            "waterDelta": trend["waterDelta"],
            "anomaly":    anom["anomaly"],
            "anomalyMsg": anom["message"],
            "modelRows":  total_rows,
            "trainedAt":  datetime.utcnow().isoformat(),
            "patSystem":  pat_sys,
            "patDay":     pat_day,
            "patMonth":   pat_month,
        }

        ok1 = db.write_ml_result(ml_result)
        ok2 = db.write_forecast_7d(rows_7d)
        db.write_ml_status("ready", total_rows)

        if anom["anomaly"]:
            db.push_alert("anomaly", anom["message"], anom["value"])

        with _lock:
            state.update(trained_rows=total_rows, training=False,
                         status="ready", error="")

        print(f"[ML] ✅ feed={feed_v}kg water={water_v}L "
              f"conf={int(conf*100)}% trend={trend['trend']} "
              f"write={'ok' if ok1 and ok2 else 'FAILED'}")

        # 10. Archive old readings ─────────────────────────────────────────────
        archived = db.archive_old_readings()
        if archived:
            print(f"[ML] Archived {archived} old readings to /archive")

    except Exception as ex:
        err = str(ex)[:200]
        print(f"[ML] ❌ {err}")
        print(traceback.format_exc())
        db.write_ml_status("error", state.get("trained_rows", 0), err)
        with _lock:
            state.update(training=False, status="error", error=err)


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def training_loop():
    last_trained_rows = 0
    print("[ML] Training loop started — waiting for first interval…")

    # Trigger immediately on startup
    _event.set()

    while True:
        _event.wait(timeout=TRAIN_INTERVAL)
        _event.clear()
        try:
            with _lock:
                if state["training"]:
                    continue

            current = db.get_reading_count()
            print(f"[ML] Row count check: {current}")

            if current < MIN_ROWS:
                msg = f"Need {MIN_ROWS} rows, have {current}"
                print(f"[ML] {msg}")
                db.write_ml_status("collecting", current, msg)
                with _lock:
                    state.update(status="collecting", trained_rows=current)
                continue

            new_rows = current - last_trained_rows
            if new_rows < RETRAIN_EVERY and last_trained_rows > 0:
                print(f"[ML] Only {new_rows} new rows since last train "
                      f"(need {RETRAIN_EVERY}) — skipping")
                continue

            train_once()
            last_trained_rows = db.get_reading_count()

        except Exception:
            print(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI  with lifespan  — THIS is the fix for Render
# When Render runs `uvicorn cloud_ml_server:app`, __main__ is never called.
# lifespan runs on startup regardless of how the app is invoked.
# ═════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: launch training thread. Shutdown: nothing needed (daemon)."""
    print("=" * 60)
    print("  Poultry Farm Cloud ML Server — STARTUP")
    print(f"  Firebase : {db.FIREBASE_URL}")
    print(f"  Retrains : every {RETRAIN_EVERY} rows or {TRAIN_INTERVAL}s")
    print(f"  Min rows : {MIN_ROWS}")
    print("=" * 60)

    t = threading.Thread(target=training_loop, daemon=True, name="ml-training")
    t.start()
    print(f"[STARTUP] Training thread started: {t.name}")

    yield   # app runs here

    print("[SHUTDOWN] ML server stopping")


app = FastAPI(title="Poultry Farm ML Server", lifespan=lifespan)


@app.get("/")
async def root():
    with _lock:
        s = dict(state)
    return JSONResponse({"service": "Poultry Farm ML", "state": s})


@app.get("/health")
async def health():
    """UptimeRobot: ping this every 5min to keep Render free tier awake."""
    return JSONResponse({
        "status": "ok",
        "ts":     datetime.utcnow().isoformat(),
        "rows":   state.get("trained_rows", 0),
        "ml":     state.get("status", "unknown"),
    })


@app.get("/status")
async def get_status():
    with _lock:
        return JSONResponse(dict(state))


@app.get("/count")
async def get_count():
    """Quick reading count check — useful for debugging."""
    count = db.get_reading_count()
    return JSONResponse({"readings": count, "firebase": db.FIREBASE_URL})


@app.post("/retrain")
async def force_retrain():
    """POST to trigger immediate retrain — useful after deploy."""
    _event.set()
    return JSONResponse({"message": "Retrain triggered"})


# ═════════════════════════════════════════════════════════════════════════════
# LOCAL RUN  (python cloud_ml_server.py)
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "cloud_ml_server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )