"""
cloud_ml_server.py  —  Cloud ML Engine
========================================
Deploy to Render (free tier) or Railway in one click.
Runs 24/7. No laptop needed.

WHAT IT DOES:
  1. Reads ESP32 readings from Firebase every 60s
  2. Retrains RandomForest + ARIMA when 50+ new rows arrive
  3. Runs anomaly detection + trend analysis
  4. Writes predictions / forecast / status back to Firebase
  5. APK reads results — phone never does ML locally

DEPLOY TO RENDER (free, 5 min):
  1. Push this repo to GitHub
  2. render.com → New Web Service → connect repo
  3. Build:   pip install -r requirements.txt
  4. Start:   python cloud_ml_server.py
  5. Done — server runs 24/7 on Render's free tier

requirements.txt:
  fastapi
  uvicorn
  pandas
  numpy
  scikit-learn
  statsmodels
  requests
  firebase-admin  (optional, we use REST so not needed)

ENV VARS to set on Render:
  FIREBASE_URL = https://poultry-farm-ai-default-rtdb.firebaseio.com
  PORT         = 8000  (Render sets this automatically)
"""

import os
import time
import threading
import traceback
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

import firebase_db as db

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
RETRAIN_EVERY   = 50      # retrain when N new rows arrive since last train
MIN_ROWS        = 20      # minimum rows before first train
ROLLING_WINDOW  = 1000    # use last N rows for training
TRAIN_INTERVAL  = 120     # also retrain every N seconds even without new rows
ANOMALY_Z       = 2.5     # Z-score threshold

# Override Firebase URL from environment (for Render deploy)
if os.getenv("FIREBASE_URL"):
    db.FIREBASE_URL = os.getenv("FIREBASE_URL")

# ═════════════════════════════════════════════════════════════════════════════
# ML STATE  (in-memory, reset on server restart — results live in Firebase)
# ═════════════════════════════════════════════════════════════════════════════
state: Dict[str, Any] = {
    "trained_rows": 0,
    "training":     False,
    "last_result":  None,
    "status":       "starting",
    "error":        "",
}
_lock  = threading.Lock()
_event = threading.Event()


# ═════════════════════════════════════════════════════════════════════════════
# ML HELPERS
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
                          message=f"⚠️ {label} reading is {z:.1f}σ from normal ({vals.iloc[-1]:.3f})",
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
    result["feedDelta"]  = round(fd, 1)
    result["waterDelta"] = round(wd, 1)
    if detect_anomaly(df)["anomaly"]:
        result.update(trend="warning", icon="🚨")
    elif abs(fd) < 5 and abs(wd) < 5:
        result.update(trend="stable",     icon="✅")
    elif fd > 5 or wd > 5:
        result.update(trend="increasing", icon="📈")
    else:
        result.update(trend="decreasing", icon="📉")
    return result


def calc_confidence(df: pd.DataFrame, rows: int) -> float:
    row_score = min(1.0, max(0.0, (rows - MIN_ROWS) / (500 - MIN_ROWS)))
    try:
        cv = (df["feed_kg"].std() / (df["feed_kg"].mean() + 1e-9) +
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


def predict_next(df: pd.DataFrame, m_feed, m_water) -> Dict:
    last = df.iloc[-1]
    nd   = pd.to_datetime(last["date"]) + timedelta(days=1)
    inp  = pd.DataFrame([{
        "water_liters": last["water_liters"],
        "system":       last.get("system", 1),
        "day_of_week":  nd.weekday(),
        "month":        nd.month,
    }])
    return {
        "feedKg":   round(float(m_feed.predict(inp)[0]),  3),
        "waterL":   round(float(m_water.predict(inp)[0]), 3),
        "predDate": str(nd.date()),
    }


def predict_7d(df: pd.DataFrame, m_feed, m_water) -> list:
    rows, tmp = [], df.copy()
    for _ in range(7):
        last = tmp.iloc[-1]
        nd   = pd.to_datetime(last["date"]) + timedelta(days=1)
        inp  = pd.DataFrame([{
            "water_liters": last["water_liters"],
            "system":       last.get("system", 1),
            "day_of_week":  nd.weekday(),
            "month":        nd.month,
        }])
        f = float(m_feed.predict(inp)[0])
        w = float(m_water.predict(inp)[0])
        rows.append({"date": str(nd.date()), "feed_kg": round(f,4), "water_liters": round(w,4)})
        tmp = pd.concat([tmp, pd.DataFrame([{
            "date": nd, "feed_kg": f, "water_liters": w,
            "system": 1, "day_of_week": nd.weekday(), "month": nd.month,
        }])], ignore_index=True)
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def train_once():
    """Full training cycle — reads Firebase, trains, writes results back."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    try:
        from statsmodels.tsa.arima.model import ARIMA
        HAS_ARIMA = True
    except Exception:
        HAS_ARIMA = False

    with _lock:
        state["training"] = True
        state["status"]   = "training"

    db.write_ml_status("training", state["trained_rows"])

    try:
        # 1. Load data
        readings = db.get_readings(limit=ROLLING_WINDOW)
        if not readings:
            db.write_ml_status("waiting", 0, "No data in Firebase yet")
            with _lock:
                state.update(training=False, status="waiting")
            return

        df = db.readings_to_df(readings)
        if df.empty or len(df) < MIN_ROWS:
            msg = f"Need {MIN_ROWS} rows, have {len(df)}"
            db.write_ml_status("collecting", len(df), msg)
            with _lock:
                state.update(training=False, status="collecting",
                             trained_rows=len(df))
            return

        total_rows = len(df)
        print(f"[ML] Training on {total_rows} rows…")

        # 2. Feature engineering
        df["hour"]      = pd.to_datetime(df["date"]).dt.hour
        df["lag1_feed"] = df["feed_kg"].shift(1).fillna(df["feed_kg"].mean())
        df["lag1_water"]= df["water_liters"].shift(1).fillna(df["water_liters"].mean())
        df["roll3_feed"] = df["feed_kg"].rolling(3, min_periods=1).mean()

        FEATURES = ["water_liters", "system", "day_of_week", "month",
                    "hour", "lag1_feed", "lag1_water", "roll3_feed"]

        X = df[FEATURES].fillna(0)
        y_feed  = df["feed_kg"]
        y_water = df["water_liters"]

        # 3. Train models
        m_feed  = Pipeline([("sc", StandardScaler()),
                            ("rf", GradientBoostingRegressor(n_estimators=100,
                                                              max_depth=4,
                                                              random_state=42))])
        m_water = Pipeline([("sc", StandardScaler()),
                            ("rf", GradientBoostingRegressor(n_estimators=100,
                                                              max_depth=4,
                                                              random_state=42))])
        m_feed.fit(X, y_feed)
        m_water.fit(X, y_water)

        # 4. Next-day prediction (needs compatible feature set)
        last = df.iloc[-1]
        nd   = pd.to_datetime(last["date"]) + timedelta(days=1)
        inp  = pd.DataFrame([{
            "water_liters":  last["water_liters"],
            "system":        last.get("system", 1),
            "day_of_week":   nd.weekday(),
            "month":         nd.month,
            "hour":          0,
            "lag1_feed":     last["feed_kg"],
            "lag1_water":    last["water_liters"],
            "roll3_feed":    df["feed_kg"].tail(3).mean(),
        }])
        feed_v  = round(float(m_feed.predict(inp)[0]),  3)
        water_v = round(float(m_water.predict(inp)[0]), 3)

        # 5. ARIMA forecast
        arima_feed = arima_water = None
        if HAS_ARIMA and total_rows >= 30:
            try:
                af = ARIMA(df["feed_kg"].values,      order=(2,1,2)).fit()
                aw = ARIMA(df["water_liters"].values, order=(2,1,2)).fit()
                arima_feed  = round(float(af.forecast(1)[0]),  3)
                arima_water = round(float(aw.forecast(1)[0]),  3)
            except Exception as e:
                print(f"[ML] ARIMA skipped: {e}")
                arima_feed  = feed_v
                arima_water = water_v
        else:
            arima_feed  = feed_v
            arima_water = water_v

        # 6. Confidence / trend / anomaly
        conf  = calc_confidence(df, total_rows)
        trend = analyze_trend(df)
        anom  = detect_anomaly(df)

        # 7. 7-day forecast
        rows_7d = []
        tmp = df.copy()
        for _ in range(7):
            l  = tmp.iloc[-1]
            nd2 = pd.to_datetime(l["date"]) + timedelta(days=1)
            xi = pd.DataFrame([{
                "water_liters":  l["water_liters"],
                "system":        1,
                "day_of_week":   nd2.weekday(),
                "month":         nd2.month,
                "hour":          0,
                "lag1_feed":     l["feed_kg"],
                "lag1_water":    l["water_liters"],
                "roll3_feed":    tmp["feed_kg"].tail(3).mean(),
            }])
            fv = float(m_feed.predict(xi)[0])
            wv = float(m_water.predict(xi)[0])
            rows_7d.append({"date": str(nd2.date()),
                            "feed_kg": round(fv,4),
                            "water_liters": round(wv,4)})
            tmp = pd.concat([tmp, pd.DataFrame([{
                "date": nd2, "feed_kg": fv, "water_liters": wv,
                "system": 1, "day_of_week": nd2.weekday(), "month": nd2.month,
                "hour": 0, "lag1_feed": fv, "lag1_water": wv, "roll3_feed": fv,
            }])], ignore_index=True)

        # 8. Patterns (simple groupby)
        try:
            pat_sys  = df.groupby("system")[["feed_kg","water_liters"]].mean().to_dict()
            pat_day  = df.groupby("day_of_week")[["feed_kg","water_liters"]].mean().to_dict()
            pat_month= df.groupby("month")[["feed_kg","water_liters"]].mean().to_dict()
        except Exception:
            pat_sys = pat_day = pat_month = {}

        # 9. Write everything to Firebase
        ml_result = {
            "feedKg":       feed_v,
            "waterL":       water_v,
            "predDate":     str(nd.date()),
            "arimaFeed":    arima_feed,
            "arimaWater":   arima_water,
            "confidence":   conf,
            "confLabel":    confidence_label(conf),
            "trend":        trend["trend"],
            "trendIcon":    trend["icon"],
            "feedDelta":    trend["feedDelta"],
            "waterDelta":   trend["waterDelta"],
            "anomaly":      anom["anomaly"],
            "anomalyMsg":   anom["message"],
            "modelRows":    total_rows,
            "trainedAt":    datetime.utcnow().isoformat(),
            "patSystem":    pat_sys,
            "patDay":       pat_day,
            "patMonth":     pat_month,
        }
        db.write_ml_result(ml_result)
        db.write_forecast_7d(rows_7d)
        db.write_ml_status("ready", total_rows)

        if anom["anomaly"]:
            db.push_alert("anomaly", anom["message"], anom["value"])

        with _lock:
            state.update(
                trained_rows = total_rows,
                training     = False,
                status       = "ready",
                last_result  = ml_result,
                error        = "",
            )

        print(f"[ML] ✅ Done — feed={feed_v}kg water={water_v}L "
              f"conf={int(conf*100)}% trend={trend['trend']}")

    except Exception as ex:
        err = str(ex)[:120]
        print(f"[ML] ❌ Error: {err}")
        print(traceback.format_exc())
        db.write_ml_status("error", state.get("trained_rows", 0), err)
        with _lock:
            state.update(training=False, status="error", error=err)


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP  — background thread
# ═════════════════════════════════════════════════════════════════════════════

def training_loop():
    last_trained_rows = 0
    print("[ML] Training loop started")
    while True:
        _event.wait(timeout=TRAIN_INTERVAL)
        _event.clear()
        try:
            with _lock:
                busy = state["training"]
            if busy:
                continue
            current = db.get_reading_count()
            if current < MIN_ROWS:
                db.write_ml_status("collecting", current,
                                   f"Need {MIN_ROWS} rows, have {current}")
                with _lock:
                    state.update(status="collecting", trained_rows=current)
                continue
            new_rows = current - last_trained_rows
            if new_rows < RETRAIN_EVERY and last_trained_rows > 0:
                continue
            train_once()
            last_trained_rows = db.get_reading_count()
        except Exception:
            print(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI  — health check + status endpoints (Render needs an HTTP port)
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Poultry Farm ML Server")


@app.get("/")
async def root():
    with _lock:
        s = dict(state)
    s.pop("last_result", None)   # don't expose large object in health check
    return JSONResponse({"service": "Poultry Farm ML", "state": s})


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "ts": datetime.utcnow().isoformat()})


@app.get("/status")
async def status():
    with _lock:
        return JSONResponse(dict(state))


@app.post("/retrain")
async def force_retrain():
    """Manually trigger a retrain (useful for testing)."""
    _event.set()
    return JSONResponse({"message": "Retrain triggered"})


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  Poultry Farm Cloud ML Server")
    print(f"  Firebase: {db.FIREBASE_URL}")
    print(f"  Retrains every {RETRAIN_EVERY} new rows or {TRAIN_INTERVAL}s")
    print("=" * 55)

    # Start training thread
    t = threading.Thread(target=training_loop, daemon=True)
    t.start()

    # Start FastAPI on Render's PORT env var (default 8000)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")