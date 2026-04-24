"""
cloud_ml_server.py  —  Cloud ML Engine (Deploy to Render / Railway)
=====================================================================
Runs 24/7. No laptop. Trains models. Writes results to Firebase.
APK reads results — zero ML on phone.

DEPLOY TO RENDER (free tier):
  1. Push these files to a GitHub repo:
       cloud_ml_server.py, firebase_db.py, requirements.txt
  2. render.com → New Web Service → connect repo
  3. Build command:  pip install -r requirements.txt
  4. Start command:  python cloud_ml_server.py
  5. Environment vars → Add:
       FIREBASE_URL = https://poultry-ai-e901a-default-rtdb.firebaseio.com
  6. Deploy — done. Runs forever.

RENDER FREE TIER NOTE:
  Free services sleep after 15min of no HTTP traffic.
  Add a free uptime monitor (e.g. UptimeRobot pinging /health every 5min)
  to keep it awake 24/7.
"""

import os
import sys
import time
import threading
import traceback
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd
import io
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# ─── Import firebase_db and override URL from env BEFORE any DB calls ────────
import firebase_db as _db

_env_url = os.getenv("FIREBASE_URL", "").strip()
if _env_url:
    _db.FIREBASE_URL = _env_url          # patch module-level var; _base() reads it live
    print(f"[CONFIG] Firebase URL from env: {_db.FIREBASE_URL}")
else:
    print(f"[CONFIG] Firebase URL from module: {_db.FIREBASE_URL}")

# Use db as alias after patching
db = _db

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
RETRAIN_EVERY  = 50     # retrain when N new rows arrive
MIN_ROWS       = 20     # minimum rows before first train
ROLLING_WINDOW = 1000   # use last N rows
TRAIN_INTERVAL = 120    # retrain every N seconds even without new rows
ANOMALY_Z      = 2.5    # Z-score threshold for anomaly detection

BG      = "#0e1117"
BORDER  = "#3d4257"
MUTED   = "#a0aec0"
CFEED   = "#50C8FF"
CWATER  = "#1f77b4"

# IMPORTANT: must match FEATURES list used during training
FEATURES = [
    "water_liters", "system", "day_of_week", "month",
    "hour", "lag1_feed", "lag1_water", "roll3_feed",
]

# ═════════════════════════════════════════════════════════════════════════════
# IN-MEMORY STATE
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
# ML ANALYSIS HELPERS
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
            result.update(
                anomaly=True,
                message=f"⚠️ {label} reading is {z:.1f}σ from normal ({vals.iloc[-1]:.3f})",
                value=float(vals.iloc[-1]),
            )
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
    """
    Add engineered features to DataFrame.
    Called once during training and once per prediction row.
    Single source of truth — prevents train/predict feature mismatch.
    """
    df = df.copy()
    df["hour"]       = pd.to_datetime(df["date"]).dt.hour
    df["lag1_feed"]  = df["feed_kg"].shift(1).fillna(df["feed_kg"].mean())
    df["lag1_water"] = df["water_liters"].shift(1).fillna(df["water_liters"].mean())
    df["roll3_feed"] = df["feed_kg"].rolling(3, min_periods=1).mean()
    return df


def build_predict_row(last_row: pd.Series, next_date: pd.Timestamp,
                      recent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single prediction input row with all FEATURES.
    Uses last_row values for lag features.
    """
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

def render_chart_b64(df: pd.DataFrame, forecast_rows: list = None) -> str:
    """
    Render feed vs water chart as base64 PNG.
    Cloud server generates this; APK only displays it.
    """
    try:
        if df.empty:
            return ""

        plot_df = df.tail(200).copy()

        fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
        ax.set_facecolor(BG)

        ax.plot(
            plot_df["date"],
            plot_df["feed_kg"],
            color=CFEED,
            lw=2.0,
            marker="o",
            ms=3,
            label="Feed / Weight",
        )

        ax.plot(
            plot_df["date"],
            plot_df["water_liters"],
            color=CWATER,
            lw=2.0,
            marker="o",
            ms=3,
            label="Water (L)",
        )

        if forecast_rows:
            try:
                fdf = pd.DataFrame(forecast_rows)
                fdf["date"] = pd.to_datetime(fdf["date"])

                ax.axvline(
                    x=plot_df["date"].iloc[-1],
                    color=BORDER,
                    lw=1,
                    ls="--",
                    alpha=0.5,
                )

                ax.plot(
                    fdf["date"],
                    fdf["feed_kg"],
                    color=CFEED,
                    lw=1.8,
                    ls="--",
                    alpha=0.8,
                    label="Feed Forecast",
                )

                ax.plot(
                    fdf["date"],
                    fdf["water_liters"],
                    color=CWATER,
                    lw=1.8,
                    ls="--",
                    alpha=0.8,
                    label="Water Forecast",
                )
            except Exception as e:
                print(f"[CHART] forecast skipped: {e}")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.xticks(rotation=25, color=MUTED, fontsize=8, ha="right")
        plt.yticks(color=MUTED, fontsize=9)

        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color=BORDER, lw=0.6, ls="--", alpha=0.7)

        for sp in ax.spines.values():
            sp.set_visible(False)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            frameon=False,
            labelcolor=MUTED,
            fontsize=8,
        )

        ax.set_title("Consumption Trends", color=MUTED, fontsize=11, pad=8)

        plt.tight_layout(pad=1.5)

        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            dpi=140,
            facecolor=BG,
        )
        plt.close(fig)

        return base64.b64encode(buf.getvalue()).decode()


    except Exception as e:
        print(f"[CHART] error: {e}")
        return None
# ═════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def train_once():
    """
    Full training cycle:
      1. Load readings from Firebase
      2. Feature engineering
      3. Train GradientBoosting models (feed + water)
      4. Run ARIMA
      5. Detect anomaly + trend
      6. Forecast 7 days
      7. Write all results back to Firebase
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    HAS_ARIMA = False
    try:
        from statsmodels.tsa.arima.model import ARIMA
        HAS_ARIMA = True
    except Exception:
        pass

    with _lock:
        state["training"] = True
        state["status"]   = "training"

    db.write_ml_status("training", state["trained_rows"])

    try:
        # ── 1. Load data ──────────────────────────────────────────────────────
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

        # ── 2. Feature engineering ────────────────────────────────────────────
        df = add_features(df)
        X       = df[FEATURES].fillna(0)
        y_feed  = df["feed_kg"]
        y_water = df["water_liters"]

        # ── 3. Train models ───────────────────────────────────────────────────
        def make_pipe():
            return Pipeline([
                ("sc", StandardScaler()),
                ("gb", GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, random_state=42)),
            ])

        m_feed  = make_pipe()
        m_water = make_pipe()
        m_feed.fit(X, y_feed)
        m_water.fit(X, y_water)

        # ── 4. Next-day prediction ────────────────────────────────────────────
        last   = df.iloc[-1]
        nd     = pd.to_datetime(last["date"]) + timedelta(days=1)
        inp    = build_predict_row(last, nd, df)
        feed_v = round(float(m_feed.predict(inp)[0]),  3)
        water_v= round(float(m_water.predict(inp)[0]), 3)

        # ── 5. ARIMA (stable version)
        arima_feed = arima_water = None

        if HAS_ARIMA and total_rows >= 30:
            try:
                from statsmodels.tsa.arima.model import ARIMA

                # Feed model (lighter)
                af = ARIMA(
                    df["feed_kg"].values,
                    order=(1, 1, 1),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit()

                arima_feed = round(float(af.forecast(1)[0]), 3)

                # Water model check if mostly same values
                if df["water_liters"].nunique() <= 2:
                    arima_water = round(float(df["water_liters"].mean()), 3)
                else:
                    aw = ARIMA(
                        df["water_liters"].values,
                        order=(1, 1, 0),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit()

                    arima_water = round(float(aw.forecast(1)[0]), 3)

            except Exception as e:
                print(f"[ML] ARIMA skipped: {e}")
                arima_feed = feed_v
                arima_water = water_v

        else:
            arima_feed = feed_v
            arima_water = water_v

        # ── 6. Confidence / trend / anomaly ───────────────────────────────────
        conf  = calc_confidence(df, total_rows)
        trend = analyze_trend(df)
        anom  = detect_anomaly(df)

        # ── 7. 7-day forecast (consistent features) ───────────────────────────
        rows_7d = []
        tmp     = df.copy()                # starts as engineered df
        for _ in range(7):
            l_row = tmp.iloc[-1]
            nd2   = pd.to_datetime(l_row["date"]) + timedelta(days=1)
            xi    = build_predict_row(l_row, nd2, tmp)
            fv    = float(m_feed.predict(xi)[0])
            wv    = float(m_water.predict(xi)[0])
            rows_7d.append({
                "date":         str(nd2.date()),
                "feed_kg":      round(fv, 4),
                "water_liters": round(wv, 4),
            })
            # Append new row with engineered features for next iteration
            new_row = pd.DataFrame([{
                "date":         nd2,
                "feed_kg":      fv,
                "water_liters": wv,
                "system":       1,
                "day_of_week":  nd2.weekday(),
                "month":        nd2.month,
                "hour":         0,
                "lag1_feed":    l_row["feed_kg"],
                "lag1_water":   l_row["water_liters"],
                "roll3_feed":   float(tmp["feed_kg"].tail(3).mean()),
                "flow":         0.0,
                "level":        "0%",
            }])
            tmp = pd.concat([tmp, new_row], ignore_index=True)

        # ── 8. Patterns ───────────────────────────────────────────────────────
        try:
            pat_sys   = df.groupby("system")[["feed_kg","water_liters"]].mean().to_dict()
            pat_day   = df.groupby("day_of_week")[["feed_kg","water_liters"]].mean().to_dict()
            pat_month = df.groupby("month")[["feed_kg","water_liters"]].mean().to_dict()
        except Exception:
            pat_sys = pat_day = pat_month = {}


        chart_b64 = render_chart_b64(df, rows_7d)
        # ── 9. Write to Firebase ──────────────────────────────────────────────
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
            "chartB64":   chart_b64,
        }

        ok1 = db.write_ml_result(ml_result)
        ok2 = db.write_forecast_7d(rows_7d)
        db.write_ml_status("ready", total_rows)

        if anom["anomaly"]:
            db.push_alert("anomaly", anom["message"], anom["value"])

        with _lock:
            state.update(
                trained_rows=total_rows,
                training=False,
                status="ready",
                error="",
            )

        print(f"[ML] ✅ feed={feed_v}kg water={water_v}L "
              f"conf={int(conf*100)}% trend={trend['trend']} "
              f"firebase_write={'ok' if ok1 and ok2 else 'FAILED'}")

    except Exception as ex:
        err = str(ex)[:200]
        print(f"[ML] ❌ {err}")
        print(traceback.format_exc())
        db.write_ml_status("error", state.get("trained_rows", 0), err)
        with _lock:
            state.update(training=False, status="error", error=err)


# ═════════════════════════════════════════════════════════════════════════════
# BACKGROUND TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def training_loop():
    last_trained_rows = 0
    print("[ML] Training loop started")

    while True:
        triggered = _event.wait(timeout=TRAIN_INTERVAL)
        _event.clear()
        try:
            with _lock:
                if state["training"]:
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
                # Not enough new data — skip but update status
                continue

            train_once()
            last_trained_rows = db.get_reading_count()

        except Exception:
            print(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI  — Render needs an HTTP server
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Poultry Farm ML Server")


@app.get("/")
async def root():
    with _lock:
        s = dict(state)
    return JSONResponse({"service": "Poultry Farm ML", "state": s})


@app.get("/health")
async def health():
    """Ping this every 5min with UptimeRobot to keep Render free tier awake."""
    return JSONResponse({"status": "ok", "ts": datetime.utcnow().isoformat()})


@app.get("/status")
async def get_status():
    with _lock:
        return JSONResponse(dict(state))


@app.post("/retrain")
async def force_retrain():
    """POST to /retrain to manually trigger a training cycle."""
    _event.set()
    return JSONResponse({"message": "Retrain triggered"})


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Poultry Farm Cloud ML Server")
    print(f"  Firebase : {db.FIREBASE_URL}")
    print(f"  Retrains : every {RETRAIN_EVERY} rows or {TRAIN_INTERVAL}s")
    print(f"  Min rows : {MIN_ROWS}")
    print("=" * 60)

    # Start training background thread
    threading.Thread(target=training_loop, daemon=True).start()

    # Render injects PORT env var; default 8000 for local testing
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
