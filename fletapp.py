"""
fletapp.py  —  Poultry Farm AI Dashboard  (APK / Browser)
===========================================================
PURE UI — Zero ML on device.
All AI results come from Firebase (written by cloud_ml_server.py).

FLOW:
  ESP32 → Firebase /latest + /readings
  Cloud ML server → Firebase /ml_result + /forecast_7d
  This app reads Firebase every 3s → displays everything live

BUILD APK:
  flet build apk --project "Poultry Farm AI"

BROWSER MODE:
  python fletapp.py --web   →  open http://<ip>:8550 on any device

WORKS OFFLINE:
  Last Firebase data cached in memory — shows stale indicator
"""

import flet as ft
import threading
import time
import io
import base64
import traceback
import sys
import warnings
from datetime import datetime, timedelta

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

import firebase_db as db

# ═════════════════════════════════════════════════════════════════════════════
# COLORS
# ═════════════════════════════════════════════════════════════════════════════
BG      = "#0e1117"
SURFACE = "#262730"
SURF2   = "#1e2130"
SURF3   = "#161b22"
BORDER  = "#3d4257"
TEXT    = "#fafafa"
MUTED   = "#a0aec0"
BLUE    = "#4da3ff"
GREEN   = "#4ade80"
RED     = "#ff4b4b"
AMBER   = "#f59e0b"
CFEED   = "#50C8FF"
CWATER  = "#1f77b4"


# ═════════════════════════════════════════════════════════════════════════════
# CHART RENDERER
# ═════════════════════════════════════════════════════════════════════════════
def render_chart(x, yf, yw, title="", fx=None, ff=None, fw=None) -> str:
    n = len(x)
    if n == 0:
        return ""
    fig, ax = plt.subplots(figsize=(13, 4.6), facecolor=BG)
    ax.set_facecolor(BG)
    if n == 1:
        ax.scatter(x, yf, color=CFEED,  s=80, zorder=5, label="Feed / Weight")
        ax.scatter(x, yw, color=CWATER, s=80, zorder=5, label="Water (L)")
    else:
        ax.plot(x, yf, color=CFEED,  lw=2.2, label="Feed / Weight", marker="o", ms=3)
        ax.plot(x, yw, color=CWATER, lw=2.2, label="Water (L)",     marker="o", ms=3)
    if fx is not None and len(fx) > 0:
        lx = x.iloc[-1] if hasattr(x, "iloc") else x[-1]
        ax.axvline(x=lx, color=BORDER, lw=1, ls="--", alpha=0.5)
        ax.plot(fx, ff, color=CFEED,  lw=1.8, ls="--", alpha=0.7, label="Feed Forecast")
        ax.plot(fx, fw, color=CWATER, lw=1.8, ls="--", alpha=0.7, label="Water Forecast")
        ax.scatter(fx, ff, color=CFEED,  s=55, zorder=6, alpha=0.8)
        ax.scatter(fx, fw, color=CWATER, s=55, zorder=6, alpha=0.8)
    if n <= 2:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    elif n <= 48:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, n // 12)))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=28, color=MUTED, fontsize=9, ha="right")
    plt.yticks(color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=BORDER, lw=0.6, ls="--", alpha=0.7)
    ax.grid(axis="x", color=BORDER, lw=0.3, ls=":",  alpha=0.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.26),
              ncol=2, frameon=False, labelcolor=MUTED, fontsize=10)
    if title:
        ax.set_title(title, color=MUTED, fontsize=11, pad=8)
    plt.tight_layout(pad=1.8)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=140, facecolor=BG)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ═════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def divider():
    return ft.Container(height=1, bgcolor=BORDER,
                        margin=ft.margin.symmetric(vertical=18))


def hdr(emoji, title, sub=""):
    items = [ft.Row([ft.Text(emoji, size=20),
                     ft.Text(title, size=20, weight=ft.FontWeight.W_600, color=TEXT)],
                    spacing=8)]
    if sub:
        items.append(ft.Text(sub, size=12, color=MUTED))
    return ft.Column(items, spacing=2, tight=True)


def pill(text, color, bg):
    return ft.Container(
        content=ft.Text(text, size=12, color=color, weight=ft.FontWeight.W_500),
        bgcolor=bg,
        padding=ft.padding.symmetric(horizontal=10, vertical=4),
        border_radius=20,
        border=ft.border.all(1, color + "55"))


def safe_s(v):
    try:
        if isinstance(v, pd.Timestamp): return str(v.date())
        if isinstance(v, float):        return f"{v:.4f}"
        return str(v)
    except Exception:
        return str(v)


def mk_table(df, idx=False):
    d = df.reset_index(drop=True)
    cols = ([ft.DataColumn(ft.Text("#", color=MUTED, size=11))] if idx else []) + [
        ft.DataColumn(ft.Text(str(c), weight=ft.FontWeight.W_600, color=MUTED, size=12))
        for c in d.columns
    ]
    rows = []
    for i, row in d.iterrows():
        cells = ([ft.DataCell(ft.Text(str(i), size=11, color=MUTED))] if idx else [])
        cells += [ft.DataCell(ft.Text(safe_s(v), size=12, color=TEXT)) for v in row]
        rows.append(ft.DataRow(cells=cells,
                               color={"": SURFACE if i % 2 == 0 else SURF2}))
    return ft.DataTable(columns=cols, rows=rows,
                        heading_row_color={"": SURF3},
                        heading_row_height=38,
                        data_row_max_height=36,
                        column_spacing=22)


def ins_block(title, emoji, data_dict):
    """Render a pattern dict {feed_kg: {0: v,...}, water_liters:{...}} as table."""
    if not data_dict:
        return ft.Container()
    try:
        df = pd.DataFrame(data_dict)
        df.index.name = df.index.name or "group"
    except Exception:
        return ft.Container()
    return ft.Column([
        ft.Row([ft.Text(emoji, size=15),
                ft.Text(title, size=15, weight=ft.FontWeight.W_600, color=TEXT)], spacing=6),
        ft.Container(content=mk_table(df.reset_index()),
                     border_radius=8, border=ft.border.all(1, BORDER),
                     bgcolor=SURF2, padding=6),
    ], spacing=6)


def conf_bar(value):
    labels = {(0.80, 1.01): ("High", GREEN),
              (0.55, 0.80): ("Medium", AMBER),
              (0.30, 0.55): ("Low", RED),
              (0.00, 0.30): ("Very Low", RED)}
    label, color = "Very Low", RED
    for (lo, hi), (lbl, clr) in labels.items():
        if lo <= value < hi:
            label, color = lbl, clr
    pct   = int(value * 100)
    bar_w = max(4, int(280 * value))
    return ft.Column([
        ft.Row([ft.Text("Confidence:", size=12, color=MUTED),
                ft.Text(f"{pct}%  —  {label}", size=12, color=color,
                        weight=ft.FontWeight.W_600)], spacing=6),
        ft.Stack([
            ft.Container(height=6, border_radius=4, bgcolor=SURF3, width=280),
            ft.Container(height=6, border_radius=4, bgcolor=color, width=bar_w),
        ]),
    ], spacing=4, tight=True)


def make_card_row(items, page_width, cols=4, gap=12):
    """
    Build a row of metric cards using fixed pixel widths.
    Safe for desktop, browser, and Android APK.
    cols collapses to 2 on mobile, 1 on very narrow.
    """
    if page_width < 420:
        cols = 1
    elif page_width < 700:
        cols = 2
    avail = max(200, page_width - 56 - gap * (cols - 1))
    cw    = avail // cols
    card_rows, row = [], []
    for label, ctrl in items:
        row.append(ft.Container(
            content=ft.Column([ft.Text(label, size=12, color=MUTED), ctrl],
                              spacing=5, tight=True),
            bgcolor=SURFACE, padding=ft.padding.all(18),
            border_radius=8, border=ft.border.all(1, BORDER),
            width=cw))
        if len(row) == cols:
            card_rows.append(ft.Row(row[:], spacing=gap))
            row = []
    if row:
        card_rows.append(ft.Row(row, spacing=gap))
    return ft.Column(card_rows, spacing=gap)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════════════════
def main(page: ft.Page):
    page.title      = "Poultry Farm AI Dashboard"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor    = BG
    page.padding    = ft.padding.symmetric(horizontal=28, vertical=24)
    page.scroll     = ft.ScrollMode.AUTO
    try:
        page.window.width     = 1280
        page.window.min_width = 380
        page.window.height    = 900
    except Exception:
        try:
            page.window_width = 1280
        except Exception:
            pass

    S = {"running": True, "no_flow": None}

    def su():
        try: page.update()
        except Exception: pass

    def pw():
        try:
            return max(380, page.window.width or 1280)
        except Exception:
            try: return max(380, page.width or 1280)
            except Exception: return 1280

    # ══════════════════════════════════════════════════════════════════════════
    # BANNER
    # ══════════════════════════════════════════════════════════════════════════
    cl_ic   = ft.Icon(ft.Icons.CLOUD_DONE_ROUNDED, color=GREEN, size=16)
    cl_st   = ft.Text("Connecting to Firebase…",   size=13, color=MUTED)
    db_lbl  = ft.Text("0 readings",                size=12, color=MUTED)
    ml_lbl  = ft.Text("Cloud ML: connecting…",     size=12, color=MUTED)
    off_pill= ft.Container(
        content=ft.Text("OFFLINE — cached", size=11, color=AMBER),
        bgcolor="#2a2210",
        padding=ft.padding.symmetric(horizontal=8, vertical=3),
        border_radius=12, border=ft.border.all(1, AMBER+"55"), visible=False)

    banner = ft.Container(
        content=ft.Column([
            ft.Row([cl_ic, cl_st, ft.Container(expand=True), off_pill], spacing=8),
            ft.Row([
                ft.Icon(ft.Icons.STORAGE_ROUNDED,    color=MUTED, size=13), db_lbl,
                ft.Text("  |  ", color=BORDER, size=11),
                ft.Icon(ft.Icons.PSYCHOLOGY_ROUNDED, color=MUTED, size=13), ml_lbl,
            ], spacing=5),
        ], spacing=6),
        bgcolor=SURF2,
        padding=ft.padding.symmetric(horizontal=18, vertical=14),
        border_radius=8, border=ft.border.all(1, GREEN+"44"))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — LIVE SENSOR MONITORING
    # ══════════════════════════════════════════════════════════════════════════
    v_wt = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    v_fl = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    v_lv = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    v_tl = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    v_ls = ft.Text("",   size=11, color=MUTED, italic=True)

    sensor_ctr = ft.Container()

    def rebuild_sensors():
        sensor_ctr.content = make_card_row([
            ("Current Weight",  v_wt),
            ("Water Flow Rate", v_fl),
            ("Water Level",     v_lv),
            ("Total Water (L)", v_tl),
        ], pw(), cols=4)

    # AI STATUS CARD  — shows cloud ML result
    ai_lbl  = ft.Text("🤖  AI Status",      size=13, color=MUTED)
    ai_main = ft.Text("Waiting for Cloud ML…", size=18, weight=ft.FontWeight.W_600, color=MUTED)
    ai_feed = ft.Text("", size=14, color=CFEED)
    ai_watr = ft.Text("", size=14, color="#60a5fa")
    ai_date = ft.Text("", size=12, color=MUTED, italic=True)
    ai_trnd = ft.Row([], spacing=8, visible=False)
    ai_spin = ft.ProgressBar(color=BLUE, bgcolor=SURF2, value=0, visible=False)

    ai_card = ft.Container(
        content=ft.Column([
            ai_lbl, ai_main, ai_feed, ai_watr, ai_date, ai_trnd, ai_spin,
        ], spacing=5, tight=True),
        bgcolor=SURFACE, padding=ft.padding.all(20),
        border_radius=8, border=ft.border.all(1, BORDER))

    def do_tare(_):
        tare_btn.text = "Tare Sent ✓"; tare_btn.bgcolor = GREEN+"22"; su()
        def _r():
            time.sleep(3); tare_btn.text = "⚖️  Tare Scale"
            tare_btn.bgcolor = SURFACE; su()
        threading.Thread(target=_r, daemon=True).start()

    tare_btn = ft.ElevatedButton(
        "⚖️  Tare Scale", bgcolor=SURFACE, color=MUTED, on_click=do_tare,
        style=ft.ButtonStyle(side=ft.BorderSide(1, BORDER),
                             padding=ft.padding.symmetric(horizontal=14, vertical=9)))

    al_ic = ft.Icon(ft.Icons.WIFI_FIND_ROUNDED, color=MUTED, size=18)
    al_mg = ft.Text("Waiting for ESP32 data in Firebase…", size=13, color=MUTED)
    al_bx = ft.Container(
        content=ft.Row([al_ic, al_mg], spacing=8),
        bgcolor=SURF2, padding=ft.padding.all(13),
        border_radius=6, border=ft.border.all(1, BORDER))

    live_sec = ft.Column([
        ft.Row([hdr("📡", "Live Sensor Monitoring", "Real-time from Firebase"),
                ft.Container(expand=True), tare_btn]),
        sensor_ctr,
        ai_card, al_bx, v_ls,
    ], spacing=14)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — TITLE
    # ══════════════════════════════════════════════════════════════════════════
    title_sec = ft.Column([
        divider(),
        ft.Row([ft.Text("🐔", size=34),
                ft.Column([
                    ft.Text("Poultry Farm AI Dashboard",
                            size=24, weight=ft.FontWeight.BOLD, color=TEXT),
                    ft.Text("ESP32 → Firebase → Cloud ML → APK  •  Fully Standalone",
                            size=12, color=MUTED),
                ], spacing=2, tight=True)], spacing=12),
    ], spacing=0)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — FIREBASE DATABASE RECORDS
    # ══════════════════════════════════════════════════════════════════════════
    db_col  = ft.Column(scroll=ft.ScrollMode.AUTO)
    db_cnt2 = ft.Text("0 total readings", size=13, color=MUTED)

    def refresh_db():
        readings = db.get_readings(limit=100)
        df = db.readings_to_df(readings) if readings else pd.DataFrame()
        db_col.controls.clear()
        if df.empty:
            db_col.controls.append(
                ft.Text("No readings yet — waiting for ESP32…", color=MUTED, size=13))
        else:
            disp = df[["date","feed_kg","water_liters","flow","level"]].tail(100).copy()
            disp["date"] = disp["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
            db_col.controls.append(mk_table(disp, idx=True))
        db_cnt2.value = f"{db.get_reading_count():,} total readings"
        su()

    db_ref = ft.IconButton(ft.Icons.REFRESH_ROUNDED, icon_color=MUTED, tooltip="Refresh",
                            on_click=lambda _: threading.Thread(target=refresh_db, daemon=True).start())

    db_sec = ft.Column([
        divider(),
        ft.Row([hdr("🗄️", "Firebase Database Records", "Live data from ESP32"),
                ft.Container(expand=True), db_cnt2, db_ref]),
        ft.Container(content=db_col, height=320, border_radius=8,
                     border=ft.border.all(1, BORDER), bgcolor=SURF2, padding=8,
                     clip_behavior=ft.ClipBehavior.HARD_EDGE),
    ], spacing=10)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — ARIMA FORECAST  (from cloud ML)
    # ══════════════════════════════════════════════════════════════════════════
    ar_fv   = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    ar_wv   = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    ar_info = ft.Text("", size=12, color=MUTED)
    arima_ctr = ft.Container()

    def rebuild_arima():
        arima_ctr.content = make_card_row([
            ("Feed Forecast",  ar_fv),
            ("Water Forecast", ar_wv),
        ], pw(), cols=2)

    arima_sec = ft.Column([
        divider(),
        hdr("🔮", "Forecast (ARIMA / Time Series)", "From cloud ML server"),
        arima_ctr, ar_info,
    ], spacing=12, visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — AI PREDICTIONS  (from cloud ML via Firebase)
    # ══════════════════════════════════════════════════════════════════════════
    ai_fv   = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    ai_wv   = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    pd_txt  = ft.Text("Prediction Date: N/A", color=BLUE, size=13)
    ai_st   = ft.Text("Fetching cloud predictions…", size=13, color=MUTED)
    ai_sb   = ft.Container(
        content=ai_st, bgcolor=SURF2, padding=ft.padding.all(13),
        border_radius=6, border=ft.border.all(1, BORDER))
    pred_ctr= ft.Container()
    conf_col= ft.Column([], visible=False)
    pd_bx   = ft.Container(
        content=ft.Row([ft.Icon(ft.Icons.CALENDAR_MONTH_ROUNDED, size=15, color=BLUE),
                        pd_txt], spacing=8),
        bgcolor="#162133", padding=ft.padding.all(13), border_radius=6,
        border=ft.border.all(1, BLUE+"44"), visible=False)

    def rebuild_pred():
        pred_ctr.content = make_card_row([
            ("Feed Consumption",  ai_fv),
            ("Water Consumption", ai_wv),
        ], pw(), cols=2)

    ai_pred_sec = ft.Column([
        divider(),
        hdr("🤖", "AI Predictions", "Computed by cloud ML — updated continuously"),
        ai_sb, pred_ctr, pd_bx, conf_col,
    ], spacing=13, visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — FARM INSIGHTS
    # ══════════════════════════════════════════════════════════════════════════
    ins_col = ft.Column(spacing=20)
    ins_sec = ft.Column([
        divider(),
        hdr("📈", "Farm Insights", "Behavioral patterns — from cloud ML"),
        ins_col,
    ], spacing=13, visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — 7-DAY FORECAST TABLE
    # ══════════════════════════════════════════════════════════════════════════
    fc7_wrap = ft.Container(visible=False, border_radius=8,
                             border=ft.border.all(1, BORDER), bgcolor=SURF2, padding=8)
    fc7_sec  = ft.Column([
        divider(),
        hdr("📅", "7-Day Forecast", "Auto-updated by cloud ML after each retrain"),
        fc7_wrap,
    ], spacing=13, visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — CONSUMPTION TRENDS CHART
    # ══════════════════════════════════════════════════════════════════════════
    chart_img = ft.Image(fit=ft.ImageFit.CONTAIN, border_radius=8, expand=True)
    chart_cap = ft.Text("", size=11, color=MUTED, italic=True)
    chart_sec = ft.Column([
        divider(),
        hdr("📊", "Consumption Trends (Feed vs Water)"),
        chart_cap,
        ft.Container(content=chart_img, bgcolor=SURF2, border_radius=10,
                     border=ft.border.all(1, BORDER), padding=ft.padding.all(16)),
    ], spacing=10, visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # RESIZE HANDLER
    # ══════════════════════════════════════════════════════════════════════════
    def on_resize(e):
        rebuild_sensors(); rebuild_arima(); rebuild_pred(); su()

    page.on_resized = on_resize

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE UPDATE LOOP
    # ══════════════════════════════════════════════════════════════════════════
    def live_loop():
        last_db = last_chart = last_ml = 0
        STALE   = 15

        while S["running"]:
            try:
                now    = time.time()
                latest = db.get_latest()
                cache  = db.get_cache_status()
                online = cache["online"]

                # Banner
                off_pill.visible = not online
                cl_ic.name  = ft.Icons.CLOUD_DONE_ROUNDED if online else ft.icons.CLOUD_OFF_ROUNDED
                cl_ic.color = GREEN if online else AMBER
                cl_st.value = "Connected to Firebase ✓" if online else "Firebase offline — cached data"
                cl_st.color = GREEN if online else AMBER

                # Sensor values
                if latest:
                    ts  = latest.get("timestamp", 0)
                    age = now - ts if ts else 9999
                    esp = age < STALE
                    flow= float(latest.get("flow", 0))

                    v_wt.value = f"{float(latest.get('weight',0)):.3f} kg"
                    v_fl.value = f"{flow:.2f} L/m"
                    v_lv.value = str(latest.get("level","--"))
                    v_tl.value = f"{float(latest.get('totalLiters',0)):.1f} L"
                    v_ls.value = (f"Last reading: {int(age)}s ago  •  "
                                  f"{db.get_reading_count():,} readings in Firebase")

                    if esp:
                        if flow <= 0:
                            if S["no_flow"] is None: S["no_flow"] = now
                            el = now - S["no_flow"]
                            if el >= 60:
                                m = int(el/60)
                                al_ic.name=ft.icons.REPORT_GMAILERRORRED_ROUNDED; al_ic.color=RED
                                al_mg.value=f"🚨 ALERT: No water flow for {m} minute{'s' if m!=1 else ''}!"
                                al_mg.color=RED; al_bx.bgcolor="#2a1a1a"; al_bx.border=ft.border.all(1,RED+"44")
                            else:
                                r=60-int(el)
                                al_ic.name=ft.icons.WARNING_AMBER_ROUNDED; al_ic.color=AMBER
                                al_mg.value=f"⚠️ Low Flow: monitoring for {r}s more…"
                                al_mg.color=AMBER; al_bx.bgcolor="#2a2210"; al_bx.border=ft.border.all(1,AMBER+"44")
                        else:
                            S["no_flow"]=None
                            al_ic.name=ft.icons.CHECK_CIRCLE_OUTLINE_ROUNDED; al_ic.color=GREEN
                            al_mg.value="✅ Water flow is stable."; al_mg.color=GREEN
                            al_bx.bgcolor="#1a2e1a"; al_bx.border=ft.border.all(1,GREEN+"44")
                    else:
                        S["no_flow"]=None
                        al_ic.name=ft.icons.WIFI_OFF_ROUNDED; al_ic.color=AMBER
                        al_mg.value=f"⚠️ ESP signal lost — last seen {int(age)}s ago"
                        al_mg.color=AMBER; al_bx.bgcolor=SURF2; al_bx.border=ft.border.all(1,BORDER)
                else:
                    v_wt.value=v_fl.value=v_lv.value=v_tl.value="--"; S["no_flow"]=None
                    al_ic.name=ft.icons.WIFI_FIND_ROUNDED; al_ic.color=MUTED
                    al_mg.value="📡 Waiting for ESP32 to post data to Firebase…"
                    al_mg.color=MUTED; al_bx.bgcolor=SURF2; al_bx.border=ft.border.all(1,BORDER)

                # Cloud ML result  ← reads Firebase /ml_result
                ml = db.get_ml_result()
                ml_status = db.get_ml_status()

                if ml_status:
                    s  = ml_status.get("status","--")
                    nr = ml_status.get("rows", 0)
                    ml_lbl.value = {
                        "ready":      f"Cloud ML: ✅ {nr:,} rows trained",
                        "training":   f"Cloud ML: 🔄 retraining on {nr:,} rows…",
                        "collecting": f"Cloud ML: collecting data ({nr} rows)…",
                        "error":      f"Cloud ML: ⚠️ {ml_status.get('error','error')[:40]}",
                        "waiting":    "Cloud ML: waiting for data…",
                    }.get(s, f"Cloud ML: {s}")

                if ml:
                    pf   = ml.get("feedKg",   0.0)
                    pw_  = ml.get("waterL",   0.0)
                    pd_  = ml.get("predDate", "N/A")
                    conf = ml.get("confidence", 0.0)
                    trend= ml.get("trend",    "stable")
                    tic  = ml.get("trendIcon","📊")
                    anom = ml.get("anomaly",  False)
                    amsg = ml.get("anomalyMsg","")
                    fd   = ml.get("feedDelta",  0.0)
                    wd   = ml.get("waterDelta", 0.0)
                    nr   = ml.get("modelRows",  0)
                    af   = ml.get("arimaFeed",  None)
                    aw   = ml.get("arimaWater", None)
                    ta   = ml.get("trainedAt",  "")

                    # AI card
                    ai_spin.visible = False
                    ai_main.value   = "🤖 Next Feed Prediction"
                    ai_main.color   = GREEN
                    ai_feed.value   = f"🌾  Feed:  {pf:.2f} kg"
                    ai_watr.value   = f"💧  Water: {pw_:.2f} L"
                    ai_date.value   = (f"For: {pd_}  •  {nr:,} readings  •  Conf: {int(conf*100)}%")
                    tc = {"stable":GREEN,"increasing":AMBER,"decreasing":BLUE,"warning":RED}.get(trend,MUTED)
                    tb = {"stable":"#1a2e1a","increasing":"#2a2210","decreasing":"#162133","warning":"#2a1a1a"}.get(trend,SURF2)
                    ai_trnd.controls=[pill(f"{tic} {trend.capitalize()}",tc,tb)]
                    if fd: ai_trnd.controls.append(ft.Text(f"Feed {'+' if fd>0 else ''}{fd:.1f}%",size=12,color=AMBER if fd>5 else MUTED))
                    if wd: ai_trnd.controls.append(ft.Text(f"Water {'+' if wd>0 else ''}{wd:.1f}%",size=12,color=AMBER if wd>5 else MUTED))
                    if anom: ai_trnd.controls.append(pill("⚠️ Anomaly",RED,"#2a1a1a"))
                    ai_trnd.visible=True

                    # ARIMA section
                    if af is not None:
                        ar_fv.value  = f"{af:.2f} kg"
                        ar_wv.value  = f"{aw:.2f} L"
                        ar_info.value= f"ARIMA projection  •  {nr:,} readings  •  {ta[:19]}"
                        rebuild_arima()
                        arima_sec.visible=True

                    # AI predictions section
                    ai_fv.value  = f"{pf:.2f} kg"
                    ai_wv.value  = f"{pw_:.2f} L"
                    pd_txt.value = f"Prediction Date: {pd_}"
                    pd_bx.visible= True
                    conf_col.controls=[conf_bar(conf)]; conf_col.visible=True
                    ai_st.value  = f"✅ Cloud ML ready — {nr:,} readings  •  updated {ta[:19]}"
                    ai_st.color  = GREEN
                    ai_sb.bgcolor= "#1a2e1a"; ai_sb.border=ft.border.all(1,GREEN+"44")
                    rebuild_pred()
                    ai_pred_sec.visible=True

                    # Insights
                    pat_sys  = ml.get("patSystem",  {})
                    pat_day  = ml.get("patDay",     {})
                    pat_month= ml.get("patMonth",   {})
                    ins_col.controls.clear()
                    if pat_sys:  ins_col.controls.append(ins_block("System Behavior","🐔",pat_sys))
                    if pat_day:  ins_col.controls.append(ins_block("Weekly Pattern","📅",pat_day))
                    if pat_month:ins_col.controls.append(ins_block("Monthly Pattern","📆",pat_month))
                    if ins_col.controls: ins_sec.visible=True
                else:
                    ai_spin.visible  = (ml_status or {}).get("status")=="training"
                    ai_main.value    = "Waiting for Cloud ML predictions…"
                    ai_main.color    = MUTED
                    ai_feed.value=ai_watr.value=ai_date.value=""
                    ai_trnd.visible=False

                # Banner DB count
                db_lbl.value = f"{db.get_reading_count():,} readings"

                # 7-day forecast
                fc7_raw = db.get_forecast_7d()
                if fc7_raw:
                    try:
                        fc7_df = pd.DataFrame(fc7_raw)
                        if not fc7_df.empty:
                            fc7_wrap.content=mk_table(fc7_df); fc7_wrap.visible=True; fc7_sec.visible=True
                    except Exception:
                        pass

                # DB table every 15s
                if now-last_db>15:
                    last_db=now
                    threading.Thread(target=refresh_db,daemon=True).start()

                # Chart every 20s
                if now-last_chart>20:
                    last_chart=now
                    rc=db.get_readings(limit=200); rdf=db.readings_to_df(rc)
                    if not rdf.empty:
                        fc7_raw2=db.get_forecast_7d()
                        fx=fw=ff=None
                        if fc7_raw2:
                            try:
                                fc7_df2=pd.DataFrame(fc7_raw2)
                                fx=pd.to_datetime(fc7_df2["date"]).tolist()
                                ff=fc7_df2["feed_kg"].tolist()
                                fw=fc7_df2["water_liters"].tolist()
                            except Exception: pass
                        chart_img.src_base64=render_chart(
                            rdf["date"],rdf["feed_kg"],rdf["water_liters"],
                            f"Last {len(rdf)} Readings from Firebase",fx=fx,ff=ff,fw=fw)
                        chart_cap.value=(f"{len(rdf)} readings  •  "
                            f"{rdf['date'].min().strftime('%Y-%m-%d %H:%M')} → "
                            f"{rdf['date'].max().strftime('%Y-%m-%d %H:%M')}  •  "
                            f"{'+ 7-day forecast' if fx else 'historical only'}")
                        chart_sec.visible=True

                su()

            except Exception:
                print(traceback.format_exc())

            time.sleep(3)

    threading.Thread(target=live_loop, daemon=True).start()
    page.on_disconnect = lambda e: S.update(running=False)

    # Initial build
    rebuild_sensors(); rebuild_arima(); rebuild_pred()

    page.add(
        banner, ft.Container(height=8),
        live_sec, title_sec, ft.Container(height=6),
        db_sec, arima_sec, ai_pred_sec,
        ins_sec, fc7_sec, chart_sec,
        ft.Container(height=48))

    threading.Thread(target=refresh_db, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    web = "--web" in sys.argv or "--browser" in sys.argv
    if web:
        import socket as _s
        try:
            _sk=_s.socket(_s.AF_INET,_s.SOCK_DGRAM); _sk.connect(("8.8.8.8",80))
            _ip=_sk.getsockname()[0]; _sk.close()
        except Exception:
            _ip="127.0.0.1"
        PORT=8550
        print(f"\n🌐 Browser mode — open: http://{_ip}:{PORT}\n")
        ft.app(target=main,view=ft.AppView.WEB_BROWSER,port=PORT,host="0.0.0.0")
    else:
        print("🖥  Desktop / APK mode  (add --web for browser)\n")
        ft.app(target=main)