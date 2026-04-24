"""
fletapp.py  —  Poultry Farm AI Dashboard  (APK / Browser)
===========================================================
PURE UI — Zero ML on device.
All AI results come from Firebase (written by cloud_ml_server.py).

FLOW:
  ESP32 → Firebase /latest + /readings
  Cloud ML → Firebase /ml_result + /forecast_7d
  This app reads Firebase every 3s → displays live

BUILD APK:
  flet build apk --project "Poultry Farm AI"

BROWSER MODE (phone on same WiFi):
  python fletapp.py --web

FIXES IN THIS VERSION:
  - ft.Icon.name reassignment uses string literals (e.g. "wifi_off_rounded")
    NOT ft.icons.X objects — that causes AttributeError at runtime
  - ft.icons constants used only for initial construction, strings for updates
  - All ft.Icons / ft.icons references unified to ft.icons (lowercase)
  - firebase_db URL matches ESP32 sketch
"""
import flet as ft
import threading
import time
import traceback
import sys
import warnings
from datetime import datetime

import firebase_db as db

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# ICON STRING CONSTANTS
# Using plain strings for .name reassignment avoids AttributeError.
# ft.icons.X works fine for initial ft.Icon(ft.icons.X) construction,
# but when you do icon_ctrl.name = ft.icons.X it may fail in some Flet
# versions. String literals always work.
# ═════════════════════════════════════════════════════════════════════════════
ICO_CLOUD_OK    = "cloud_done_rounded"
ICO_CLOUD_OFF   = "cloud_off_rounded"
ICO_WIFI_FIND   = "wifi_find_rounded"
ICO_WIFI_OFF    = "wifi_off_rounded"
ICO_CHECK       = "check_circle_outline_rounded"
ICO_WARNING     = "warning_amber_rounded"
ICO_ALERT       = "report_gmailerrorred_rounded"
ICO_STORAGE     = "storage_rounded"
ICO_PSYCH       = "psychology_rounded"
ICO_REFRESH     = "refresh_rounded"
ICO_CALENDAR    = "calendar_month_rounded"

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
        border=ft.border.all(1, color + "55"),
    )


def safe_s(v):
    try:
        if v is None:
            return "--"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    except Exception:
        return str(v)


def mk_table_list(rows, idx=False):
    if not rows:
        return ft.DataTable(
            columns=[ft.DataColumn(ft.Text("No data", color=MUTED))],
            rows=[]
        )

    keys = list(rows[0].keys())

    cols = []
    if idx:
        cols.append(ft.DataColumn(ft.Text("#", color=MUTED, size=11)))

    for k in keys:
        cols.append(
            ft.DataColumn(
                ft.Text(str(k), weight=ft.FontWeight.W_600, color=MUTED, size=11)
            )
        )

    data_rows = []
    for i, row in enumerate(rows):
        cells = []

        if idx:
            cells.append(ft.DataCell(ft.Text(str(i), size=11, color=MUTED)))

        for k in keys:
            cells.append(
                ft.DataCell(ft.Text(safe_s(row.get(k)), size=11, color=TEXT))
            )

        data_rows.append(
            ft.DataRow(
                cells=cells,
                color={"": SURFACE if i % 2 == 0 else SURF2}
            )
        )

    return ft.DataTable(
        columns=cols,
        rows=data_rows,
        heading_row_color={"": SURF3},
        heading_row_height=38,
        data_row_max_height=36,
        column_spacing=22,
    )


def ins_block(title, emoji, data_dict):
    """Render pattern dict/list from Firebase without pandas."""
    if not data_dict:
        return ft.Container()

    rows = []

    try:
        # CASE 1: {"feed_kg": {"4": 0.2137}, "water_liters": {"4": 0}}
        has_nested_dict = any(isinstance(v, dict) for v in data_dict.values())

        if has_nested_dict:
            keys = set()
            for col_data in data_dict.values():
                if isinstance(col_data, dict):
                    keys.update([str(k) for k in col_data.keys()])

            for key in sorted(keys):
                row = {"group": key}
                for col_name, col_data in data_dict.items():
                    if isinstance(col_data, dict):
                        val = col_data.get(key, col_data.get(str(key), "--"))
                        row[col_name] = val
                rows.append(row)

        # CASE 2: {"feed_kg": [None, 0.21367], "water_liters": [None, 0.0]}
        else:
            max_len = 0
            for v in data_dict.values():
                if isinstance(v, list):
                    max_len = max(max_len, len(v))

            if max_len > 0:
                for i in range(max_len):
                    row = {"group": i}
                    for col_name, values in data_dict.items():
                        if isinstance(values, list):
                            val = values[i] if i < len(values) else "--"
                            if val is not None:
                                row[col_name] = val
                    if len(row) > 1:
                        rows.append(row)
            else:
                for k, v in data_dict.items():
                    rows.append({"name": k, "value": v})

    except Exception as e:
        rows = [{"error": str(e)}]

    return ft.Column([
        ft.Row([
            ft.Text(emoji, size=15),
            ft.Text(title, size=15, weight=ft.FontWeight.W_600, color=TEXT),
        ], spacing=6),

        ft.Container(
            content=ft.Row(
                [mk_table_list(rows)],
                scroll=ft.ScrollMode.AUTO
            ),
            border_radius=8,
            border=ft.border.all(1, BORDER),
            bgcolor=SURF2,
            padding=6,
        ),
    ], spacing=6)


def conf_bar(value):
    """Confidence bar widget."""
    if value >= 0.80:   label, color = "High",     GREEN
    elif value >= 0.55: label, color = "Medium",   AMBER
    elif value >= 0.30: label, color = "Low",      RED
    else:               label, color = "Very Low", RED
    pct   = int(value * 100)
    bar_w = max(4, int(280 * value))
    return ft.Column([
        ft.Row([
            ft.Text("Confidence:", size=12, color=MUTED),
            ft.Text(f"{pct}%  —  {label}", size=12, color=color,
                    weight=ft.FontWeight.W_600),
        ], spacing=6),
        ft.Stack([
            ft.Container(height=6, border_radius=4, bgcolor=SURF3, width=280),
            ft.Container(height=6, border_radius=4, bgcolor=color, width=bar_w),
        ]),
    ], spacing=4, tight=True)


def make_card_row(items, page_width, cols=4, gap=12):
    """
    Build metric cards with fixed pixel widths.
    Responsive: 4 cols desktop, 2 cols tablet, 1 col mobile.
    SAFE for Flet desktop / browser / Android APK.
    Does NOT use wrap=True or expand=True (avoids grey-box bug).
    """
    if page_width < 850:
        cols = 1
    elif page_width < 1100:
        cols = 2
    else:
        cols = 4
    avail = max(200, page_width - 56 - gap * (cols - 1))
    cw    = avail // cols
    card_rows, row = [], []
    for label, ctrl in items:
        row.append(ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        label,
                        size=11 if page_width < 850 else 12,
                        color=MUTED,
                        max_lines=1,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                    ctrl
                ],
                spacing=5, tight=True),
            bgcolor=SURFACE,
            padding=ft.padding.all(12 if page_width < 850 else 18),
            border_radius=8,
            border=ft.border.all(1, BORDER),
            width=cw,
        ))
        if len(row) == cols:
            card_rows.append(ft.Row(row[:], spacing=gap))
            row = []
    if row:
        card_rows.append(ft.Row(row, spacing=gap))
    return ft.Column(card_rows, spacing=gap)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════════════════
def is_mobile_width(width):
    return width < 850

def main(page: ft.Page):
    page.title      = "Poultry Farm AI Dashboard"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor    = BG
    page.padding = ft.padding.symmetric(
        horizontal=8 if is_mobile_width(page.width or 1280) else 28,
        vertical=10 if is_mobile_width(page.width or 1280) else 24
    )
    page.scroll     = ft.ScrollMode.AUTO
    try:
        page.window.min_width = 380
        page.window.height = 900
    except Exception:
        pass

    S = {"running": True, "no_flow": None}
    update_lock = threading.Lock()

    def su():
        try:
            with update_lock:
                page.update()
        except Exception as e:
            print("PAGE UPDATE ERROR:", e)

    def pw():
        try:
            if page.width:
                return max(320, page.width)
        except Exception:
            pass

        try:
            if page.window.width:
                return max(320, page.window.width)
        except Exception:
            pass

        return 380

    def is_mobile():
        return pw() < 850

    # ══════════════════════════════════════════════════════════════════════════
    # BANNER
    # ══════════════════════════════════════════════════════════════════════════
    # Use ICO_* string constants for icon that gets .name reassigned later
    cl_ic   = ft.Icon(ICO_CLOUD_OK,  color=GREEN, size=16)
    cl_st   = ft.Text("Connecting to Firebase…", size=13, color=MUTED)
    db_lbl  = ft.Text("0 readings",              size=12, color=MUTED)
    ml_lbl  = ft.Text("Cloud ML: connecting…",   size=12, color=MUTED)
    off_pill = ft.Container(
        content=ft.Text("OFFLINE — cached", size=11, color=AMBER),
        bgcolor="#2a2210",
        padding=ft.padding.symmetric(horizontal=8, vertical=3),
        border_radius=12,
        border=ft.border.all(1, AMBER + "55"),
        visible=False,
    )

    banner = ft.Container(
        content=ft.Column([
            ft.Row([cl_ic, cl_st, ft.Container(expand=True), off_pill], spacing=8),
            ft.Row([
                ft.Icon(ICO_STORAGE, color=MUTED, size=13), db_lbl,
                ft.Text("  |  ", color=BORDER, size=11),
                ft.Icon(ICO_PSYCH,   color=MUTED, size=13), ml_lbl,
            ], spacing=5),
        ], spacing=6),
        bgcolor=SURF2,
        padding=ft.padding.symmetric(
            horizontal=10 if is_mobile() else 18,
            vertical=10 if is_mobile() else 14
        ),
        border_radius=8,
        border=ft.border.all(1, GREEN + "44"),
    )

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

    # AI STATUS CARD
    ai_lbl  = ft.Text("🤖  AI Status",          size=13, color=MUTED)
    ai_main = ft.Text("Waiting for Cloud ML…",  size=18,
                      weight=ft.FontWeight.W_600, color=MUTED)
    ai_feed = ft.Text("", size=14, color=CFEED)
    ai_watr = ft.Text("", size=14, color="#60a5fa")
    ai_date = ft.Text("", size=12, color=MUTED, italic=True)
    ai_trnd = ft.Row([], spacing=8, visible=False)
    ai_spin = ft.ProgressBar(color=BLUE, bgcolor=SURF2, value=0, visible=False)

    ai_card = ft.Container(
        content=ft.Column(
            [ai_lbl, ai_main, ai_feed, ai_watr, ai_date, ai_trnd, ai_spin],
            spacing=5, tight=True),
        bgcolor=SURFACE,
        padding=ft.padding.all(20),
        border_radius=8,
        border=ft.border.all(1, BORDER),
    )

    def do_tare(_):
        tare_btn.text    = "Tare Sent ✓"
        tare_btn.bgcolor = GREEN + "22"
        su()
        def _r():
            time.sleep(3)
            tare_btn.text    = "⚖️  Tare Scale"
            tare_btn.bgcolor = SURFACE
            su()
        threading.Thread(target=_r, daemon=True).start()

    tare_btn = ft.ElevatedButton(
        "⚖️  Tare Scale", bgcolor=SURFACE, color=MUTED, on_click=do_tare,
        style=ft.ButtonStyle(
            side=ft.BorderSide(1, BORDER),
            padding=ft.padding.symmetric(horizontal=14, vertical=9),
        ),
    )

    # Alert box — use ICO_* strings for icons that get .name reassigned
    al_ic = ft.Icon(ICO_WIFI_FIND, color=MUTED, size=18)
    al_mg = ft.Text("Waiting for ESP32 data in Firebase…", size=13, color=MUTED)
    al_bx = ft.Container(
        content=ft.Row([al_ic, al_mg], spacing=8),
        bgcolor=SURF2, padding=ft.padding.all(13),
        border_radius=6, border=ft.border.all(1, BORDER),
    )

    live_sec = ft.Column([
        ft.Column([
            hdr("📡", "Live Sensor Monitoring", "Real-time from Firebase"),
            tare_btn
        ]) if is_mobile() else ft.Row([
            hdr("📡", "Live Sensor Monitoring", "Real-time from Firebase"),
            ft.Container(expand=True),
            tare_btn,
        ]),
        sensor_ctr,
        ai_card,
        al_bx,
        v_ls,
    ], spacing=10 if is_mobile() else 14)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — TITLE
    # ══════════════════════════════════════════════════════════════════════════
    title_sec = ft.Column([
        divider(),
        ft.Row([
            ft.Text("🐔", size=34),
            ft.Column([
                ft.Text(
                    "Poultry Farm AI Dashboard",
                    size=18 if is_mobile() else 24, weight=ft.FontWeight.BOLD, color=TEXT),
                ft.Text("ESP32 → Firebase → Cloud ML → APK ",
                        size=12, color=MUTED),
            ], spacing=2, tight=True),
        ], spacing=12),
    ], spacing=0)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — FIREBASE DATABASE RECORDS
    # ══════════════════════════════════════════════════════════════════════════
    db_col  = ft.Column(scroll=ft.ScrollMode.AUTO)
    db_cnt2 = ft.Text("0 total readings", size=13, color=MUTED)

    def refresh_db():
        readings = db.get_readings(limit=100)
        db_col.controls.clear()

        if not readings:
            db_col.controls.append(
                ft.Text("No readings yet — waiting for ESP32…", color=MUTED, size=13)
            )
        else:
            rows = []
            for r in readings[-100:]:
                rows.append({
                    "date": str(r.get("ts", ""))[:19],
                    "feed_kg": float(r.get("weight", 0)),
                    "water_liters": float(r.get("totalLiters", 0)),
                    "flow": float(r.get("flow", 0)),
                    "level": str(r.get("level", "--")),
                })

            db_col.controls.append(
                ft.Row(
                    [mk_table_list(rows, idx=True)],
                    scroll=ft.ScrollMode.AUTO
                )
            )
        db_cnt2.value = f"{db.get_reading_count():,} total readings"
        # su()

    db_ref = ft.IconButton(
        ICO_REFRESH, icon_color=MUTED, tooltip="Refresh",
        on_click=lambda _: threading.Thread(target=refresh_db, daemon=True).start(),
    )

    db_sec = ft.Column([
        divider(),
        ft.Column([
            hdr("🗄️", "Firebase Database Records", "Live data from ESP32"),

            ft.Row([
                db_cnt2,
                ft.Container(expand=True),
                db_ref,
            ])
        ], spacing=8),

        ft.Container(
            content=db_col,
            height=200 if is_mobile() else 320,
            border_radius=8,
            border=ft.border.all(1, BORDER),
            bgcolor=SURF2,
            padding=8,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        ),
    ], spacing=10)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — ARIMA FORECAST (from cloud ML)
    # ══════════════════════════════════════════════════════════════════════════
    ar_fv     = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    ar_wv     = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    ar_info   = ft.Text("",   size=12, color=MUTED)
    arima_ctr = ft.Container()

    def rebuild_arima():
        arima_ctr.content = make_card_row([
            ("Feed Forecast",  ar_fv),
            ("Water Forecast", ar_wv),
        ], pw(), cols=2)

    arima_sec = ft.Column([
        divider(),
        hdr("🔮", "Forecast" if is_mobile() else "Forecast (ARIMA / Time Series)", "From cloud ML server"),
        arima_ctr,
        ar_info,
    ], spacing=12, visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — AI PREDICTIONS (from cloud ML via Firebase)
    # ══════════════════════════════════════════════════════════════════════════
    ai_fv    = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    ai_wv    = ft.Text("--", size=28, weight=ft.FontWeight.W_500, color=TEXT)
    pd_txt   = ft.Text("Prediction Date: N/A", color=BLUE, size=13)
    ai_st    = ft.Text("Fetching cloud predictions…", size=13, color=MUTED)
    ai_sb    = ft.Container(
        content=ai_st, bgcolor=SURF2,
        padding=ft.padding.all(13), border_radius=6,
        border=ft.border.all(1, BORDER),
    )
    pred_ctr = ft.Container()
    conf_col = ft.Column([], visible=False)
    pd_bx    = ft.Container(
        content=ft.Row([ft.Icon(ICO_CALENDAR, size=15, color=BLUE), pd_txt], spacing=8),
        bgcolor="#162133", padding=ft.padding.all(13), border_radius=6,
        border=ft.border.all(1, BLUE + "44"), visible=False,
    )

    def rebuild_pred():
        pred_ctr.content = make_card_row([
            ("Feed Consumption",  ai_fv),
            ("Water Consumption", ai_wv),
        ], pw(), cols=2)

    ai_pred_sec = ft.Column([
        divider(),
        hdr(
            "🤖",
            "AI Predictions",
            "Cloud ML results" if is_mobile() else "Computed by cloud ML — updated continuously"
        ),
        ai_sb, pred_ctr, pd_bx, conf_col,
    ], spacing=13, visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — FARM INSIGHTS
    # ══════════════════════════════════════════════════════════════════════════
    ins_col = ft.Column(spacing=20)
    ins_sec = ft.Column([
        divider(),
        hdr("📈", "Farm Insights", "Behavioral patterns from cloud ML"),
        ins_col,
    ], spacing=13, visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — 7-DAY FORECAST TABLE
    # ══════════════════════════════════════════════════════════════════════════
    fc7_wrap = ft.Container(
        visible=False, border_radius=8,
        border=ft.border.all(1, BORDER), bgcolor=SURF2, padding=8,
    )
    fc7_sec = ft.Column([
        divider(),
        hdr(
            "📅",
            "7-Day Forecast",
            "Cloud ML forecast" if is_mobile() else "Auto-updated by cloud ML after each retrain"
        ),
        fc7_wrap,
    ], spacing=13, visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — CONSUMPTION TRENDS CHART
    # ══════════════════════════════════════════════════════════════════════════
    chart_img = ft.Image(
        src_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0d8AAAAASUVORK5CYII=",
        fit="contain",
        border_radius=8,
        height=240 if is_mobile() else 360,
    )

    chart_cap = ft.Text(
        "",
        size=10 if is_mobile() else 11,
        color=MUTED,
        italic=True,
        max_lines=3 if is_mobile() else 1,
        overflow=ft.TextOverflow.ELLIPSIS,
    )

    chart_sec = ft.Column([
        divider(),

        ft.Row([
            ft.Text("📊", size=18 if is_mobile() else 20),

            ft.Column([
                ft.Text(
                    "Consumption Trends",
                    size=16 if is_mobile() else 20,
                    weight=ft.FontWeight.W_600,
                    color=TEXT,
                ),
                ft.Text(
                    "Feed vs Water",
                    size=10 if is_mobile() else 12,
                    color=MUTED,
                ),
            ], spacing=0, tight=True),

        ], spacing=8),

        chart_cap,

        ft.Container(
            content=ft.Column(
                [
                    chart_img,
                    ft.Text(
                        "Chart appears after Render creates chartB64",
                        size=10,
                        color=MUTED,
                        visible=True,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=SURF2,
            border_radius=10,
            border=ft.border.all(1, BORDER),
            padding=ft.padding.symmetric(
                horizontal=2 if is_mobile() else 12,
                vertical=4 if is_mobile() else 12,
            ),
            alignment=ft.alignment.center,
        ),

    ], spacing=8 if is_mobile() else 10, visible=False)

    # ══════════════════════════════════════════════════════════════════════════
    # RESIZE HANDLER
    # ══════════════════════════════════════════════════════════════════════════
    def on_resize(e):
        rebuild_sensors()
        rebuild_arima()
        rebuild_pred()

        su()

    # page.on_resized = on_resize

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE UPDATE LOOP — polls Firebase every 3s
    # ══════════════════════════════════════════════════════════════════════════
    def live_loop():
        last_db = last_chart = 0
        STALE   = 15           # seconds before ESP marked as lost

        while True:
            if not S["running"]:
                break
            try:
                now    = time.time()
                latest = db.get_latest()
                cache  = db.get_cache_status()
                online = cache["online"]

                # ── Banner ────────────────────────────────────────────────────
                off_pill.visible = not online
                cl_ic.name  = ICO_CLOUD_OK  if online else ICO_CLOUD_OFF
                cl_ic.color = GREEN         if online else AMBER
                cl_st.value = ("Connected to Firebase ✓"
                               if online else "Firebase offline — cached data")
                cl_st.color = GREEN if online else AMBER

                # ── Sensor values ─────────────────────────────────────────────
                if latest:
                    ts   = latest.get("timestamp", 0)
                    age  = now - float(ts) if ts else 9999
                    esp  = age < STALE
                    flow = float(latest.get("flow", 0))

                    v_wt.value = f"{float(latest.get('weight',     0)):.3f} kg"
                    v_fl.value = f"{flow:.2f} L/m"
                    v_lv.value = str(latest.get("level", "--"))
                    v_tl.value = f"{float(latest.get('totalLiters', 0)):.1f} L"
                    v_ls.value = (f"Last reading: {int(age)}s ago  •  "
                                  f"{db.get_reading_count():,} readings in Firebase")

                    if esp:
                        if flow <= 0:
                            if S["no_flow"] is None:
                                S["no_flow"] = now
                            el = now - S["no_flow"]
                            if el >= 60:
                                m = int(el / 60)
                                al_ic.name    = ICO_ALERT
                                al_ic.color   = RED
                                al_mg.value   = (f"🚨 ALERT: No water flow for "
                                                 f"{m} minute{'s' if m!=1 else ''}!")
                                al_mg.color   = RED
                                al_bx.bgcolor = "#2a1a1a"
                                al_bx.border  = ft.border.all(1, RED + "44")
                            else:
                                r = 60 - int(el)
                                al_ic.name    = ICO_WARNING
                                al_ic.color   = AMBER
                                al_mg.value   = f"⚠️ Low Flow: monitoring for {r}s more…"
                                al_mg.color   = AMBER
                                al_bx.bgcolor = "#2a2210"
                                al_bx.border  = ft.border.all(1, AMBER + "44")
                        else:
                            S["no_flow"] = None
                            al_ic.name    = ICO_CHECK
                            al_ic.color   = GREEN
                            al_mg.value   = "✅ Water flow is stable."
                            al_mg.color   = GREEN
                            al_bx.bgcolor = "#1a2e1a"
                            al_bx.border  = ft.border.all(1, GREEN + "44")
                    else:
                        S["no_flow"] = None
                        al_ic.name    = ICO_WIFI_OFF
                        al_ic.color   = AMBER
                        al_mg.value   = f"⚠️ ESP signal lost — last seen {int(age)}s ago"
                        al_mg.color   = AMBER
                        al_bx.bgcolor = SURF2
                        al_bx.border  = ft.border.all(1, BORDER)
                else:
                    v_wt.value = v_fl.value = v_lv.value = v_tl.value = "--"
                    S["no_flow"] = None
                    al_ic.name    = ICO_WIFI_FIND
                    al_ic.color   = MUTED
                    al_mg.value   = "📡 Waiting for ESP32 to post data to Firebase…"
                    al_mg.color   = MUTED
                    al_bx.bgcolor = SURF2
                    al_bx.border  = ft.border.all(1, BORDER)

                # ── Cloud ML result ───────────────────────────────────────────
                ml       = db.get_ml_result()
                print("Chart Exists:", bool((ml or {}).get("chartB64")))
                ml_stat  = db.get_ml_status()

                # ML status banner label
                if ml_stat:
                    s  = ml_stat.get("status", "--")
                    nr = ml_stat.get("rows",    0)
                    ml_lbl.value = {
                        "ready":      f"Cloud ML: ✅ {nr:,} rows trained",
                        "training":   f"Cloud ML: 🔄 retraining {nr:,} rows…",
                        "collecting": f"Cloud ML: collecting ({nr} rows)…",
                        "error":      f"Cloud ML: ⚠️ {ml_stat.get('error','')[:40]}",
                        "waiting":    "Cloud ML: waiting for data…",
                    }.get(s, f"Cloud ML: {s}")

                if ml:
                    pf   = float(ml.get("feedKg",    0.0))
                    pw_  = float(ml.get("waterL",    0.0))
                    pd_  = str(ml.get("predDate",    "N/A"))
                    conf = float(ml.get("confidence", 0.0))
                    trend= str(ml.get("trend",       "stable"))
                    tic  = str(ml.get("trendIcon",   "📊"))
                    anom = bool(ml.get("anomaly",    False))
                    fd   = float(ml.get("feedDelta",  0.0))
                    wd   = float(ml.get("waterDelta", 0.0))
                    nr   = int(ml.get("modelRows",    0))
                    af   = ml.get("arimaFeed",        None)
                    aw_  = ml.get("arimaWater",       None)
                    ta   = str(ml.get("trainedAt",    ""))

                    # AI card in sensor section
                    ai_spin.visible = False
                    ai_main.value   = "🤖 Next Feed Prediction"
                    ai_main.color   = GREEN
                    ai_feed.value   = f"🌾  Feed:  {pf:.2f} kg"
                    ai_watr.value   = f"💧  Water: {pw_:.2f} L"
                    ai_date.value   = (f"For: {pd_}  •  {nr:,} readings  •  "
                                       f"Conf: {int(conf*100)}%")

                    tc = {"stable":GREEN,"increasing":AMBER,
                          "decreasing":BLUE,"warning":RED}.get(trend, MUTED)
                    tb = {"stable":"#1a2e1a","increasing":"#2a2210",
                          "decreasing":"#162133","warning":"#2a1a1a"}.get(trend, SURF2)
                    ai_trnd.controls = [pill(f"{tic} {trend.capitalize()}", tc, tb)]
                    if fd:
                        ai_trnd.controls.append(
                            ft.Text(f"Feed {'+' if fd>0 else ''}{fd:.1f}%",
                                    size=12, color=AMBER if fd > 5 else MUTED))
                    if wd:
                        ai_trnd.controls.append(
                            ft.Text(f"Water {'+' if wd>0 else ''}{wd:.1f}%",
                                    size=12, color=AMBER if wd > 5 else MUTED))
                    if anom:
                        ai_trnd.controls.append(pill("⚠️ Anomaly", RED, "#2a1a1a"))
                    ai_trnd.visible = True

                    # ARIMA section
                    if af is not None:
                        ar_fv.value   = f"{float(af):.2f} kg"
                        ar_wv.value   = f"{float(aw_):.2f} L"
                        ar_info.value = (f"ARIMA projection  •  {nr:,} readings  •  "
                                         f"{ta[:19]}")
                        rebuild_arima()
                        arima_sec.visible = True

                    # AI predictions section
                    ai_fv.value  = f"{pf:.2f} kg"
                    ai_wv.value  = f"{pw_:.2f} L"
                    pd_txt.value = f"Prediction Date: {pd_}"
                    pd_bx.visible = True
                    conf_col.controls = [conf_bar(conf)]
                    conf_col.visible  = True
                    ai_st.value   = (f"✅ Cloud ML ready — {nr:,} readings  •  "
                                     f"updated {ta[:19]}")
                    ai_st.color   = GREEN
                    ai_sb.bgcolor = "#1a2e1a"
                    ai_sb.border  = ft.border.all(1, GREEN + "44")
                    rebuild_pred()
                    ai_pred_sec.visible = True

                    # Farm insights
                    pat_sys   = ml.get("patSystem",  {})
                    pat_day   = ml.get("patDay",     {})
                    pat_month = ml.get("patMonth",   {})
                    ins_col.controls.clear()
                    if pat_sys:
                        ins_col.controls.append(
                            ins_block("System Behavior", "🐔", pat_sys))
                    if pat_day:
                        ins_col.controls.append(
                            ins_block("Weekly Pattern", "📅", pat_day))
                    if pat_month:
                        ins_col.controls.append(
                            ins_block("Monthly Pattern", "📆", pat_month))
                    if ins_col.controls:
                        ins_sec.visible = True

                else:
                    # No ML result yet
                    ai_spin.visible = (ml_stat or {}).get("status") == "training"
                    ai_main.value   = "Waiting for Cloud ML predictions…"
                    ai_main.color   = MUTED
                    ai_feed.value = ai_watr.value = ai_date.value = ""
                    ai_trnd.visible = False

                # Banner DB count
                db_lbl.value = f"{db.get_reading_count():,} readings"

                # 7-day forecast
                fc7_raw = db.get_forecast_7d()
                if fc7_raw:
                    try:
                        fc7_wrap.content = mk_table_list(fc7_raw)
                        fc7_wrap.visible = True
                        fc7_sec.visible = True
                    except Exception:
                        pass

                # DB table refresh every 15s
                if now - last_db > 30:
                    last_db = now
                    threading.Thread(target=live_loop, daemon=True).start()
                    threading.Thread(target=refresh_db, daemon=True).start()

                # Chart image from Cloud ML result only
                if ml:
                    chart_b64 = ml.get("chartB64", "")

                    if chart_b64 and len(chart_b64) > 100:
                        chart_img.src_base64 = chart_b64
                        chart_img.visible = True
                        chart_cap.value = "Rendered by Cloud ML"
                    else:
                        chart_img.visible = False
                        chart_cap.value = "Waiting for Cloud ML chart..."

                    chart_sec.visible = True

                su()

            except Exception:
                print(traceback.format_exc())

            time.sleep(10)

    page.on_disconnect = lambda e: S.update(running=False)

    # Initial layout build
    rebuild_sensors()
    rebuild_arima()
    rebuild_pred()

    page.add(
        banner,
        ft.Container(height=8),
        live_sec,
        title_sec,
        ft.Container(height=6),
        db_sec,
        arima_sec,
        ai_pred_sec,
        ins_sec,
        fc7_sec,
        chart_sec,
        ft.Container(height=48),
    )

    threading.Thread(target=refresh_db, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# APK mode:     flet build apk  → native ft.app(target=main), NOT web browser mode
# Browser mode: python fletapp.py --web
# Desktop mode: python fletapp.py
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # sys.argv is safe on desktop/browser but may be empty on APK
    argv = sys.argv if sys.argv else []
    web  = "--web" in argv or "--browser" in argv

    if web:
        try:
            import socket as _s
            _sk = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
            _sk.connect(("8.8.8.8", 80))
            _ip = _sk.getsockname()[0]
            _sk.close()
        except Exception:
            _ip = "127.0.0.1"
        PORT = 8550
        print(f"\n🌐 Browser mode — open on phone: http://{_ip}:{PORT}\n")
        ft.app(target=main, view=ft.AppView.WEB_BROWSER,
               port=PORT, host="0.0.0.0")
    else:
        print("📱 Running native app mode")
        ft.app(target=main)
