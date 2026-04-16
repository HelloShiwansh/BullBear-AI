"""
dashboard.py — BullBear AI Streamlit Dashboard
===============================================
Live prediction interface for all 20 NIFTY stocks.

Run:
    streamlit run dashboard.py
"""

import pickle
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from predict import (
    NIFTY_TOP20,
    SIGNAL_COLOR,
    SIGNAL_MAP,
    load_model,
    predict_all,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BullBear AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Font + palette (classic dark: ink + brass + slate) */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Playfair+Display:wght@600;700&display=swap');
        :root{
            --bg: #0B1220;
            --panel: #111C2E;
            --panel-2: #0F1A2B;
            --border: #22324A;
            --text: #E6EDF7;
            --muted: #9AA7B8;
            --muted-2: #6C7A90;
            --accent: #C9A227;      /* brass */
            --accent-2: #2AA6B8;    /* teal */
            --good: #2FBF71;
            --bad: #E35D6A;
        }

        html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
        h1, h2, h3 { font-family: "Playfair Display", Georgia, "Times New Roman", serif; }

        /* Main app background (Streamlit classnames are unstable; keep selectors broad) */
        .stApp { background: radial-gradient(1200px 700px at 10% 0%, #122340 0%, var(--bg) 60%); }

        /* Remove excessive top whitespace (but avoid header overlap) */
        div[data-testid="stAppViewContainer"] .block-container { padding-top: 3.2rem; }
        section[data-testid="stSidebar"] > div { padding-top: 1.0rem; }

        /* Sidebar polish */
        section[data-testid="stSidebar"] { background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%); border-right: 1px solid var(--border); }
        section[data-testid="stSidebar"] .stButton > button { border-radius: 10px; }
        section[data-testid="stSidebar"] h2 { margin-top: 0.2rem; margin-bottom: 0.25rem; }
        section[data-testid="stSidebar"] .stMarkdown p { margin-bottom: 0.35rem; }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.65rem; }
        section[data-testid="stSidebar"] hr { margin: 0.65rem 0; opacity: 0.35; }
        section[data-testid="stSidebar"] label { margin-bottom: 0.2rem; }
        section[data-testid="stSidebar"] * { text-align: left; }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { align-items: stretch; }

        /* Buttons */
        .stButton > button {
            border-radius: 10px;
            border: 1px solid var(--border);
            background: rgba(201, 162, 39, 0.12);
            color: var(--text);
            font-weight: 700;
        }
        .stButton > button:hover { border-color: rgba(201, 162, 39, 0.55); background: rgba(201, 162, 39, 0.18); }

        /* Tabs */
        button[role="tab"] { font-weight: 700; letter-spacing: 0.2px; }

        /* Ticker card */
        .ticker-card {
            background: rgba(17, 28, 46, 0.92);
            border-radius: 12px;
            padding: 14px 12px;
            text-align: center;
            margin: 3px 0;
            border: 1px solid var(--border);
            transition: border-color 0.2s;
        }
        .ticker-card:hover { border-color: rgba(201, 162, 39, 0.6); }
        .ticker-name  {
            font-size: 12px;
            color: var(--text);
            font-weight: 800;
            margin-bottom: 6px;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            background: rgba(42, 166, 184, 0.10);
            border: 1px solid rgba(42, 166, 184, 0.18);
        }
        .ticker-signal{ font-size: 20px; font-weight: 900; margin: 2px 0; }
        .ticker-conf  { font-size: 11px; color: var(--muted-2); }
        .ticker-price { font-size: 12px; color: var(--muted); margin-top: 2px; }

        /* Signal pill */
        .pill {
            display: inline-block;
            padding: 10px 12px;
            border-radius: 14px;
            font-size: 12px;
            font-weight: 800;
            line-height: 1;
            min-width: 78px;
            text-align: center;
            box-shadow: 0 10px 24px rgba(0,0,0,0.25);
            backdrop-filter: blur(6px);
        }
        .pill .pill-value { display:block; font-size: 16px; font-weight: 900; letter-spacing: 0.3px; }
        .pill .pill-label { display:block; margin-top: 6px; font-size: 10px; letter-spacing: 1.2px; text-transform: uppercase; opacity: 0.95; }
        .pill-buy  { background: rgba(47, 191, 113, 0.12); color: #63E6A4; border: 1px solid rgba(47, 191, 113, 0.35); }
        .pill-sell { background: rgba(227, 93, 106, 0.12); color: #FF8E98; border: 1px solid rgba(227, 93, 106, 0.35); }
        .pill-hold { background: rgba(154, 167, 184, 0.10); color: var(--muted); border: 1px solid rgba(154, 167, 184, 0.22); }

        .pill-buy  { box-shadow: 0 10px 24px rgba(47, 191, 113, 0.08), 0 10px 24px rgba(0,0,0,0.25); }
        .pill-hold { box-shadow: 0 10px 24px rgba(154, 167, 184, 0.06), 0 10px 24px rgba(0,0,0,0.25); }
        .pill-sell { box-shadow: 0 10px 24px rgba(227, 93, 106, 0.08), 0 10px 24px rgba(0,0,0,0.25); }

        .pill-row {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
            align-items: center;
        }

        /* Market status badge */
        .market-badge {
            display: inline-block;
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 700;
        }

        /* Metric card */
        .metric-card {
            background: rgba(17, 28, 46, 0.92);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            border: 1px solid var(--border);
            border-top: 3px solid var(--accent);
        }
        .metric-value { font-size: 28px; font-weight: 900; color: var(--accent); }
        .metric-label { font-size: 11px; color: var(--muted); margin-top: 2px; }

        /* Hide streamlit default footer */
        footer { visibility: hidden; }
        /* Keep header visible so sidebar toggle remains accessible */
        header { visibility: visible; }
        header[data-testid="stHeader"] {
            background: rgba(11, 18, 32, 0.35);
            backdrop-filter: blur(6px);
            border-bottom: 1px solid rgba(34, 50, 74, 0.65);
        }

        /* Section headers */
        .section-header {
            font-size: 13px;
            font-weight: 700;
            color: var(--accent);
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
IST = ZoneInfo("Asia/Kolkata")
NSE_OPEN  = time(9, 15)
NSE_CLOSE = time(15, 30)


def get_market_status() -> tuple[str, str, str]:
    """Return (status_label, description, hex_color)."""
    now = datetime.now(IST)
    weekday = now.weekday()
    t = now.time()

    if weekday >= 5:
        return "CLOSED", "Weekend — NSE closed", "#DC2626"
    if t < NSE_OPEN:
        opens = datetime.combine(now.date(), NSE_OPEN).strftime("%I:%M %p").lstrip("0")
        return "PRE-MARKET", f"NSE opens at {opens} IST", "#E8A020"
    if t <= NSE_CLOSE:
        closes = datetime.combine(now.date(), NSE_CLOSE).strftime("%I:%M %p").lstrip("0")
        return "LIVE", f"NSE open until {closes} IST", "#16A34A"
    return "CLOSED", "Market closed for today", "#DC2626"


def signal_bg(signal_name: str) -> str:
    return {"BUY": "#14532D", "SELL": "#450A0A", "HOLD": "#1E293B"}.get(signal_name, "#1E293B")


def prob_bar_html(prob_buy: float, prob_hold: float, prob_sell: float) -> str:
    """Render a horizontal probability breakdown bar."""
    b = f"{prob_buy*100:.0f}"
    h = f"{prob_hold*100:.0f}"
    s = f"{prob_sell*100:.0f}"
    return f"""
    <div style="display:flex; border-radius:4px; overflow:hidden; height:18px; font-size:10px; font-weight:700; margin-top:6px;">
        <div style="width:{prob_buy*100:.1f}%; background:#14532D; color:#4ADE80;
                    display:flex; align-items:center; justify-content:center;">{b}%</div>
        <div style="width:{prob_hold*100:.1f}%; background:#1E293B; color:#94A3B8;
                    display:flex; align-items:center; justify-content:center;">{h}%</div>
        <div style="width:{prob_sell*100:.1f}%; background:#450A0A; color:#F87171;
                    display:flex; align-items:center; justify-content:center;">{s}%</div>
    </div>
    <div style="display:flex; font-size:9px; color:#64748B; margin-top:2px;">
        <span style="flex:1; text-align:center;">BUY</span>
        <span style="flex:1; text-align:center;">HOLD</span>
        <span style="flex:1; text-align:center;">SELL</span>
    </div>
    """


def ticker_card_html(pred: dict) -> str:
    sn    = pred["signal_name"]
    color = SIGNAL_COLOR[pred["signal"]]
    conf  = f"{pred['confidence']*100:.0f}%"
    price = f"₹{pred['close']:,.1f}"

    return f"""
    <div class="ticker-card" style="border-color:{color}33;">
        <div class="ticker-name">{pred['ticker_short']}</div>
        <div class="ticker-signal" style="color:{color};">{sn}</div>
        <div class="ticker-conf">{conf} confidence</div>
        <div class="ticker-price">{price}</div>
    </div>
    """


def failed_ticker_card_html(ticker: str) -> str:
    short = ticker.replace(".NS", "")
    return f"""
    <div class="ticker-card" style="border-color:#00000066; background: rgba(0,0,0,0.35);">
        <div class="ticker-name" style="background: rgba(0,0,0,0.35); border-color: rgba(255,255,255,0.10);">{short}</div>
        <div class="ticker-signal" style="color:#E6EDF7;">FAILED</div>
        <div class="ticker-conf" style="color:#9AA7B8;">Data fetch failed</div>
        <div class="ticker-price" style="color:#6C7A90;">—</div>
    </div>
    """


# ── Data loading (cached 5 min) ───────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_predictions(history_days: int) -> tuple[list, dict]:
    return predict_all(history_days=history_days)


@st.cache_resource
def get_model_and_importances():
    model, feature_cols, scaler = load_model()
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    return model, feature_cols, scaler, importances


# ── Candlestick chart ─────────────────────────────────────────────────────────
def make_candlestick(hist: pd.DataFrame, ticker_short: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=hist["date"],
            open=hist["open"], high=hist["high"],
            low=hist["low"],   close=hist["close"],
            name="OHLC",
            increasing_line_color="#4ADE80",
            decreasing_line_color="#F87171",
            increasing_fillcolor="#166534",
            decreasing_fillcolor="#7F1D1D",
        ),
        row=1, col=1,
    )

    # ── BUY signal markers ────────────────────────────────────────────────────
    buy = hist[hist["signal"] == 2]
    if not buy.empty:
        fig.add_trace(
            go.Scatter(
                x=buy["date"],
                y=buy["low"] * 0.997,
                mode="markers",
                name="BUY",
                marker=dict(
                    symbol="triangle-up", size=12,
                    color="#16A34A",
                    line=dict(color="#FFFFFF", width=1),
                ),
                hovertemplate="BUY  %{x}<br>Conf: %{customdata:.0%}<extra></extra>",
                customdata=buy["confidence"],
            ),
            row=1, col=1,
        )

    # ── SELL signal markers ───────────────────────────────────────────────────
    sell = hist[hist["signal"] == 0]
    if not sell.empty:
        fig.add_trace(
            go.Scatter(
                x=sell["date"],
                y=sell["high"] * 1.003,
                mode="markers",
                name="SELL",
                marker=dict(
                    symbol="triangle-down", size=12,
                    color="#DC2626",
                    line=dict(color="#FFFFFF", width=1),
                ),
                hovertemplate="SELL  %{x}<br>Conf: %{customdata:.0%}<extra></extra>",
                customdata=sell["confidence"],
            ),
            row=1, col=1,
        )

    # ── Volume bars ───────────────────────────────────────────────────────────
    vol_colors = [
        "#166534" if c >= o else "#7F1D1D"
        for c, o in zip(hist["close"], hist["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=hist["date"],
            y=hist["volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.7,
            hovertemplate="%{x}<br>Vol: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=f"{ticker_short} — Price & Signals", font=dict(color="#F4F6F9", size=15)),
        template="plotly_dark",
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0D1B2A",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            font=dict(color="#94A3B8", size=11),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=480,
    )
    fig.update_yaxes(gridcolor="#1E3248", zerolinecolor="#1E3248")
    fig.update_xaxes(gridcolor="#1E3248", showgrid=False)

    return fig


# ── Feature importance chart ──────────────────────────────────────────────────
def make_feature_importance(importances: pd.Series, top_n: int = 15) -> go.Figure:
    top = importances.head(top_n).sort_values(ascending=True)
    colors = [
        "#E8A020" if i == len(top) - 1 else "#0891B2"
        for i in range(len(top))
    ]

    fig = go.Figure(
        go.Bar(
            x=top.values,
            y=top.index,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Top {top_n} Feature Importances — Random Forest",
            font=dict(color="#F4F6F9", size=14),
        ),
        template="plotly_dark",
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0D1B2A",
        xaxis=dict(title="Importance", gridcolor="#1E3248"),
        yaxis=dict(gridcolor="#1E3248"),
        margin=dict(l=0, r=10, t=50, b=0),
        height=430,
    )
    return fig


# ── Signal confidence gauge ────────────────────────────────────────────────────
def make_gauge(confidence: float, signal_name: str) -> go.Figure:
    color = SIGNAL_COLOR[{"BUY": 2, "HOLD": 1, "SELL": 0}[signal_name]]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number=dict(suffix="%", font=dict(color=color, size=32)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#64748B"),
                bar=dict(color=color),
                bgcolor="#1E3248",
                bordercolor="#2D4A6A",
                steps=[
                    dict(range=[0, 40],  color="#1E293B"),
                    dict(range=[40, 70], color="#1E3248"),
                    dict(range=[70, 100],color="#1E3A52"),
                ],
                threshold=dict(
                    line=dict(color="#E8A020", width=2), thickness=0.8, value=70
                ),
            ),
            title=dict(
                text=f"Confidence — {signal_name}",
                font=dict(color="#94A3B8", size=13),
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor="#0D1B2A",
        margin=dict(l=20, r=20, t=30, b=10),
        height=220,
    )
    return fig


# ── Signal distribution donut ─────────────────────────────────────────────────
def make_signal_donut(signals: list) -> go.Figure:
    counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
    for s in signals:
        counts[s["signal_name"]] = counts.get(s["signal_name"], 0) + 1

    fig = go.Figure(
        go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            hole=0.6,
            marker_colors=["#16A34A", "#64748B", "#DC2626"],
            textinfo="label+value",
            hoverinfo="label+percent",
            textfont=dict(color="#F4F6F9", size=12),
        )
    )
    fig.update_layout(
        title=dict(text="Signal Distribution — Today", font=dict(color="#F4F6F9", size=13)),
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0D1B2A",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        height=230,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## BullBear AI")
    st.markdown("*NIFTY Top 20 — Live Signals*")
    st.divider()

    # Market status
    status, desc, status_color = get_market_status()
    st.markdown(
        f"""
        <div style="background:{status_color}22; border:1px solid {status_color};
                    border-radius:8px; padding:10px 14px; margin-bottom:12px;">
            <div style="font-size:12px; font-weight:700; color:{status_color};">
                NSE {status}
            </div>
            <div style="font-size:11px; color:#94A3B8; margin-top:3px;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # History window
    history_days = st.selectbox(
        "History window",
        options=[30, 60, 90],
        index=0,
        format_func=lambda x: f"Last {x} trading days",
    )

    # Refresh
    st.divider()
    refresh = st.button("Refresh signals", use_container_width=True, type="primary")
    if refresh:
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        """
        <div style="font-size:10px; color:#475569; margin-top:8px;">
        Cache: 5 min · Model: Random Forest w/o sentiment · Data: yfinance
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Model quick stats
    st.markdown('<div class="section-header">Model Stats</div>', unsafe_allow_html=True)
    model_metrics = [
        ("F1 Macro",  "0.3769"),
        ("Accuracy",  "51.2%"),
        ("Backtest",  "+38.1%"),
        ("B&H bench", "+4.98%"),
    ]
    for label, val in model_metrics:
        col_l, col_r = st.columns([3, 2])
        col_l.markdown(f"<span style='font-size:12px;color:#94A3B8;'>{label}</span>", unsafe_allow_html=True)
        col_r.markdown(f"<span style='font-size:12px;font-weight:700;color:#E8A020;'>{val}</span>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════════
with st.spinner("Fetching live data and running predictions…"):
    signals, histories = get_predictions(history_days)

_, _, _, importances = get_model_and_importances()

if not signals:
    st.error("No predictions available. Check your internet connection and that the model file exists.")
    st.stop()

# Show missing tickers (if any)
expected = set(NIFTY_TOP20)
received = {s["ticker"] for s in signals}
missing = sorted(expected - received)

# Build a quick-lookup dict
signals_dict = {s["ticker"]: s for s in signals}
now_ist = datetime.now(IST)


# ════════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════════
head_l, head_r = st.columns([3, 1])
with head_l:
    st.markdown(
        f"""
        <h1 style="font-size:28px; margin-bottom:2px; color:#F4F6F9;">
            BullBear AI Dashboard
        </h1>
        <div style="font-size:13px; color:#64748B;">
            Signals as of {now_ist.strftime('%d %b %Y, %I:%M %p')} IST ·
            {len(signals)} / 20 tickers loaded
        </div>
        """,
        unsafe_allow_html=True,
    )

# Summary counts
buy_c  = sum(1 for s in signals if s["signal_name"] == "BUY")
sell_c = sum(1 for s in signals if s["signal_name"] == "SELL")
hold_c = sum(1 for s in signals if s["signal_name"] == "HOLD")
avg_conf = np.mean([s["confidence"] for s in signals]) * 100

with head_r:
    st.markdown(
        f"""
        <div class="pill-row" style="height:100%;">
            <span class="pill pill-buy">
                <span class="pill-value">{buy_c}</span>
                <span class="pill-label">Buy</span>
            </span>
            <span class="pill pill-hold">
                <span class="pill-value">{hold_c}</span>
                <span class="pill-label">Hold</span>
            </span>
            <span class="pill pill-sell">
                <span class="pill-value">{sell_c}</span>
                <span class="pill-label">Sell</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
tab_overview, tab_detail, tab_model = st.tabs(
    ["Overview — All 20 Tickers", "Ticker Detail", "Model Performance"]
)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
with tab_overview:
    # Sort: BUY first → SELL → HOLD, then by confidence desc within each group
    order = {"BUY": 0, "SELL": 1, "HOLD": 2}
    sorted_signals = sorted(
        signals,
        key=lambda s: (order[s["signal_name"]], -s["confidence"]),
    )

    # 4-column grid — 5 rows for 20 tickers (fill any missing with FAILED cards)
    grid_items = list(sorted_signals)
    for t in missing:
        grid_items.append({"_failed": t})

    for row_idx in range(5):
        cols = st.columns(4, gap="small")
        for col_idx in range(4):
            ticker_idx = row_idx * 4 + col_idx
            if ticker_idx >= len(grid_items):
                break
            item = grid_items[ticker_idx]
            with cols[col_idx]:
                if isinstance(item, dict) and item.get("_failed"):
                    st.markdown(failed_ticker_card_html(item["_failed"]), unsafe_allow_html=True)
                else:
                    pred = item
                    st.markdown(ticker_card_html(pred), unsafe_allow_html=True)
                    st.markdown(
                        prob_bar_html(pred["prob_buy"], pred["prob_hold"], pred["prob_sell"]),
                        unsafe_allow_html=True,
                    )

    st.divider()

    # Summary row: donut + metrics
    col_donut, col_m1, col_m2, col_m3, col_m4 = st.columns([2, 1, 1, 1, 1])
    with col_donut:
        st.plotly_chart(make_signal_donut(signals), use_container_width=True, config={"displayModeBar": False})

    for col, (metric_val, metric_label, color) in zip(
        [col_m1, col_m2, col_m3, col_m4],
        [
            (f"{buy_c}", "BUY signals", "#16A34A"),
            (f"{sell_c}", "SELL signals", "#DC2626"),
            (f"{hold_c}", "HOLD signals", "#64748B"),
            (f"{avg_conf:.0f}%", "Avg confidence", "#E8A020"),
        ],
    ):
        with col:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top-color:{color};">
                    <div class="metric-value" style="color:{color};">{metric_val}</div>
                    <div class="metric-label">{metric_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — TICKER DETAIL
# ──────────────────────────────────────────────────────────────────────────────
with tab_detail:
    # Ticker selector
    available_tickers = [s["ticker"] for s in signals] + missing
    selected_ticker = st.selectbox(
        "Select ticker",
        options=available_tickers,
        format_func=lambda t: t.replace(".NS", ""),
        label_visibility="collapsed",
    )

    if selected_ticker in missing:
        st.markdown(
            f"""
            <div style="background:rgba(0,0,0,0.35); border:1px solid rgba(255,255,255,0.10);
                        border-radius:12px; padding:16px 18px; color:#E6EDF7;">
                <div style="font-size:12px; letter-spacing:1.2px; text-transform:uppercase; color:#9AA7B8;">
                    {selected_ticker.replace(".NS","")} · NSE
                </div>
                <div style="font-size:22px; font-weight:900; margin-top:6px;">
                    Data fetch failed for this ticker
                </div>
                <div style="font-size:12px; color:#9AA7B8; margin-top:6px;">
                    Try refresh signals, or it may resolve automatically in a few minutes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    pred  = signals_dict[selected_ticker]
    hist  = histories.get(selected_ticker, pd.DataFrame())
    short = pred["ticker_short"]

    # ── Signal headline ───────────────────────────────────────────────────────
    sig_color = SIGNAL_COLOR[pred["signal"]]
    head_signal, head_gauge = st.columns([2, 1])

    with head_signal:
        st.markdown(
            f"""
            <div style="background:#1E3248; border-radius:12px; padding:20px 24px;
                        border-left:5px solid {sig_color};">
                <div style="font-size:13px; color:#94A3B8; letter-spacing:1px;">{short} · NSE</div>
                <div style="font-size:42px; font-weight:900; color:{sig_color}; margin:6px 0;">
                    {pred['signal_name']}
                </div>
                <div style="display:flex; gap:24px; margin-top:6px;">
                    <div>
                        <div style="font-size:11px; color:#64748B;">CLOSE</div>
                        <div style="font-size:20px; font-weight:700; color:#F4F6F9;">₹{pred['close']:,.2f}</div>
                    </div>
                    <div>
                        <div style="font-size:11px; color:#64748B;">DATE</div>
                        <div style="font-size:15px; color:#94A3B8;">{pred['date'].strftime('%d %b %Y')}</div>
                    </div>
                </div>
                <div style="margin-top:12px;">
                    {prob_bar_html(pred['prob_buy'], pred['prob_hold'], pred['prob_sell'])}
                
            """,
            unsafe_allow_html=True,
        )

    with head_gauge:
        st.plotly_chart(
            make_gauge(pred["confidence"], pred["signal_name"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Candlestick chart ─────────────────────────────────────────────────────
    if not hist.empty:
        st.plotly_chart(
            make_candlestick(hist, short),
            use_container_width=True,
            config={"displayModeBar": True, "displaylogo": False},
        )
    else:
        st.warning("No historical data available for this ticker.")

    st.divider()

    # ── Feature importance + signal history table ─────────────────────────────
    col_fi, col_hist = st.columns([3, 2])

    with col_fi:
        st.plotly_chart(
            make_feature_importance(importances, top_n=15),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_hist:
        st.markdown('<div class="section-header">Signal History</div>', unsafe_allow_html=True)
        if not hist.empty:
            # Evaluation is strictly post-hoc for display:
            # predicted signal at time t is compared against next-day return (t+1 close).
            eval_hist = hist[["date", "close", "signal_name", "confidence"]].copy()
            eval_hist = eval_hist.sort_values("date", ascending=True).reset_index(drop=True)
            eval_hist["next_return"] = eval_hist["close"].shift(-1) / eval_hist["close"] - 1.0

            def _actual_label(next_ret: float) -> str:
                if next_ret > 0.01:
                    return "BUY"
                if next_ret < -0.01:
                    return "SELL"
                return "HOLD"

            eval_hist["actual"] = eval_hist["next_return"].apply(_actual_label)
            eval_hist["is_correct"] = np.where(
                eval_hist["signal_name"] == eval_hist["actual"], "Correct", "Incorrect"
            )

            # Skip last row because next-day close is unavailable.
            eval_hist = eval_hist.dropna(subset=["next_return"])

            display_hist = eval_hist[["date", "close", "signal_name", "actual", "is_correct", "confidence"]].copy()
            display_hist = display_hist.sort_values("date", ascending=False).reset_index(drop=True)
            display_hist.columns = ["Date", "Close (₹)", "Signal", "Actual", "Correct/Incorrect", "Confidence"]
            display_hist["Date"]        = pd.to_datetime(display_hist["Date"]).dt.strftime("%d %b")
            display_hist["Close (₹)"]   = display_hist["Close (₹)"].map(lambda x: f"₹{x:,.1f}")
            display_hist["Confidence"]  = display_hist["Confidence"].map(lambda x: f"{x*100:.0f}%")

            def colour_signal(val):
                if val == "BUY":
                    return "color: #4ADE80; font-weight: bold"
                elif val == "SELL":
                    return "color: #F87171; font-weight: bold"
                return "color: #94A3B8"

            def colour_eval(val):
                if val == "Correct":
                    return "color: #4ADE80; font-weight: bold"
                return "color: #F87171; font-weight: bold"

            styled = (
                display_hist.style
                .applymap(colour_signal, subset=["Signal", "Actual"])
                .applymap(colour_eval, subset=["Correct/Incorrect"])
            )
            st.dataframe(styled, use_container_width=True, height=390, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
with tab_model:
    st.markdown('<div class="section-header">Production Model — Random Forest Without Sentiment</div>', unsafe_allow_html=True)

    # Key metric callouts
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, (v, l, c) in zip(
        [m1, m2, m3, m4, m5],
        [
            ("0.3769",  "F1 Macro",       "#E8A020"),
            ("51.2%",   "Test Accuracy",  "#0891B2"),
            ("+38.1%",  "Backtest Return","#16A34A"),
            ("+4.98%",  "Buy-and-Hold",   "#64748B"),
            ("7.6×",    "Outperformance", "#E8A020"),
        ],
    ):
        col.markdown(
            f"""<div class="metric-card" style="border-top-color:{c};">
                <div class="metric-value" style="color:{c};">{v}</div>
                <div class="metric-label">{l}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Full comparison table
    st.markdown('<div class="section-header">All 8 Models — Test Set (2024–2026)</div>', unsafe_allow_html=True)

    comparison = pd.DataFrame([
        {"Model": "Random Forest", "Condition": "w/o sentiment", "F1 Macro": 0.3769, "F1 BUY": 0.220, "F1 SELL": 0.239, "F1 HOLD": 0.672, "Accuracy": "51.2%", "Backtest": "+38.1%"},
        {"Model": "Random Forest", "Condition": "with sentiment", "F1 Macro": 0.3743, "F1 BUY": 0.212, "F1 SELL": 0.236, "F1 HOLD": 0.675, "Accuracy": "51.3%", "Backtest": "+32.1%"},
        {"Model": "XGBoost",       "Condition": "w/o sentiment", "F1 Macro": 0.3745, "F1 BUY": 0.238, "F1 SELL": 0.250, "F1 HOLD": 0.636, "Accuracy": "47.9%", "Backtest": "+21.6%"},
        {"Model": "XGBoost",       "Condition": "with sentiment","F1 Macro": 0.3692, "F1 BUY": 0.231, "F1 SELL": 0.235, "F1 HOLD": 0.642, "Accuracy": "48.2%", "Backtest": "−24.8%"},
        {"Model": "LightGBM",      "Condition": "w/o sentiment", "F1 Macro": 0.3679, "F1 BUY": 0.229, "F1 SELL": 0.243, "F1 HOLD": 0.631, "Accuracy": "47.2%", "Backtest": "−2.5%"},
        {"Model": "LightGBM",      "Condition": "with sentiment","F1 Macro": 0.3676, "F1 BUY": 0.222, "F1 SELL": 0.244, "F1 HOLD": 0.636, "Accuracy": "47.6%", "Backtest": "−10.3%"},
        {"Model": "LSTM",          "Condition": "with sentiment","F1 Macro": 0.3294, "F1 BUY": 0.152, "F1 SELL": 0.094, "F1 HOLD": 0.742, "Accuracy": "58.9%", "Backtest": "+69.5%"},
        {"Model": "LSTM",          "Condition": "w/o sentiment", "F1 Macro": 0.3197, "F1 BUY": 0.200, "F1 SELL": 0.046, "F1 HOLD": 0.713, "Accuracy": "55.0%", "Backtest": "−40.7%"},
    ])

    def colour_row(row):
        if row.name == 0:
            return ["background-color: #162A40; font-weight: bold; color: #E8A020"] * len(row)
        return [""] * len(row)

    def colour_backtest(val):
        if isinstance(val, str):
            if val.startswith("+"):  return "color: #4ADE80; font-weight: bold"
            if val.startswith("−"):  return "color: #F87171"
        return ""

    styled_comp = (
        comparison.style
        .apply(colour_row, axis=1)
        .applymap(colour_backtest, subset=["Backtest"])
        .format({"F1 Macro": "{:.4f}", "F1 BUY": "{:.3f}", "F1 SELL": "{:.3f}", "F1 HOLD": "{:.3f}"})
    )
    st.dataframe(styled_comp, use_container_width=True, hide_index=True, height=330)

    st.markdown(
        """
        <div style="background:#1E293B; border-radius:8px; padding:12px 16px; margin-top:12px;
                    font-size:12px; color:#64748B; border-left:3px solid #E8A020;">
        <b style="color:#E8A020;">Buy-and-Hold benchmark: +4.98%</b>
        &nbsp;·&nbsp; Equal-weight 20-stock portfolio &nbsp;·&nbsp; Test period: 2024–2026
        &nbsp;·&nbsp; Random Forest strategy achieves <b style="color:#4ADE80;">7.6× outperformance</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Four findings
    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    f3, f4 = st.columns(2)

    findings = [
        ("Classical ML Beats LSTM",
         "RF outperforms LSTM by ~0.05 F1 macro. Technical indicators already encode sequential patterns LSTM would learn. Well-documented in financial ML literature.",
         "#0891B2"),
        ("Sentiment at 10.7% Coverage Adds No Value",
         "3 of 4 model families perform better without sentiment. Root cause: only 2017–2021 news exists — 89.3% of rows are zero-filled, adding noise not signal.",
         "#8B5CF6"),
        ("LSTM +69.5% Backtest Paradox",
         "LSTM with sentiment predicts HOLD 88% of the time. Selective models inflate backtest returns in rising markets despite poor F1. F1 macro is the reliable metric.",
         "#E8A020"),
        ("RF Is the Right Production Choice",
         "+38.1% vs +4.98% buy-and-hold. Highest F1 macro. Selective 16.7% BUY signal rate = high-confidence trades, not noise. No sentiment = works today.",
         "#16A34A"),
    ]

    for (col, (title, body, color)) in zip([f1, f2, f3, f4], findings):
        with col:
            st.markdown(
                f"""
                <div style="background:#1E3248; border-radius:10px; padding:16px;
                            border-left:4px solid {color}; height:130px;">
                    <div style="font-size:12px; font-weight:700; color:{color}; margin-bottom:6px;">{title}</div>
                    <div style="font-size:11px; color:#94A3B8; line-height:1.5;">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
