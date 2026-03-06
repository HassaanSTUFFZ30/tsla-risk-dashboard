# ============================================================
#   TSLA — QUANTITATIVE MARKET RISK & TECHNICAL ANALYSIS
#   DASHBOARD  —  Streamlit Version
#   Author: Hassaan Ali
# ============================================================
#
#  HOW TO RUN LOCALLY:
#  1. pip install streamlit yfinance pandas numpy plotly scipy
#  2. streamlit run tsla_streamlit_dashboard.py
#
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =============================================================
#  PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title  = "QUANT//RISK — TSLA Dashboard",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "collapsed"
)

# =============================================================
#  NEON COLORS
# =============================================================
NEON_CYAN   = "#00f5ff"
NEON_PINK   = "#ff00e5"
NEON_GREEN  = "#00ff88"
NEON_YELLOW = "#ffe600"
NEON_RED    = "#ff2d55"
NEON_ORANGE = "#ff9500"
NEON_WHITE  = "#ffffff"
BG_DARK     = "#020610"
BG_PANEL    = "#06101f"
BG_CARD     = "#080f1e"
GRID_COLOR  = "rgba(0,245,255,0.08)"
TEXT_COLOR  = "#e8f4ff"
MUTED_COLOR = "#5a8aaa"

BASE = dict(
    paper_bgcolor = BG_PANEL,
    plot_bgcolor  = BG_PANEL,
    font          = dict(family="Consolas, monospace", color=TEXT_COLOR, size=13),
    xaxis         = dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                         color=TEXT_COLOR, tickfont=dict(size=12, color=TEXT_COLOR)),
    yaxis         = dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                         color=TEXT_COLOR, tickfont=dict(size=12, color=TEXT_COLOR)),
    margin        = dict(l=60, r=30, t=50, b=50),
    legend        = dict(bgcolor="rgba(2,6,16,0.9)", bordercolor=NEON_CYAN,
                         borderwidth=1, font=dict(color=TEXT_COLOR, size=12)),
)

# =============================================================
#  CUSTOM CSS — dark neon theme
# =============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #020610; }
    .block-container { padding: 2rem 3rem; }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #080f1e;
        border: 1px solid rgba(0,245,255,0.2);
        border-top: 3px solid #00f5ff;
        border-radius: 4px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label {
        color: #5a8aaa !important;
        font-family: Consolas, monospace !important;
        font-size: 11px !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00f5ff !important;
        font-family: Consolas, monospace !important;
        font-size: 22px !important;
        font-weight: bold !important;
    }

    /* Divider */
    hr { border-color: rgba(0,245,255,0.15); }

    /* Checkbox */
    .stCheckbox label { color: #e8f4ff !important; font-family: Consolas, monospace !important; }

    /* Dataframe */
    .dataframe { background: #080f1e !important; color: #e8f4ff !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #06101f; }

    /* Section headers */
    h1, h2, h3 { font-family: Consolas, monospace !important; color: #00f5ff !important; }
    p { color: #e8f4ff; font-family: Consolas, monospace; }
</style>
""", unsafe_allow_html=True)

# =============================================================
#  DATA FUNCTIONS
# =============================================================

@st.cache_data
def fetch_data(ticker="TSLA", start="2019-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.dropna(inplace=True)
    return df

def calculate_returns(df):
    df = df.copy()
    df["Simple_Return"] = df["Close"].pct_change()
    df["Log_Return"]    = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df

def calculate_indicators(df):
    df = df.copy()
    df["SMA_20"]  = df["Close"].rolling(20).mean()
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["EMA_12"]  = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"]  = df["Close"].ewm(span=26, adjust=False).mean()
    bb_std         = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA_20"] + 2 * bb_std
    df["BB_Lower"] = df["SMA_20"] - 2 * bb_std
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    df["Vol_20d"] = df["Log_Return"].rolling(20).std() * np.sqrt(252)
    df["Vol_60d"] = df["Log_Return"].rolling(60).std() * np.sqrt(252)
    return df

def calculate_var_cvar(returns, confidence_levels=[0.90, 0.95, 0.99]):
    results = {}
    # Force to flat numpy array — fixes yfinance MultiIndex issues
    r = np.array(returns.squeeze().dropna().values, dtype=float)
    r = r[~np.isnan(r)]  # extra safety: remove any remaining NaNs
    for cl in confidence_levels:
        alpha = 1 - cl
        label = f"{int(cl*100)}%"
        mu, sigma = float(np.mean(r)), float(np.std(r))
        z = stats.norm.ppf(alpha)
        hist_var   = float(np.percentile(r, alpha * 100))
        hist_cvar  = float(np.mean(r[r <= hist_var]))
        para_var   = mu + z * sigma
        para_cvar  = mu - sigma * (stats.norm.pdf(z) / alpha)
        np.random.seed(42)
        sim        = np.random.normal(mu, sigma, 10000)
        mc_var     = float(np.percentile(sim, alpha * 100))
        mc_cvar    = float(np.mean(sim[sim <= mc_var]))
        results[label] = {
            "Historical VaR":   hist_var,  "Historical CVaR":  hist_cvar,
            "Parametric VaR":   para_var,  "Parametric CVaR":  para_cvar,
            "Monte Carlo VaR":  mc_var,    "Monte Carlo CVaR": mc_cvar,
        }
    return results

def calculate_risk_stats(df):
    r          = df["Log_Return"].dropna()
    annual_ret = r.mean() * 252
    annual_vol = r.std()  * np.sqrt(252)
    rf         = 0.05
    sharpe     = (annual_ret - rf) / annual_vol
    down_vol   = r[r < 0].std() * np.sqrt(252)
    sortino    = (annual_ret - rf) / down_vol
    cum        = (1 + r).cumprod()
    roll_max   = cum.cummax()
    drawdown   = (cum - roll_max) / roll_max
    max_dd     = drawdown.min()
    calmar     = annual_ret / abs(max_dd) if max_dd != 0 else 0
    return {
        "Annual Return": annual_ret, "Annual Volatility": annual_vol,
        "Sharpe Ratio":  sharpe,     "Sortino Ratio":     sortino,
        "Max Drawdown":  max_dd,     "Calmar Ratio":      calmar,
        "Skewness":      r.skew(),   "Kurtosis":          r.kurt(),
        "Win Rate":      (r > 0).sum() / len(r),
        "Drawdown Series": drawdown, "Cumulative": cum,
    }

# =============================================================
#  CHART FUNCTIONS
# =============================================================

def price_chart(df, show_bb=True, show_sma=True):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing=dict(line=dict(color=NEON_GREEN, width=1), fillcolor=NEON_GREEN),
        decreasing=dict(line=dict(color=NEON_RED,   width=1), fillcolor=NEON_RED),
        name="TSLA"), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"],
            line=dict(color=NEON_CYAN, width=2), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"],
            line=dict(color=NEON_CYAN, width=2),
            fill="tonexty", fillcolor="rgba(0,245,255,0.07)",
            name="BB Lower"), row=1, col=1)
    if show_sma:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"],
            line=dict(color=NEON_YELLOW, width=2.5), name="SMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"],
            line=dict(color=NEON_ORANGE, width=2.5), name="SMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"],
            line=dict(color=NEON_PINK,   width=2.5), name="SMA 200"), row=1, col=1)
    vol_colors = [NEON_GREEN if c >= o else NEON_RED for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
        marker_color=vol_colors, opacity=0.75, name="Volume"), row=2, col=1)
    layout = dict(**BASE)
    layout["yaxis2"] = dict(gridcolor=GRID_COLOR, color=TEXT_COLOR,
                             tickfont=dict(color=TEXT_COLOR, size=11))
    fig.update_layout(**layout,
        title=dict(text="TSLA — Price Chart", font=dict(color=NEON_CYAN, size=17)),
        xaxis_rangeslider_visible=False, height=540)
    return fig

def returns_dist(df):
    r = df["Log_Return"].dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=80,
        marker=dict(color=NEON_CYAN, line=dict(color="rgba(0,245,255,0.3)", width=0.5)),
        opacity=0.85, name="Daily Returns"))
    x_r = np.linspace(r.min(), r.max(), 300)
    ny  = stats.norm.pdf(x_r, r.mean(), r.std())
    ny  = ny * len(r) * (r.max() - r.min()) / 80
    fig.add_trace(go.Scatter(x=x_r, y=ny,
        line=dict(color=NEON_YELLOW, width=3), name="Normal Fit"))
    v95  = np.percentile(r, 5)
    cv95 = r[r <= v95].mean()
    fig.add_vline(x=v95,  line=dict(color=NEON_ORANGE, width=3, dash="dash"),
        annotation=dict(text=f"VaR 95%: {v95:.2f}%",
                        font=dict(color=NEON_ORANGE, size=13), bgcolor=BG_CARD))
    fig.add_vline(x=cv95, line=dict(color=NEON_RED, width=3, dash="dash"),
        annotation=dict(text=f"CVaR 95%: {cv95:.2f}%",
                        font=dict(color=NEON_RED, size=13), bgcolor=BG_CARD))
    fig.update_layout(**BASE,
        title=dict(text="Return Distribution + VaR/CVaR", font=dict(color=NEON_CYAN, size=16)),
        height=400, xaxis_title="Daily Log Return (%)", yaxis_title="Frequency")
    return fig

def var_comparison(var_data):
    cls     = ["90%", "95%", "99%"]
    methods = ["Historical", "Parametric", "Monte Carlo"]
    colors  = [NEON_CYAN, NEON_PINK, NEON_GREEN]
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Value at Risk (VaR)", "CVaR — Expected Shortfall"])
    fig.update_annotations(font=dict(color=NEON_WHITE, size=14))
    for i, m in enumerate(methods):
        vv = [var_data[c][f"{m} VaR"]  * 100 for c in cls]
        cv = [var_data[c][f"{m} CVaR"] * 100 for c in cls]
        fig.add_trace(go.Bar(name=m, x=cls, y=vv,
            marker=dict(color=colors[i], line=dict(color=NEON_WHITE, width=0.5)),
            opacity=0.9, showlegend=True), row=1, col=1)
        fig.add_trace(go.Bar(name=m, x=cls, y=cv,
            marker=dict(color=colors[i], line=dict(color=NEON_WHITE, width=0.5)),
            opacity=0.9, showlegend=False), row=1, col=2)
    layout = dict(**BASE)
    layout["yaxis2"] = dict(gridcolor=GRID_COLOR, color=TEXT_COLOR,
                             tickfont=dict(color=TEXT_COLOR, size=12))
    fig.update_layout(**layout,
        title=dict(text="VaR & CVaR — Method Comparison", font=dict(color=NEON_CYAN, size=16)),
        barmode="group", height=400)
    return fig

def drawdown_chart(st_data):
    dd = st_data["Drawdown Series"] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values,
        fill="tozeroy", fillcolor="rgba(255,45,85,0.25)",
        line=dict(color=NEON_RED, width=2.5), name="Drawdown"))
    dd_date = dd.idxmin()
    fig.add_annotation(x=dd_date, y=dd.min(),
        text=f"Max: {dd.min():.1f}%", showarrow=True, arrowhead=2,
        arrowcolor=NEON_YELLOW, arrowwidth=2,
        font=dict(color=NEON_YELLOW, size=13), bgcolor=BG_CARD)
    fig.update_layout(**BASE,
        title=dict(text="Drawdown from Peak", font=dict(color=NEON_CYAN, size=16)),
        height=330, yaxis_title="Drawdown (%)")
    return fig

def rsi_chart(df):
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,45,85,0.18)",  line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,255,136,0.18)", line_width=0)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
        line=dict(color=NEON_WHITE, width=3), name="RSI (14)"))
    fig.add_hline(y=70, line=dict(color=NEON_RED,    width=2, dash="dot"))
    fig.add_hline(y=30, line=dict(color=NEON_GREEN,  width=2, dash="dot"))
    fig.add_hline(y=50, line=dict(color=MUTED_COLOR, width=1, dash="dot"))
    fig.add_annotation(x=df.index[30], y=75, text="OVERBOUGHT  70",
        font=dict(color=NEON_RED,   size=12), showarrow=False, bgcolor=BG_CARD)
    fig.add_annotation(x=df.index[30], y=25, text="OVERSOLD  30",
        font=dict(color=NEON_GREEN, size=12), showarrow=False, bgcolor=BG_CARD)
    layout = dict(**BASE)
    layout["yaxis"] = dict(gridcolor=GRID_COLOR, color=TEXT_COLOR,
                            tickfont=dict(color=TEXT_COLOR, size=12),
                            range=[0, 100], title="RSI",
                            title_font=dict(color=TEXT_COLOR))
    fig.update_layout(**layout,
        title=dict(text="RSI — Relative Strength Index (14)", font=dict(color=NEON_CYAN, size=16)),
        height=330)
    return fig

def macd_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
        line=dict(color=NEON_CYAN, width=2.5), name="MACD"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"],
        line=dict(color=NEON_PINK, width=2.5), name="Signal"), row=1, col=1)
    hcolors = [NEON_GREEN if v >= 0 else NEON_RED for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"],
        marker_color=hcolors, opacity=0.85, name="Histogram"), row=2, col=1)
    layout = dict(**BASE)
    layout["yaxis2"] = dict(gridcolor=GRID_COLOR, color=TEXT_COLOR,
                             tickfont=dict(color=TEXT_COLOR, size=12))
    fig.update_layout(**layout,
        title=dict(text="MACD — Trend Momentum", font=dict(color=NEON_CYAN, size=16)),
        height=390)
    return fig

def volatility_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol_20d"] * 100,
        line=dict(color=NEON_CYAN, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,245,255,0.1)", name="20-day Vol"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol_60d"] * 100,
        line=dict(color=NEON_PINK, width=3), name="60-day Vol"))
    fig.update_layout(**BASE,
        title=dict(text="Rolling Annualized Volatility (%)", font=dict(color=NEON_CYAN, size=16)),
        height=330, yaxis_title="Volatility (%)")
    return fig

def cumulative_chart(st_data):
    cum = st_data["Cumulative"] * 100 - 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=cum.values,
        line=dict(color=NEON_GREEN, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,255,136,0.1)",
        name="Cumulative Return"))
    fig.add_hline(y=0, line=dict(color=MUTED_COLOR, width=1, dash="dot"))
    fig.update_layout(**BASE,
        title=dict(text="Cumulative Return Since 2019", font=dict(color=NEON_CYAN, size=16)),
        height=330, yaxis_title="Return (%)")
    return fig

# =============================================================
#  LOAD DATA
# =============================================================
with st.spinner("⏳ Fetching TSLA data from Yahoo Finance..."):
    raw_df      = fetch_data()
    ret_df      = calculate_returns(raw_df)
    full_df     = calculate_indicators(ret_df)
    # Force flat numpy-compatible Series
    log_returns = full_df["Log_Return"].squeeze().dropna().reset_index(drop=True)
    var_data    = calculate_var_cvar(log_returns)
    st_data     = calculate_risk_stats(full_df)

# =============================================================
#  HEADER
# =============================================================
col_title, col_author = st.columns([3, 1])
with col_title:
    st.markdown(f"""
    <div style='font-family: Consolas, monospace;'>
        <span style='color:{NEON_CYAN}; font-size:36px; font-weight:900;
                     letter-spacing:8px; text-shadow: 0 0 20px {NEON_CYAN};'>QUANT</span>
        <span style='color:{NEON_PINK}; font-size:36px;
                     text-shadow: 0 0 20px {NEON_PINK};'> // </span>
        <span style='color:#ffffff; font-size:36px; font-weight:900; letter-spacing:8px;'>RISK</span>
    </div>
    <p style='color:{MUTED_COLOR}; font-size:12px; letter-spacing:3px;
              font-family:Consolas,monospace; margin-top:4px;'>
        TSLA  —  MARKET RISK & TECHNICAL ANALYSIS TERMINAL  |  2019 – 2024
    </p>
    """, unsafe_allow_html=True)

with col_author:
    st.markdown(f"""
    <div style='text-align:right; padding-top:10px;'>
        <p style='color:{MUTED_COLOR}; font-size:10px; letter-spacing:1px;
                  font-family:Consolas,monospace; margin:0;'>
            Quantitative Market Risk &<br>Technical Analysis Dashboard
        </p>
        <p style='color:{NEON_CYAN}; font-size:11px; letter-spacing:2px;
                  font-family:Consolas,monospace; margin:4px 0 0 0; opacity:0.8;'>
            Hassaan Ali
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================
#  KPI METRICS
# =============================================================
latest  = float(full_df["Close"].iloc[-1])
v95     = var_data["95%"]["Historical VaR"]  * 100
cv95    = var_data["95%"]["Historical CVaR"] * 100
rsi_now = float(full_df["RSI"].iloc[-1])

k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
k1.metric("TSLA Price",    f"${latest:.2f}")
k2.metric("VaR 95%",       f"{v95:.2f}%")
k3.metric("CVaR 95%",      f"{cv95:.2f}%")
k4.metric("Annual Return", f"{st_data['Annual Return']*100:.1f}%")
k5.metric("Annual Vol",    f"{st_data['Annual Volatility']*100:.1f}%")
k6.metric("Sharpe Ratio",  f"{st_data['Sharpe Ratio']:.2f}")
k7.metric("Max Drawdown",  f"{st_data['Max Drawdown']*100:.1f}%")
k8.metric("RSI (14)",      f"{rsi_now:.1f}")

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================
#  CONTROLS
# =============================================================
c1, c2 = st.columns(2)
with c1:
    show_bb  = st.checkbox("Show Bollinger Bands",  value=True)
with c2:
    show_sma = st.checkbox("Show Moving Averages",  value=True)

# =============================================================
#  CHARTS
# =============================================================

# Price Chart
st.plotly_chart(price_chart(full_df, show_bb, show_sma),
                use_container_width=True)

# Returns + VaR Comparison
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(returns_dist(full_df), use_container_width=True)
with col2:
    st.plotly_chart(var_comparison(var_data), use_container_width=True)

# Cumulative + Drawdown
col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(cumulative_chart(st_data), use_container_width=True)
with col4:
    st.plotly_chart(drawdown_chart(st_data), use_container_width=True)

# Volatility + RSI
col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(volatility_chart(full_df), use_container_width=True)
with col6:
    st.plotly_chart(rsi_chart(full_df), use_container_width=True)

# MACD
st.plotly_chart(macd_chart(full_df), use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================
#  VaR TABLE
# =============================================================
st.markdown(f"<h3 style='color:{NEON_CYAN}; font-family:Consolas,monospace; "
            f"letter-spacing:3px; font-size:14px;'>VaR / CVaR — FULL RESULTS TABLE</h3>",
            unsafe_allow_html=True)

table_rows = []
for cl in ["90%", "95%", "99%"]:
    for method in ["Historical", "Parametric", "Monte Carlo"]:
        table_rows.append({
            "Confidence":  cl,
            "Method":      method,
            "VaR":         f"{var_data[cl][f'{method} VaR']*100:.3f}%",
            "CVaR (Expected Shortfall)": f"{var_data[cl][f'{method} CVaR']*100:.3f}%",
        })

st.dataframe(
    pd.DataFrame(table_rows),
    use_container_width=True,
    hide_index=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================
#  RISK STATS
# =============================================================
st.markdown(f"<h3 style='color:{NEON_CYAN}; font-family:Consolas,monospace; "
            f"letter-spacing:3px; font-size:14px;'>PORTFOLIO RISK STATISTICS</h3>",
            unsafe_allow_html=True)

s1,s2,s3,s4,s5,s6,s7,s8,s9 = st.columns(9)
s1.metric("Annual Return",   f"{st_data['Annual Return']*100:.2f}%")
s2.metric("Annual Vol",      f"{st_data['Annual Volatility']*100:.2f}%")
s3.metric("Sharpe Ratio",    f"{st_data['Sharpe Ratio']:.3f}")
s4.metric("Sortino Ratio",   f"{st_data['Sortino Ratio']:.3f}")
s5.metric("Max Drawdown",    f"{st_data['Max Drawdown']*100:.2f}%")
s6.metric("Calmar Ratio",    f"{st_data['Calmar Ratio']:.3f}")
s7.metric("Skewness",        f"{st_data['Skewness']:.3f}")
s8.metric("Kurtosis",        f"{st_data['Kurtosis']:.3f}")
s9.metric("Win Rate",        f"{st_data['Win Rate']*100:.1f}%")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<p style='color:#1a3a4a; font-size:11px; text-align:center;
          font-family:Consolas,monospace; letter-spacing:2px;'>
    QUANT//RISK  ·  Data: Yahoo Finance via yfinance  ·  TSLA 2019–2024  ·  Hassaan Ali
</p>
""", unsafe_allow_html=True)
