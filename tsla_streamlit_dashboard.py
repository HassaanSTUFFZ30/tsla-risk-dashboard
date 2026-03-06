# ============================================================
#   TSLA — QUANTITATIVE MARKET RISK & TECHNICAL ANALYSIS
#   DASHBOARD  —  Streamlit Version
#   Author: Hassaan Ali
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="QUANT//RISK — TSLA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── COLORS ───────────────────────────────────────────────────
NC = "#00f5ff"   # cyan
NP = "#ff00e5"   # pink
NG = "#00ff88"   # green
NY = "#ffe600"   # yellow
NR = "#ff2d55"   # red
NO = "#ff9500"   # orange
NW = "#ffffff"   # white
BD = "#020610"   # bg dark
BP = "#06101f"   # bg panel
BC = "#080f1e"   # bg card
GR = "rgba(0,245,255,0.08)"
TC = "#e8f4ff"   # text
MC = "#5a8aaa"   # muted

BASE = dict(
    paper_bgcolor=BP, plot_bgcolor=BP,
    font=dict(family="Consolas, monospace", color=TC, size=13),
    xaxis=dict(gridcolor=GR, zerolinecolor=GR, color=TC, tickfont=dict(size=12, color=TC)),
    yaxis=dict(gridcolor=GR, zerolinecolor=GR, color=TC, tickfont=dict(size=12, color=TC)),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(bgcolor="rgba(2,6,16,0.9)", bordercolor=NC, borderwidth=1,
                font=dict(color=TC, size=12)),
)

st.markdown("""
<style>
.stApp { background-color: #020610; }
.block-container { padding: 2rem 3rem; }
#MainMenu, footer, header { visibility: hidden; }
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
hr { border-color: rgba(0,245,255,0.15); }
h1,h2,h3 { font-family: Consolas, monospace !important; color: #00f5ff !important; }
</style>
""", unsafe_allow_html=True)

# ── PURE PYTHON PERCENTILE ───────────────────────────────────
def py_percentile(data_list, pct):
    """100% pure Python — no numpy — works on any version."""
    clean = sorted([x for x in data_list if x == x and x is not None])
    if not clean:
        return 0.0
    n   = len(clean)
    idx = (pct / 100.0) * (n - 1)
    lo  = int(idx)
    hi  = min(lo + 1, n - 1)
    return clean[lo] + (idx - lo) * (clean[hi] - clean[lo])

# ── DATA ─────────────────────────────────────────────────────
@st.cache_data
def fetch_data():
    df = yf.download("TSLA", start="2019-01-01", end="2024-01-01",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].squeeze(), errors="coerce")
    df.dropna(inplace=True)
    return df

def calc_returns(df):
    df = df.copy()
    close = df["Close"].squeeze()
    df["Simple_Return"] = close.pct_change()
    df["Log_Return"]    = np.log(close / close.shift(1))
    df.dropna(inplace=True)
    return df

def calc_indicators(df):
    df   = df.copy()
    c    = df["Close"].squeeze()
    h    = df["High"].squeeze()
    l    = df["Low"].squeeze()
    df["SMA_20"]  = c.rolling(20).mean()
    df["SMA_50"]  = c.rolling(50).mean()
    df["SMA_200"] = c.rolling(200).mean()
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["EMA_12"]  = ema12
    df["EMA_26"]  = ema26
    bb_std         = c.rolling(20).std()
    df["BB_Upper"] = df["SMA_20"] + 2 * bb_std
    df["BB_Lower"] = df["SMA_20"] - 2 * bb_std
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    hl  = h - l
    hc  = (h - c.shift()).abs()
    lc  = (l - c.shift()).abs()
    df["ATR"]         = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    lr = df["Log_Return"].squeeze()
    df["Vol_20d"] = lr.rolling(20).std() * np.sqrt(252)
    df["Vol_60d"] = lr.rolling(60).std() * np.sqrt(252)
    return df

def calc_var_cvar(df):
    """Uses pure Python percentile — no numpy percentile calls."""
    lr     = df["Log_Return"].squeeze()
    r_list = [float(x) for x in lr.dropna().values if str(x) != "nan"]
    n      = len(r_list)
    mu     = sum(r_list) / n
    sigma  = (sum((x - mu)**2 for x in r_list) / n) ** 0.5

    results = {}
    for cl in [0.90, 0.95, 0.99]:
        alpha = 1 - cl
        label = f"{int(cl*100)}%"
        z     = float(stats.norm.ppf(alpha))

        hvar  = py_percentile(r_list, alpha * 100)
        htail = [x for x in r_list if x <= hvar]
        hcvar = sum(htail) / len(htail) if htail else hvar

        pvar  = mu + z * sigma
        pcvar = mu - sigma * float(stats.norm.pdf(z)) / alpha

        random.seed(42)
        sim   = [random.gauss(mu, sigma) for _ in range(10000)]
        mvar  = py_percentile(sim, alpha * 100)
        mtail = [x for x in sim if x <= mvar]
        mcvar = sum(mtail) / len(mtail) if mtail else mvar

        results[label] = {
            "Historical VaR":  hvar,  "Historical CVaR":  hcvar,
            "Parametric VaR":  pvar,  "Parametric CVaR":  pcvar,
            "Monte Carlo VaR": mvar,  "Monte Carlo CVaR": mcvar,
        }
    return results

def calc_stats(df):
    r   = df["Log_Return"].squeeze().dropna()
    ar  = float(r.mean() * 252)
    av  = float(r.std()  * np.sqrt(252))
    rf  = 0.05
    sh  = (ar - rf) / av
    dv  = float(r[r < 0].std() * np.sqrt(252))
    so  = (ar - rf) / dv
    cum = (1 + r).cumprod()
    rm  = cum.cummax()
    dd  = (cum - rm) / rm
    md  = float(dd.min())
    cal = ar / abs(md) if md != 0 else 0
    return {
        "Annual Return": ar, "Annual Volatility": av,
        "Sharpe Ratio":  sh, "Sortino Ratio":     so,
        "Max Drawdown":  md, "Calmar Ratio":      cal,
        "Skewness":      float(r.skew()),
        "Kurtosis":      float(r.kurt()),
        "Win Rate":      float((r > 0).sum() / len(r)),
        "Drawdown":      dd, "Cumulative":        cum,
    }

# ── CHARTS ───────────────────────────────────────────────────
def chart_price(df, bb=True, sma=True):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.02)
    c = df["Close"].squeeze()
    o = df["Open"].squeeze()
    fig.add_trace(go.Candlestick(
        x=df.index, open=o, high=df["High"].squeeze(),
        low=df["Low"].squeeze(), close=c,
        increasing=dict(line=dict(color=NG, width=1), fillcolor=NG),
        decreasing=dict(line=dict(color=NR, width=1), fillcolor=NR),
        name="TSLA"), row=1, col=1)
    if bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"],
            line=dict(color=NC, width=2), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"],
            line=dict(color=NC, width=2), fill="tonexty",
            fillcolor="rgba(0,245,255,0.07)", name="BB Lower"), row=1, col=1)
    if sma:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"],
            line=dict(color=NY, width=2.5), name="SMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"],
            line=dict(color=NO, width=2.5), name="SMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"],
            line=dict(color=NP, width=2.5), name="SMA 200"), row=1, col=1)
    vc = [NG if cv >= ov else NR for cv, ov in zip(c, o)]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"].squeeze(),
        marker_color=vc, opacity=0.75, name="Volume"), row=2, col=1)
    lay = dict(**BASE)
    lay["yaxis2"] = dict(gridcolor=GR, color=TC, tickfont=dict(color=TC, size=11))
    fig.update_layout(**lay,
        title=dict(text="TSLA — Price Chart", font=dict(color=NC, size=17)),
        xaxis_rangeslider_visible=False, height=540)
    return fig

def chart_dist(df, vd):
    r   = [float(x) for x in df["Log_Return"].squeeze().dropna().values]
    r_s = [x * 100 for x in r]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r_s, nbinsx=80,
        marker=dict(color=NC, line=dict(color="rgba(0,245,255,0.3)", width=0.5)),
        opacity=0.85, name="Daily Returns"))
    mu_s  = sum(r_s) / len(r_s)
    sig_s = (sum((x-mu_s)**2 for x in r_s)/len(r_s))**0.5
    x_r   = [min(r_s) + i*(max(r_s)-min(r_s))/299 for i in range(300)]
    ny_v  = [stats.norm.pdf(x, mu_s, sig_s) * len(r_s) * (max(r_s)-min(r_s))/80 for x in x_r]
    fig.add_trace(go.Scatter(x=x_r, y=ny_v,
        line=dict(color=NY, width=3), name="Normal Fit"))
    v95  = vd["95%"]["Historical VaR"]  * 100
    cv95 = vd["95%"]["Historical CVaR"] * 100
    fig.add_vline(x=v95,  line=dict(color=NO, width=3, dash="dash"),
        annotation=dict(text=f"VaR 95%: {v95:.2f}%",
                        font=dict(color=NO, size=13), bgcolor=BC))
    fig.add_vline(x=cv95, line=dict(color=NR, width=3, dash="dash"),
        annotation=dict(text=f"CVaR 95%: {cv95:.2f}%",
                        font=dict(color=NR, size=13), bgcolor=BC))
    fig.update_layout(**BASE,
        title=dict(text="Return Distribution + VaR/CVaR", font=dict(color=NC, size=16)),
        height=400, xaxis_title="Daily Log Return (%)", yaxis_title="Frequency")
    return fig

def chart_var(vd):
    cls  = ["90%","95%","99%"]
    mths = ["Historical","Parametric","Monte Carlo"]
    cols = [NC, NP, NG]
    fig  = make_subplots(rows=1, cols=2,
        subplot_titles=["Value at Risk (VaR)", "CVaR — Expected Shortfall"])
    fig.update_annotations(font=dict(color=NW, size=14))
    for i, m in enumerate(mths):
        vv = [vd[c][f"{m} VaR"]  * 100 for c in cls]
        cv = [vd[c][f"{m} CVaR"] * 100 for c in cls]
        fig.add_trace(go.Bar(name=m, x=cls, y=vv,
            marker=dict(color=cols[i]), opacity=0.9, showlegend=True), row=1, col=1)
        fig.add_trace(go.Bar(name=m, x=cls, y=cv,
            marker=dict(color=cols[i]), opacity=0.9, showlegend=False), row=1, col=2)
    lay = dict(**BASE)
    lay["yaxis2"] = dict(gridcolor=GR, color=TC, tickfont=dict(color=TC, size=12))
    fig.update_layout(**lay,
        title=dict(text="VaR & CVaR — Method Comparison", font=dict(color=NC, size=16)),
        barmode="group", height=400)
    return fig

def chart_drawdown(sd):
    dd = sd["Drawdown"] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values,
        fill="tozeroy", fillcolor="rgba(255,45,85,0.25)",
        line=dict(color=NR, width=2.5), name="Drawdown"))
    ddate = dd.idxmin()
    fig.add_annotation(x=ddate, y=float(dd.min()),
        text=f"Max: {dd.min():.1f}%", showarrow=True, arrowhead=2,
        arrowcolor=NY, font=dict(color=NY, size=13), bgcolor=BC)
    fig.update_layout(**BASE,
        title=dict(text="Drawdown from Peak", font=dict(color=NC, size=16)),
        height=330, yaxis_title="Drawdown (%)")
    return fig

def chart_rsi(df):
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,45,85,0.18)",  line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,255,136,0.18)", line_width=0)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"].squeeze(),
        line=dict(color=NW, width=3), name="RSI (14)"))
    fig.add_hline(y=70, line=dict(color=NR, width=2, dash="dot"))
    fig.add_hline(y=30, line=dict(color=NG, width=2, dash="dot"))
    fig.add_hline(y=50, line=dict(color=MC, width=1, dash="dot"))
    fig.add_annotation(x=df.index[30], y=75, text="OVERBOUGHT  70",
        font=dict(color=NR, size=12), showarrow=False, bgcolor=BC)
    fig.add_annotation(x=df.index[30], y=25, text="OVERSOLD  30",
        font=dict(color=NG, size=12), showarrow=False, bgcolor=BC)
    lay = dict(**BASE)
    lay["yaxis"] = dict(gridcolor=GR, color=TC, tickfont=dict(color=TC, size=12),
                        range=[0,100], title="RSI", title_font=dict(color=TC))
    fig.update_layout(**lay,
        title=dict(text="RSI — Relative Strength Index (14)", font=dict(color=NC, size=16)),
        height=330)
    return fig

def chart_macd(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6,0.4], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"].squeeze(),
        line=dict(color=NC, width=2.5), name="MACD"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"].squeeze(),
        line=dict(color=NP, width=2.5), name="Signal"), row=1, col=1)
    hc = [NG if v >= 0 else NR for v in df["MACD_Hist"].squeeze()]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"].squeeze(),
        marker_color=hc, opacity=0.85, name="Histogram"), row=2, col=1)
    lay = dict(**BASE)
    lay["yaxis2"] = dict(gridcolor=GR, color=TC, tickfont=dict(color=TC, size=12))
    fig.update_layout(**lay,
        title=dict(text="MACD — Trend Momentum", font=dict(color=NC, size=16)),
        height=390)
    return fig

def chart_vol(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol_20d"].squeeze()*100,
        line=dict(color=NC, width=2.5), fill="tozeroy",
        fillcolor="rgba(0,245,255,0.1)", name="20-day Vol"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol_60d"].squeeze()*100,
        line=dict(color=NP, width=3), name="60-day Vol"))
    fig.update_layout(**BASE,
        title=dict(text="Rolling Annualized Volatility (%)", font=dict(color=NC, size=16)),
        height=330, yaxis_title="Volatility (%)")
    return fig

def chart_cum(sd):
    cum = sd["Cumulative"] * 100 - 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=cum.values,
        line=dict(color=NG, width=2.5), fill="tozeroy",
        fillcolor="rgba(0,255,136,0.1)", name="Cumulative Return"))
    fig.add_hline(y=0, line=dict(color=MC, width=1, dash="dot"))
    fig.update_layout(**BASE,
        title=dict(text="Cumulative Return Since 2019", font=dict(color=NC, size=16)),
        height=330, yaxis_title="Return (%)")
    return fig

# ── LOAD ─────────────────────────────────────────────────────
with st.spinner("⏳ Fetching TSLA data..."):
    raw  = fetch_data()
    ret  = calc_returns(raw)
    full = calc_indicators(ret)
    vd   = calc_var_cvar(full)
    sd   = calc_stats(full)

# ── HEADER ───────────────────────────────────────────────────
ca, cb = st.columns([3,1])
with ca:
    st.markdown(f"""
    <div style='font-family:Consolas,monospace'>
        <span style='color:{NC};font-size:36px;font-weight:900;letter-spacing:8px;
                     text-shadow:0 0 20px {NC}'>QUANT</span>
        <span style='color:{NP};font-size:36px'> // </span>
        <span style='color:#fff;font-size:36px;font-weight:900;letter-spacing:8px'>RISK</span>
    </div>
    <p style='color:{MC};font-size:12px;letter-spacing:3px;font-family:Consolas,monospace;margin-top:4px'>
        TSLA — MARKET RISK & TECHNICAL ANALYSIS TERMINAL | 2019–2024
    </p>""", unsafe_allow_html=True)
with cb:
    st.markdown(f"""
    <div style='text-align:right;padding-top:10px'>
        <p style='color:{MC};font-size:10px;letter-spacing:1px;font-family:Consolas,monospace;margin:0'>
            Quantitative Market Risk &<br>Technical Analysis Dashboard
        </p>
        <p style='color:{NC};font-size:11px;letter-spacing:2px;font-family:Consolas,monospace;
                  margin:4px 0 0 0;opacity:0.8'>Hassaan Ali</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── KPIs ─────────────────────────────────────────────────────
latest  = float(full["Close"].squeeze().iloc[-1])
v95     = vd["95%"]["Historical VaR"]  * 100
cv95    = vd["95%"]["Historical CVaR"] * 100
rsi_now = float(full["RSI"].squeeze().iloc[-1])

k = st.columns(8)
k[0].metric("TSLA Price",    f"${latest:.2f}")
k[1].metric("VaR 95%",       f"{v95:.2f}%")
k[2].metric("CVaR 95%",      f"{cv95:.2f}%")
k[3].metric("Annual Return", f"{sd['Annual Return']*100:.1f}%")
k[4].metric("Annual Vol",    f"{sd['Annual Volatility']*100:.1f}%")
k[5].metric("Sharpe Ratio",  f"{sd['Sharpe Ratio']:.2f}")
k[6].metric("Max Drawdown",  f"{sd['Max Drawdown']*100:.1f}%")
k[7].metric("RSI (14)",      f"{rsi_now:.1f}")

st.markdown("<hr>", unsafe_allow_html=True)

# ── CONTROLS ─────────────────────────────────────────────────
c1, c2 = st.columns(2)
show_bb  = c1.checkbox("Show Bollinger Bands", value=True)
show_sma = c2.checkbox("Show Moving Averages", value=True)

# ── CHARTS ───────────────────────────────────────────────────
st.plotly_chart(chart_price(full, show_bb, show_sma), use_container_width=True)

col1, col2 = st.columns(2)
col1.plotly_chart(chart_dist(full, vd),  use_container_width=True)
col2.plotly_chart(chart_var(vd),         use_container_width=True)

col3, col4 = st.columns(2)
col3.plotly_chart(chart_cum(sd),         use_container_width=True)
col4.plotly_chart(chart_drawdown(sd),    use_container_width=True)

col5, col6 = st.columns(2)
col5.plotly_chart(chart_vol(full),       use_container_width=True)
col6.plotly_chart(chart_rsi(full),       use_container_width=True)

st.plotly_chart(chart_macd(full),        use_container_width=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── VAR TABLE ────────────────────────────────────────────────
st.markdown(f"<h3 style='color:{NC};font-family:Consolas,monospace;"
            f"letter-spacing:3px;font-size:14px'>VaR / CVaR — FULL RESULTS TABLE</h3>",
            unsafe_allow_html=True)
rows = []
for cl in ["90%","95%","99%"]:
    for m in ["Historical","Parametric","Monte Carlo"]:
        rows.append({
            "Confidence": cl, "Method": m,
            "VaR":  f"{vd[cl][f'{m} VaR']*100:.3f}%",
            "CVaR": f"{vd[cl][f'{m} CVaR']*100:.3f}%",
        })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── RISK STATS ───────────────────────────────────────────────
st.markdown(f"<h3 style='color:{NC};font-family:Consolas,monospace;"
            f"letter-spacing:3px;font-size:14px'>PORTFOLIO RISK STATISTICS</h3>",
            unsafe_allow_html=True)
s = st.columns(9)
s[0].metric("Annual Return",  f"{sd['Annual Return']*100:.2f}%")
s[1].metric("Annual Vol",     f"{sd['Annual Volatility']*100:.2f}%")
s[2].metric("Sharpe Ratio",   f"{sd['Sharpe Ratio']:.3f}")
s[3].metric("Sortino Ratio",  f"{sd['Sortino Ratio']:.3f}")
s[4].metric("Max Drawdown",   f"{sd['Max Drawdown']*100:.2f}%")
s[5].metric("Calmar Ratio",   f"{sd['Calmar Ratio']:.3f}")
s[6].metric("Skewness",       f"{sd['Skewness']:.3f}")
s[7].metric("Kurtosis",       f"{sd['Kurtosis']:.3f}")
s[8].metric("Win Rate",       f"{sd['Win Rate']*100:.1f}%")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#1a3a4a;font-size:11px;text-align:center;"
            f"font-family:Consolas,monospace;letter-spacing:2px'>"
            f"QUANT//RISK · Data: Yahoo Finance · TSLA 2019–2024 · Hassaan Ali</p>",
            unsafe_allow_html=True)
