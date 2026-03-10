"""
Historical Trends — Heatwave Days + Maximum Temperature + LST + Heat Index
Module: Scenario Analysis | ResSolv
"""

import json
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Historical Trends | ResSolv",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        html { font-size: 75%; }
        .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
        [data-testid="stMetric"] {
            background: #f8f9fb;
            border: 1px solid #e8eaf0;
            border-radius: 10px;
            padding: 0.4rem 0.5rem;
        }
        [data-testid="stMetricLabel"] { font-size: 0.56rem; color: #666; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        [data-testid="stMetricValue"] { font-size: 0.82rem; font-weight: 700; color: #1a1a2e; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .section-title {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            color: #999;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }
        .chart-label {
            font-size: 0.75rem;
            color: #888;
            margin-bottom: 0.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
HW_FILE   = Path(__file__).parent / "Heatwave Days"    / "Raghwa_27.3954_70.4780.json"
MAXT_FILE = Path(__file__).parent / "max temp weekly"  / "Raghwa_27.3954_70.4780.json"
LST_FILE  = Path(__file__).parent / "output_jsons_lst" / "Raghwa_LST.json"
HI_FILE   = Path(__file__).parent / "hi"               / "Ragha_new_HI_result.json"

THRESHOLD_SPELL_C  = 40.0
THRESHOLD_SEVERE_C = 45.0

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

MONTH_COLORS = {
    1:  "#4E79A7", 2:  "#76B7B2", 3:  "#59A14F", 4:  "#EDC948",
    5:  "#F28E2B", 6:  "#E15759", 7:  "#B07AA1", 8:  "#FF9DA7",
    9:  "#9C755F", 10: "#BAB0AC", 11: "#D37295", 12: "#499894",
}

TIME_OPTIONS = [
    "Last 1 Month", "Last 6 Months", "Last 1 Year",
    "Last 5 Years", "Last 10 Years", "Last 20 Years", "Custom Range",
]

DAYS_MAP = {
    "Last 1 Month":   30,   "Last 6 Months":  182,
    "Last 1 Year":    365,  "Last 5 Years":   365 * 5,
    "Last 10 Years":  365 * 10, "Last 20 Years": 365 * 20,
}

AGG_MODE_MAP = {
    "Last 1 Month":   "daily",   "Last 6 Months":  "weekly",
    "Last 1 Year":    "weekly",  "Last 5 Years":   "monthly",
    "Last 10 Years":  "yearly",  "Last 20 Years":  "yearly",
}

CHART_H = 300   # shared chart height


# ── Data loaders (cached) ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading climate data…")
def load_raw(filepath: str) -> pd.DataFrame:
    with open(filepath, encoding="utf-8") as fh:
        raw = json.load(fh)
    t2m = raw["properties"]["parameter"]["T2M_MAX"]
    df = pd.DataFrame(t2m.items(), columns=["date_str", "tmax_c"])
    df["date"]   = pd.to_datetime(df["date_str"], format="%Y%m%d")
    df["tmax_f"] = df["tmax_c"] * 9 / 5 + 32
    df = df.drop(columns="date_str").sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Detecting heatwave days…")
def detect_heatwave(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    severe     = df["tmax_c"] >= THRESHOLD_SEVERE_C
    above40    = (df["tmax_c"] >= THRESHOLD_SPELL_C).astype(int)
    run_groups = (above40 != above40.shift()).cumsum()
    run_length = above40.groupby(run_groups).transform("sum") * above40
    df["is_heatwave"] = ((run_length >= 2) | severe).astype(int)
    return df


@st.cache_data(show_spinner="Loading LST data…")
def load_lst(filepath: str) -> pd.DataFrame:
    with open(filepath, encoding="utf-8") as fh:
        raw = json.load(fh)
    df = pd.DataFrame(raw)
    df["lst_c"]  = pd.to_numeric(df["LST_C"], errors="coerce")
    df["lst_f"]  = df["lst_c"] * 9 / 5 + 32
    df["period"] = pd.to_datetime(df["date"] + "-01", format="%Y-%m-%d")
    df["year"]   = df["period"].dt.year
    df["month"]  = df["period"].dt.month
    df = df[["period", "year", "month", "lst_c", "lst_f"]].sort_values("period").reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Loading Heat Index data…")
def load_hi(filepath: str) -> pd.DataFrame:
    with open(filepath, encoding="utf-8") as fh:
        raw = json.load(fh)
    df = pd.DataFrame(raw)
    df["date"]  = pd.to_datetime(df["date"])
    df["hi_c"]  = pd.to_numeric(df["HI"], errors="coerce")
    df["hi_f"]  = df["hi_c"] * 9 / 5 + 32
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df = df[["date", "year", "month", "hi_c", "hi_f"]].sort_values("date").reset_index(drop=True)
    return df


@st.cache_data
def monthly_hw_counts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["year"]  = d["date"].dt.year
    d["month"] = d["date"].dt.month
    m = (
        d.groupby(["year", "month"])["is_heatwave"]
        .sum().reset_index()
        .rename(columns={"is_heatwave": "heatwave_days"})
    )
    m["period"] = pd.to_datetime(
        m["year"].astype(str) + "-" + m["month"].astype(str).str.zfill(2) + "-01"
    )
    return m


# ── Load data ──────────────────────────────────────────────────────────────────
raw_hw   = load_raw(str(HW_FILE))
hw_df    = detect_heatwave(raw_hw)
monthly_hw = monthly_hw_counts(hw_df)

raw_mt   = load_raw(str(MAXT_FILE))   # max-temp dataset
lst_df   = load_lst(str(LST_FILE))    # LST monthly dataset
hi_df    = load_hi(str(HI_FILE))      # Heat Index weekly dataset

DATA_START     = raw_hw["date"].min().date()
DATA_END       = raw_hw["date"].max().date()
LST_DATA_START = lst_df["period"].min().date()
LST_DATA_END   = lst_df["period"].max().date()
HI_DATA_START  = hi_df["date"].min().date()
HI_DATA_END    = hi_df["date"].max().date()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filters")

    st.markdown('<p class="section-title">Time Range</p>', unsafe_allow_html=True)
    selected_range = st.radio(
        label="time_range", options=TIME_OPTIONS, index=5,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<p class="section-title">Temperature Unit</p>', unsafe_allow_html=True)
    unit = st.radio(
        label="unit", options=["Celsius (°C)", "Fahrenheit (°F)"], index=0,
        label_visibility="collapsed",
    )
    use_fahrenheit = unit.startswith("F")
    unit_label     = "°F" if use_fahrenheit else "°C"
    temp_col       = "tmax_f" if use_fahrenheit else "tmax_c"

    if selected_range == "Custom Range":
        st.markdown("---")
        st.markdown('<p class="section-title">Custom Date Range</p>', unsafe_allow_html=True)
        custom_start = st.date_input(
            "From", value=DATA_END - timedelta(days=365 * 5),
            min_value=DATA_START, max_value=DATA_END,
        )
        custom_end = st.date_input(
            "To", value=DATA_END, min_value=DATA_START, max_value=DATA_END,
        )
        if custom_start >= custom_end:
            st.error("Start date must be before end date.")
            st.stop()
        range_start = pd.Timestamp(custom_start)
        range_end   = pd.Timestamp(custom_end)
    else:
        range_end   = pd.Timestamp(DATA_END)
        range_start = range_end - pd.Timedelta(days=DAYS_MAP[selected_range])


# ── Aggregation mode ───────────────────────────────────────────────────────────
def resolve_agg_mode(selected: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    if selected != "Custom Range":
        return AGG_MODE_MAP[selected]
    span_days  = (end - start).days
    span_years = span_days / 365.25
    if span_years > 8:   return "yearly"
    if span_years >= 2:  return "monthly"
    if span_days > 31:   return "weekly"
    return "daily"


agg_mode = resolve_agg_mode(selected_range, range_start, range_end)


# ── Filter windows ─────────────────────────────────────────────────────────────
hw_mask   = (hw_df["date"] >= range_start)         & (hw_df["date"] <= range_end)
mt_mask   = (raw_mt["date"] >= range_start)        & (raw_mt["date"] <= range_end)
mhw_mask  = (monthly_hw["period"] >= range_start)  & (monthly_hw["period"] <= range_end)
lst_mask  = (lst_df["period"] >= range_start)  & (lst_df["period"] <= range_end)
hi_mask   = (hi_df["date"]   >= range_start)   & (hi_df["date"]   <= range_end)

f_hw_daily   = hw_df[hw_mask].copy()
f_mt_daily   = raw_mt[mt_mask].copy()
f_monthly_hw = monthly_hw[mhw_mask].copy()
f_lst        = lst_df[lst_mask].copy()
f_hi         = hi_df[hi_mask].copy()


# ── Heatwave chart data ────────────────────────────────────────────────────────
def build_hw_plot(mode: str) -> tuple[pd.DataFrame, str, str]:
    """Build plot_df for Heatwave Days chart."""
    if mode == "daily":
        df = f_monthly_hw.rename(columns={"heatwave_days": "value"}).copy()
        df["label"] = df["period"].dt.strftime("%b %Y")
        df["color"] = df["month"].map(MONTH_COLORS)
        return df[["period", "value", "label", "color"]], "Month", "Heatwave Days"

    if mode == "weekly":
        d = f_hw_daily.copy()
        d["week_start"] = d["date"].dt.to_period("W").apply(lambda p: p.start_time)
        w = d.groupby("week_start")["is_heatwave"].sum().reset_index()
        w.columns = ["period", "value"]
        w["label"] = w["period"].dt.strftime("Week of %d %b %Y")
        w["color"] = "#E15759"
        return w, "Week", "Heatwave Days"

    if mode == "monthly":
        df = f_monthly_hw.rename(columns={"heatwave_days": "value"}).copy()
        df["label"] = df["period"].dt.strftime("%b %Y")
        df["color"] = df["month"].map(MONTH_COLORS)
        return df[["period", "value", "label", "color"]], "Month", "Heatwave Days"

    # yearly
    fm = f_monthly_hw.copy()
    total = fm.groupby("year")["heatwave_days"].sum().reset_index().rename(columns={"heatwave_days": "value"})
    peak  = fm.loc[fm.groupby("year")["heatwave_days"].idxmax(), ["year", "month"]].rename(columns={"month": "peak_month_num"})
    yearly = total.merge(peak, on="year")
    yearly["period"] = pd.to_datetime(yearly["year"].astype(str) + "-01-01")
    yearly["label"]  = yearly["year"].astype(str)
    yearly["color"]  = yearly["peak_month_num"].map(MONTH_COLORS)
    return yearly[["period", "value", "label", "color", "peak_month_num"]], "Year", "Total Heatwave Days / Year"


# ── Max-temperature chart data ─────────────────────────────────────────────────
def build_mt_plot(mode: str) -> tuple[pd.DataFrame, str, str]:
    """Build plot_df for Maximum Temperature chart. Values in selected unit."""
    y_title = f"Max Temperature ({unit_label})"

    if mode == "daily":
        df = f_mt_daily[["date", temp_col]].rename(columns={"date": "period", temp_col: "value"})
        df["label"] = df["period"].dt.strftime("%d %b %Y")
        df["color"] = "#4E79A7"
        return df, "Date", y_title

    if mode == "weekly":
        d = f_mt_daily.copy()
        d["week_start"] = d["date"].dt.to_period("W").apply(lambda p: p.start_time)
        w = d.groupby("week_start")[temp_col].max().reset_index()
        w.columns = ["period", "value"]
        w["label"] = w["period"].dt.strftime("Week of %d %b %Y")
        w["color"] = "#4E79A7"
        return w, "Week", y_title

    if mode == "monthly":
        d = f_mt_daily.copy()
        d["year"]  = d["date"].dt.year
        d["month"] = d["date"].dt.month
        m = d.groupby(["year", "month"])[temp_col].max().reset_index()
        m.columns = ["year", "month", "value"]
        m["period"] = pd.to_datetime(m["year"].astype(str) + "-" + m["month"].astype(str).str.zfill(2) + "-01")
        m["label"]  = m["period"].dt.strftime("%b %Y")
        m["color"]  = m["month"].map(MONTH_COLORS)
        return m[["period", "value", "label", "color"]], "Month", y_title

    # yearly — highest temp of the year, colored by month where it occurred
    d = f_mt_daily.copy()
    d["year"]  = d["date"].dt.year
    d["month"] = d["date"].dt.month
    idx_max    = d.groupby("year")[temp_col].idxmax()
    yearly     = d.loc[idx_max, ["year", "month", temp_col]].reset_index(drop=True)
    yearly.columns = ["year", "peak_month_num", "value"]
    yearly["period"] = pd.to_datetime(yearly["year"].astype(str) + "-01-01")
    yearly["label"]  = yearly["year"].astype(str)
    yearly["color"]  = yearly["peak_month_num"].map(MONTH_COLORS)
    return yearly[["period", "value", "label", "color", "peak_month_num"]], "Year", y_title


# ── Shared chart layout ────────────────────────────────────────────────────────
def base_layout(title: str, x_title: str, y_title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14, color="#333", family="sans-serif"), x=0, xanchor="left"),
        xaxis=dict(title=dict(text=x_title, font=dict(size=11, color="#555")), showgrid=False, zeroline=False, linecolor="#e0e0e0"),
        yaxis=dict(title=dict(text=y_title, font=dict(size=11, color="#555")), showgrid=True, gridcolor="#f0f0f0", zeroline=False, linecolor="#e0e0e0"),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=55, r=25, t=65, b=55),
        font=dict(family="Inter, Helvetica Neue, sans-serif", size=11, color="#333"),
        height=CHART_H,
    )


def add_avg_line(fig: go.Figure, avg: float, label: str) -> None:
    fig.add_hline(
        y=avg, line_dash="dash", line_color="#aaaaaa", line_width=1.5,
        annotation_text=f"Avg  {avg:.1f}  {label}",
        annotation_position="top right",
        annotation_font=dict(color="#888", size=10),
    )


# ── Build heatwave figure ──────────────────────────────────────────────────────
def make_hw_fig(mode: str) -> go.Figure:
    plot_df, x_title, y_title = build_hw_plot(mode)
    avg_val = plot_df["value"].mean() if not plot_df.empty else 0
    fig = go.Figure()

    if mode == "daily":
        fig.add_trace(go.Scatter(
            x=plot_df["period"], y=plot_df["value"], mode="lines+markers",
            name="Heatwave Days",
            line=dict(color="#ddd", width=1.5),
            marker=dict(size=9, color=plot_df["color"], line=dict(width=1, color="white")),
            customdata=np.column_stack([plot_df["label"], plot_df["value"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>Heatwave Days = %{customdata[1]:.0f}<extra></extra>",
        ))

    elif mode == "weekly":
        fig.add_trace(go.Scatter(
            x=plot_df["period"], y=plot_df["value"], mode="lines+markers",
            name="Heatwave Days",
            line=dict(color="#E15759", width=2),
            marker=dict(size=5, color="#E15759"),
            fill="tozeroy", fillcolor="rgba(225,87,89,0.08)",
            customdata=np.column_stack([plot_df["label"], plot_df["value"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>Heatwave Days = %{customdata[1]}<extra></extra>",
        ))

    elif mode == "monthly":
        fig.add_trace(go.Scatter(
            x=plot_df["period"], y=plot_df["value"], mode="lines+markers",
            name="Heatwave Days",
            line=dict(color="#ddd", width=1.5),
            marker=dict(size=9, color=plot_df["color"], line=dict(width=1, color="white")),
            customdata=np.column_stack([plot_df["label"], plot_df["value"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>Heatwave Days = %{customdata[1]:.0f}<extra></extra>",
        ))

    else:  # yearly
        hover_texts = [
            f"<b>{r['label']}</b><br>Total Heatwave Days = {int(r['value'])}<br>"
            f"Peak Month = {MONTH_NAMES.get(int(r['peak_month_num']), '—')}"
            for _, r in plot_df.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=plot_df["period"], y=plot_df["value"], mode="lines+markers",
            name="Heatwave Days",
            line=dict(color="#ddd", width=1.5),
            marker=dict(size=13, color=plot_df["color"], line=dict(width=1.5, color="white")),
            text=hover_texts, hovertemplate="%{text}<extra></extra>",
        ))

    if not plot_df.empty:
        add_avg_line(fig, avg_val, "days")

    fig.update_layout(**base_layout(f"Heatwave Days — {selected_range}", x_title, y_title))
    return fig


# ── Build max-temperature figure ───────────────────────────────────────────────
def make_mt_fig(mode: str) -> go.Figure:
    plot_df, x_title, y_title = build_mt_plot(mode)
    avg_val = plot_df["value"].mean() if not plot_df.empty else 0
    fig = go.Figure()

    LINE_COLOR   = "#3A6EA5"
    FILL_COLOR   = "rgba(58,110,165,0.07)"
    TREND_COLOR  = "#E15759"

    if mode == "daily":
        fig.add_trace(go.Scatter(
            x=plot_df["period"], y=plot_df["value"], mode="lines",
            name=f"Daily Tmax",
            line=dict(color=LINE_COLOR, width=1.2),
            fill="tozeroy", fillcolor=FILL_COLOR,
            customdata=np.column_stack([plot_df["label"], plot_df["value"]]),
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>Max Temp = %{{customdata[1]:.1f}} {unit_label}<extra></extra>",
        ))

    elif mode == "weekly":
        fig.add_trace(go.Scatter(
            x=plot_df["period"], y=plot_df["value"], mode="lines+markers",
            name=f"Weekly Tmax",
            line=dict(color=LINE_COLOR, width=1.5),
            marker=dict(size=4, color=LINE_COLOR),
            fill="tozeroy", fillcolor=FILL_COLOR,
            customdata=np.column_stack([plot_df["label"], plot_df["value"]]),
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>Max Temp = %{{customdata[1]:.1f}} {unit_label}<extra></extra>",
        ))

    elif mode == "monthly":
        fig.add_trace(go.Scatter(
            x=plot_df["period"], y=plot_df["value"], mode="lines+markers",
            name=f"Monthly Tmax",
            line=dict(color="#ddd", width=1.5),
            marker=dict(size=8, color=plot_df["color"], line=dict(width=1, color="white")),
            customdata=np.column_stack([plot_df["label"], plot_df["value"]]),
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>Max Temp = %{{customdata[1]:.1f}} {unit_label}<extra></extra>",
        ))

    else:  # yearly
        hover_texts = [
            f"<b>{r['label']}</b><br>Highest Temp = {r['value']:.1f} {unit_label}<br>"
            f"Month = {MONTH_NAMES.get(int(r['peak_month_num']), '—')}"
            for _, r in plot_df.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=plot_df["period"], y=plot_df["value"], mode="lines+markers",
            name="Annual Peak Tmax",
            line=dict(color="#ddd", width=1.5),
            marker=dict(size=13, color=plot_df["color"], line=dict(width=1.5, color="white")),
            text=hover_texts, hovertemplate="%{text}<extra></extra>",
        ))

    # ── Trendline (linear regression) ──
    if len(plot_df) >= 3:
        x_ord = plot_df["period"].map(pd.Timestamp.toordinal).values.astype(float)
        y_val = plot_df["value"].values.astype(float)
        mask  = ~np.isnan(y_val)
        if mask.sum() >= 2:
            coeffs    = np.polyfit(x_ord[mask], y_val[mask], 1)
            trend_y   = np.polyval(coeffs, x_ord)
            slope_per_decade = coeffs[0] * 365.25 * 10   # change per decade

            direction = "↑" if slope_per_decade > 0 else "↓"
            trend_label = (
                f"Trend  {direction} {abs(slope_per_decade):.2f} {unit_label}/decade"
            )
            fig.add_trace(go.Scatter(
                x=plot_df["period"], y=trend_y,
                mode="lines", name=trend_label,
                line=dict(color=TREND_COLOR, width=2, dash="dot"),
                hoverinfo="skip",
            ))

    if not plot_df.empty:
        add_avg_line(fig, avg_val, unit_label)

    fig.update_layout(**base_layout(f"Maximum Temperature — {selected_range}", x_title, y_title))
    return fig


# ── LST chart data ─────────────────────────────────────────────────────────────
LST_LINE_COLOR  = "#2E8B57"
LST_FILL_COLOR  = "rgba(46,139,87,0.08)"
LST_TREND_COLOR = "#D4A017"

# LST is monthly — map finer agg modes to "monthly"
def lst_agg_mode(mode: str) -> str:
    return "monthly" if mode in ("daily", "weekly", "monthly") else "yearly"


def build_lst_plot(mode: str) -> tuple[pd.DataFrame, str, str]:
    lst_col   = "lst_f" if use_fahrenheit else "lst_c"
    y_title   = f"LST ({unit_label})"
    effective = lst_agg_mode(mode)

    if effective == "monthly":
        df = f_lst[["period", "month", lst_col]].copy()
        df = df.rename(columns={lst_col: "value"})
        df["label"] = df["period"].dt.strftime("%b %Y")
        df["color"] = df["month"].map(MONTH_COLORS)
        return df[["period", "value", "label", "color"]], "Month", y_title

    # yearly — mean LST per year, colored by warmest month
    lst_col_data = "lst_f" if use_fahrenheit else "lst_c"
    yearly_mean = f_lst.groupby("year")[lst_col_data].mean().reset_index().rename(columns={lst_col_data: "value"})
    peak_month  = f_lst.loc[f_lst.groupby("year")[lst_col_data].idxmax(), ["year", "month"]].rename(columns={"month": "peak_month_num"})
    yearly = yearly_mean.merge(peak_month, on="year")
    yearly["period"] = pd.to_datetime(yearly["year"].astype(str) + "-01-01")
    yearly["label"]  = yearly["year"].astype(str)
    yearly["color"]  = yearly["peak_month_num"].map(MONTH_COLORS)
    return yearly[["period", "value", "label", "color", "peak_month_num"]], "Year", y_title


def make_lst_fig(mode: str) -> go.Figure:
    plot_df, x_title, y_title = build_lst_plot(mode)
    avg_val   = plot_df["value"].mean() if not plot_df.empty else 0
    effective = lst_agg_mode(mode)
    fig = go.Figure()

    if effective == "monthly":
        fig.add_trace(go.Scatter(
            x=plot_df["period"], y=plot_df["value"], mode="lines+markers",
            name="Monthly LST",
            line=dict(color="#ddd", width=1.5),
            marker=dict(size=8, color=plot_df["color"], line=dict(width=1, color="white")),
            customdata=np.column_stack([plot_df["label"], plot_df["value"]]),
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>LST = %{{customdata[1]:.2f}} {unit_label}<extra></extra>",
        ))
    else:  # yearly
        hover_texts = [
            f"<b>{r['label']}</b><br>Avg LST = {r['value']:.2f} {unit_label}<br>"
            f"Warmest Month = {MONTH_NAMES.get(int(r['peak_month_num']), '—')}"
            for _, r in plot_df.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=plot_df["period"], y=plot_df["value"], mode="lines+markers",
            name="Annual Avg LST",
            line=dict(color="#ddd", width=1.5),
            marker=dict(size=13, color=plot_df["color"], line=dict(width=1.5, color="white")),
            text=hover_texts, hovertemplate="%{text}<extra></extra>",
        ))

    # ── Trendline ──
    if len(plot_df) >= 3:
        x_ord = plot_df["period"].map(pd.Timestamp.toordinal).values.astype(float)
        y_val = plot_df["value"].values.astype(float)
        mask  = ~np.isnan(y_val)
        if mask.sum() >= 2:
            coeffs           = np.polyfit(x_ord[mask], y_val[mask], 1)
            trend_y          = np.polyval(coeffs, x_ord)
            slope_per_decade = coeffs[0] * 365.25 * 10
            direction        = "↑" if slope_per_decade > 0 else "↓"
            trend_label      = f"Trend  {direction} {abs(slope_per_decade):.2f} {unit_label}/decade"
            fig.add_trace(go.Scatter(
                x=plot_df["period"], y=trend_y,
                mode="lines", name=trend_label,
                line=dict(color=LST_TREND_COLOR, width=2, dash="dot"),
                hoverinfo="skip",
            ))

    if not plot_df.empty:
        add_avg_line(fig, avg_val, unit_label)

    fig.update_layout(**base_layout(f"Land Surface Temperature (LST) — {selected_range}", x_title, y_title))
    return fig


# ── HI chart data ──────────────────────────────────────────────────────────────
HI_LINE_COLOR  = "#FF7043"
HI_FILL_COLOR  = "rgba(255,112,67,0.08)"
HI_TREND_COLOR = "#9B59B6"


def build_hi_plot(mode: str) -> tuple[pd.DataFrame, str, str]:
    hi_col  = "hi_f" if use_fahrenheit else "hi_c"
    y_title = f"Heat Index ({unit_label})"

    if mode == "daily":
        df = f_hi[["date", hi_col]].rename(columns={"date": "period", hi_col: "value"})
        df["label"] = df["period"].dt.strftime("%d %b %Y")
        df["color"] = HI_LINE_COLOR
        return df, "Date", y_title

    if mode == "weekly":
        d = f_hi.copy()
        d["week_start"] = d["date"].dt.to_period("W").apply(lambda p: p.start_time)
        w = d.groupby("week_start")[hi_col].mean().reset_index()
        w.columns = ["period", "value"]
        w["label"] = w["period"].dt.strftime("Week of %d %b %Y")
        w["color"] = HI_LINE_COLOR
        return w, "Week", y_title

    if mode == "monthly":
        d = f_hi.copy()
        m = d.groupby(["year", "month"])[hi_col].max().reset_index()
        m.columns = ["year", "month", "value"]
        m["period"] = pd.to_datetime(m["year"].astype(str) + "-" + m["month"].astype(str).str.zfill(2) + "-01")
        m["label"]  = m["period"].dt.strftime("%b %Y")
        m["color"]  = m["month"].map(MONTH_COLORS)
        return m[["period", "value", "label", "color"]], "Month", y_title

    # yearly — peak HI of the year, colored by month where it occurred
    d = f_hi.copy()
    idx_max = d.groupby("year")[hi_col].idxmax()
    yearly  = d.loc[idx_max, ["year", "month", hi_col]].reset_index(drop=True)
    yearly.columns = ["year", "peak_month_num", "value"]
    yearly["period"] = pd.to_datetime(yearly["year"].astype(str) + "-01-01")
    yearly["label"]  = yearly["year"].astype(str)
    yearly["color"]  = yearly["peak_month_num"].map(MONTH_COLORS)
    return yearly[["period", "value", "label", "color", "peak_month_num"]], "Year", y_title


def make_hi_fig(mode: str) -> go.Figure:
    plot_df, x_title, y_title = build_hi_plot(mode)
    avg_val = plot_df["value"].mean() if not plot_df.empty else 0
    fig = go.Figure()

    if mode == "daily":
        fig.add_trace(go.Scatter(
            x=plot_df["period"], y=plot_df["value"], mode="lines",
            name="Daily HI",
            line=dict(color=HI_LINE_COLOR, width=1.2),
            fill="tozeroy", fillcolor=HI_FILL_COLOR,
            customdata=np.column_stack([plot_df["label"], plot_df["value"]]),
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>Heat Index = %{{customdata[1]:.1f}} {unit_label}<extra></extra>",
        ))

    elif mode == "weekly":
        fig.add_trace(go.Scatter(
            x=plot_df["period"], y=plot_df["value"], mode="lines+markers",
            name="Weekly HI",
            line=dict(color=HI_LINE_COLOR, width=1.5),
            marker=dict(size=4, color=HI_LINE_COLOR),
            fill="tozeroy", fillcolor=HI_FILL_COLOR,
            customdata=np.column_stack([plot_df["label"], plot_df["value"]]),
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>Heat Index = %{{customdata[1]:.1f}} {unit_label}<extra></extra>",
        ))

    elif mode == "monthly":
        fig.add_trace(go.Scatter(
            x=plot_df["period"], y=plot_df["value"], mode="lines+markers",
            name="Monthly Peak HI",
            line=dict(color="#ddd", width=1.5),
            marker=dict(size=8, color=plot_df["color"], line=dict(width=1, color="white")),
            customdata=np.column_stack([plot_df["label"], plot_df["value"]]),
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>Heat Index = %{{customdata[1]:.1f}} {unit_label}<extra></extra>",
        ))

    else:  # yearly
        hover_texts = [
            f"<b>{r['label']}</b><br>Peak HI = {r['value']:.1f} {unit_label}<br>"
            f"Month = {MONTH_NAMES.get(int(r['peak_month_num']), '—')}"
            for _, r in plot_df.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=plot_df["period"], y=plot_df["value"], mode="lines+markers",
            name="Annual Peak HI",
            line=dict(color="#ddd", width=1.5),
            marker=dict(size=13, color=plot_df["color"], line=dict(width=1.5, color="white")),
            text=hover_texts, hovertemplate="%{text}<extra></extra>",
        ))

    # ── Trendline ──
    if len(plot_df) >= 3:
        x_ord = plot_df["period"].map(pd.Timestamp.toordinal).values.astype(float)
        y_val = plot_df["value"].values.astype(float)
        mask  = ~np.isnan(y_val)
        if mask.sum() >= 2:
            coeffs           = np.polyfit(x_ord[mask], y_val[mask], 1)
            trend_y          = np.polyval(coeffs, x_ord)
            slope_per_decade = coeffs[0] * 365.25 * 10
            direction        = "↑" if slope_per_decade > 0 else "↓"
            trend_label      = f"Trend  {direction} {abs(slope_per_decade):.2f} {unit_label}/decade"
            fig.add_trace(go.Scatter(
                x=plot_df["period"], y=trend_y,
                mode="lines", name=trend_label,
                line=dict(color=HI_TREND_COLOR, width=2, dash="dot"),
                hoverinfo="skip",
            ))

    if not plot_df.empty:
        add_avg_line(fig, avg_val, unit_label)

    fig.update_layout(**base_layout(f"Heat Index (HI) — {selected_range}", x_title, y_title))
    return fig


# ── Page header ────────────────────────────────────────────────────────────────
st.title("Historical Trends")
st.markdown(
    "<p style='color:#777;font-size:0.92rem;margin-top:-0.6rem'>"
    "Displays long-term historical climate behaviour for the selected location and hazard."
    "</p>",
    unsafe_allow_html=True,
)

info1, info2, info3 = st.columns([2, 2, 3])
info1.markdown("📍 **Location**  \nRaghwa (27.3954°N, 70.4780°E)")
info2.markdown("🌡️ **Hazard**  \nHeatwave Days · Max Temperature · LST · Heat Index")
info3.markdown(
    f"📅 **Data Range**  \n"
    f"ERA5: {DATA_START.strftime('%b %Y')} – {DATA_END.strftime('%b %Y')}  ·  "
    f"LST: {LST_DATA_START.strftime('%b %Y')} – {LST_DATA_END.strftime('%b %Y')}  ·  "
    f"HI: {HI_DATA_START.strftime('%b %Y')} – {HI_DATA_END.strftime('%b %Y')}"
)

st.markdown("---")

# ── LST summary values ─────────────────────────────────────────────────────────
lst_col_sel = "lst_f" if use_fahrenheit else "lst_c"

# ── Compute summary values ─────────────────────────────────────────────────────
total_hw_days   = int(f_hw_daily["is_heatwave"].sum())
years_in_window = max((range_end - range_start).days / 365.25, 1.0)
avg_hw_per_year = round(total_hw_days / years_in_window, 1)

peak_month_val   = (
    f_monthly_hw.loc[f_monthly_hw["heatwave_days"].idxmax(), "month"]
    if not f_monthly_hw.empty else None
)
peak_month_label = MONTH_NAMES.get(peak_month_val, "—") if peak_month_val else "—"

max_temp_window = f_mt_daily[temp_col].max()  if not f_mt_daily.empty else float("nan")
avg_temp_window = f_mt_daily[temp_col].mean() if not f_mt_daily.empty else float("nan")
min_temp_window = f_mt_daily[temp_col].min()  if not f_mt_daily.empty else float("nan")

# Peak heatwave month in max-temp dataset (month with highest weekly tmax on average)
if not f_mt_daily.empty:
    mt_monthly_avg = (
        f_mt_daily.assign(month=f_mt_daily["date"].dt.month)
        .groupby("month")[temp_col].mean()
    )
    peak_mt_month_label = MONTH_NAMES.get(int(mt_monthly_avg.idxmax()), "—")
else:
    peak_mt_month_label = "—"

# LST summary
avg_lst_window = f_lst[lst_col_sel].mean() if not f_lst.empty else float("nan")
max_lst_window = f_lst[lst_col_sel].max()  if not f_lst.empty else float("nan")

if not f_lst.empty:
    lst_monthly_avg = f_lst.groupby("month")[lst_col_sel].mean()
    peak_lst_month_label = MONTH_NAMES.get(int(lst_monthly_avg.idxmax()), "—")
else:
    peak_lst_month_label = "—"

# HI summary
hi_col_sel = "hi_f" if use_fahrenheit else "hi_c"
max_hi_window = f_hi[hi_col_sel].max()  if not f_hi.empty else float("nan")
avg_hi_window = f_hi[hi_col_sel].mean() if not f_hi.empty else float("nan")

if not f_hi.empty:
    hi_monthly_avg = f_hi.groupby("month")[hi_col_sel].mean()
    peak_hi_month_label = MONTH_NAMES.get(int(hi_monthly_avg.idxmax()), "—")
else:
    peak_hi_month_label = "—"

# ── Side-by-side charts ────────────────────────────────────────────────────────
col_hw, col_mt = st.columns(2, gap="medium")

with col_hw:
    st.markdown(
        "<p class='chart-label'>ERA5-LAND  ·  Heatwave Days (Monthly)</p>",
        unsafe_allow_html=True,
    )
    hw_fig = make_hw_fig(agg_mode)
    st.plotly_chart(hw_fig, width="stretch")

with col_mt:
    st.markdown(
        "<p class='chart-label'>ERA5-LAND  ·  Maximum Temperature (Weekly)</p>",
        unsafe_allow_html=True,
    )
    mt_fig = make_mt_fig(agg_mode)
    st.plotly_chart(mt_fig, width="stretch")

# ── Row 2: LST + HI side by side ──────────────────────────────────────────────
col_lst, col_hi = st.columns(2, gap="medium")

with col_lst:
    st.markdown(
        "<p class='chart-label'>Landsat/Sentinel  ·  Land Surface Temperature (Monthly)</p>",
        unsafe_allow_html=True,
    )
    lst_fig = make_lst_fig(agg_mode)
    st.plotly_chart(lst_fig, width="stretch")

with col_hi:
    st.markdown(
        "<p class='chart-label'>ERA5-LAND  ·  Heat Index (Weekly)</p>",
        unsafe_allow_html=True,
    )
    hi_fig = make_hi_fig(agg_mode)
    st.plotly_chart(hi_fig, width="stretch")

st.markdown("---")

# ── Month colour legend (yearly / monthly views) ───────────────────────────────
if agg_mode in ("yearly", "monthly"):
    st.markdown(
        "<p class='section-title' style='margin-bottom:0.5rem'>"
        "Marker Colour → Month of Peak Activity</p>",
        unsafe_allow_html=True,
    )
    leg_cols = st.columns(12)
    for i, (m_num, m_name) in enumerate(MONTH_NAMES.items()):
        with leg_cols[i]:
            st.markdown(
                f"<div style='background:{MONTH_COLORS[m_num]};border-radius:5px;"
                f"padding:5px 0;text-align:center;color:white;font-size:11px;font-weight:700'>"
                f"{m_name}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("")

# ── Four-panel metric layout ───────────────────────────────────────────────────
panel_hw, div1, panel_mt, div2, panel_lst, div3, panel_hi = st.columns([9, 1, 9, 1, 9, 1, 9])

with panel_hw:
    st.markdown(
        "<p style='font-size:0.78rem;font-weight:700;color:#E15759;"
        "letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.6rem'>"
        "🔥 Heatwave Days (Monthly)</p>",
        unsafe_allow_html=True,
    )
    h1, h2, h3 = st.columns(3)
    with h1:
        st.metric("Total Heatwave Days", f"{total_hw_days:,}")
    with h2:
        st.metric("Avg HW Days / Year", f"{avg_hw_per_year}")
    with h3:
        st.metric("Peak Month", peak_month_label)

with div1:
    st.markdown(
        "<div style='border-left:1px solid #e0e0e0;height:100px;margin:0 auto;width:1px'></div>",
        unsafe_allow_html=True,
    )

with panel_mt:
    st.markdown(
        "<p style='font-size:0.78rem;font-weight:700;color:#3A6EA5;"
        "letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.6rem'>"
        f"🌡️ Maximum Temperature — Weekly ({unit_label})</p>",
        unsafe_allow_html=True,
    )
    t1, t2, t3 = st.columns(3)
    with t1:
        st.metric(f"Highest Tmax", f"{max_temp_window:.1f} {unit_label}")
    with t2:
        st.metric(f"Avg Daily Tmax", f"{avg_temp_window:.1f} {unit_label}")
    with t3:
        st.metric("Peak Month (avg)", peak_mt_month_label)

with div2:
    st.markdown(
        "<div style='border-left:1px solid #e0e0e0;height:100px;margin:0 auto;width:1px'></div>",
        unsafe_allow_html=True,
    )

with panel_lst:
    st.markdown(
        "<p style='font-size:0.78rem;font-weight:700;color:#2E8B57;"
        "letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.6rem'>"
        f"🌍 LST — Monthly ({unit_label})</p>",
        unsafe_allow_html=True,
    )
    l1, l2, l3 = st.columns(3)
    with l1:
        st.metric("Highest LST", f"{max_lst_window:.1f} {unit_label}" if not pd.isna(max_lst_window) else "—")
    with l2:
        st.metric("Avg Monthly LST", f"{avg_lst_window:.1f} {unit_label}" if not pd.isna(avg_lst_window) else "—")
    with l3:
        st.metric("Peak Month (avg)", peak_lst_month_label)

with div3:
    st.markdown(
        "<div style='border-left:1px solid #e0e0e0;height:100px;margin:0 auto;width:1px'></div>",
        unsafe_allow_html=True,
    )

with panel_hi:
    st.markdown(
        "<p style='font-size:0.78rem;font-weight:700;color:#FF7043;"
        "letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.6rem'>"
        f"🌬️ Heat Index — Weekly ({unit_label})</p>",
        unsafe_allow_html=True,
    )
    i1, i2, i3 = st.columns(3)
    with i1:
        st.metric("Peak HI", f"{max_hi_window:.1f} {unit_label}" if not pd.isna(max_hi_window) else "—")
    with i2:
        st.metric("Avg Weekly HI", f"{avg_hi_window:.1f} {unit_label}" if not pd.isna(avg_hi_window) else "—")
    with i3:
        st.metric("Peak Month (avg)", peak_hi_month_label)

# ── Expanders ─────────────────────────────────────────────────────────────────
rule_col_hw, rule_col_mt, rule_col_lst, rule_col_hi = st.columns(4, gap="medium")

with rule_col_hw:
    with st.expander("Heatwave Detection Rules"):
        spell_thr  = f"{THRESHOLD_SPELL_C:.0f} °C"  if not use_fahrenheit else f"{THRESHOLD_SPELL_C * 9/5 + 32:.0f} °F"
        severe_thr = f"{THRESHOLD_SEVERE_C:.0f} °C" if not use_fahrenheit else f"{THRESHOLD_SEVERE_C * 9/5 + 32:.0f} °F"
        st.markdown(
            f"""
**Rule 1 — Consecutive Spell**
A sequence of **2 or more consecutive days** where Tmax ≥ {spell_thr}. All days in the spell count.

**Rule 2 — Severe Day**
Any single day where Tmax ≥ {severe_thr} always counts, even if isolated.
            """
        )

with rule_col_mt:
    with st.expander("Maximum Temperature — Methodology"):
        st.markdown(
            f"""
**Data Source**
ERA5-LAND daily maximum temperature (T2M_MAX) at the closest grid point to the selected location.

**Weekly Aggregation**
Daily Tmax values are grouped by ISO calendar week. The **maximum** value within each week is plotted as a single data point.

**Trendline**
A **linear regression** (least-squares best-fit line) is computed over all visible data points in the selected window. The slope is expressed as **{unit_label} per decade** and shown in the legend.

**Average Line**
A flat horizontal line at the mean of all visible weekly Tmax values in the selected window. Recalculates whenever the time range or unit changes.
            """
        )

with rule_col_lst:
    with st.expander("LST — Methodology"):
        st.markdown(
            f"""
**Data Source**
Landsat/Sentinel Series via Google Earth Engine (GEE). Monthly average LST values are extracted as the spatial mean of all grid cells within the AOI.

**Temporal Resolution**
Data is **monthly** — the finest granularity available. For shorter time windows (Last 1 Month to Last 5 Years) individual monthly values are plotted. For longer windows (Last 10 / 20 Years) values are aggregated to **annual averages**.

**Trendline**
A **linear regression** (least-squares best-fit line) over all visible monthly or yearly LST values. Slope is expressed as **{unit_label} per decade** and shown in the legend.

**Average Line**
Mean of all visible LST values in the selected window. Recalculates whenever the time range or unit changes.

**Marker Colour**
In monthly view each marker is coloured by its calendar month. In yearly view each marker is coloured by the warmest month of that year.
            """
        )

with rule_col_hi:
    with st.expander("Heat Index — Methodology"):
        st.markdown(
            f"""
**Data Source**
ERA5-LAND hourly temperature and humidity data aggregated to **weekly** averages. Heat Index is computed from the weekly average temperature and relative humidity using the Rothfusz regression formula.

**Weekly Aggregation**
Hourly T and RH data are averaged over each ISO calendar week. The resulting weekly HI value is plotted as a single data point.

**Trendline**
A **linear regression** (least-squares best-fit line) over all visible weekly, monthly, or yearly HI values. Slope is expressed as **{unit_label} per decade** and shown in the legend.

**Average Line**
A flat horizontal line at the mean of all visible HI values in the selected window. Recalculates whenever the time range or unit changes.

**Marker Colour**
In monthly / yearly views each marker is coloured by calendar month of peak HI.
            """
        )

exp1, exp2, exp3, exp4 = st.columns(4)
with exp1:
    with st.expander("View Heatwave Data"):
        hw_plot_df, _, _ = build_hw_plot(agg_mode)
        d = hw_plot_df[["label", "value"]].rename(columns={"label": "Period", "value": "Heatwave Days"})
        st.dataframe(d, width="stretch", hide_index=True)

with exp2:
    with st.expander("View Max Temperature Data"):
        mt_plot_df, _, _ = build_mt_plot(agg_mode)
        d = mt_plot_df[["label", "value"]].rename(columns={"label": "Period", "value": f"Max Temp ({unit_label})"})
        st.dataframe(d, width="stretch", hide_index=True)

with exp3:
    with st.expander("View LST Data"):
        lst_plot_df, _, _ = build_lst_plot(agg_mode)
        d = lst_plot_df[["label", "value"]].rename(columns={"label": "Period", "value": f"LST ({unit_label})"})
        st.dataframe(d, width="stretch", hide_index=True)

with exp4:
    with st.expander("View Heat Index Data"):
        hi_plot_df, _, _ = build_hi_plot(agg_mode)
        d = hi_plot_df[["label", "value"]].rename(columns={"label": "Period", "value": f"Heat Index ({unit_label})"})
        st.dataframe(d, width="stretch", hide_index=True)
