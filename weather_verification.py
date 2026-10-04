"""
Weather Model Verification & Dashboard
Builds end-to-end verification stats + interactive HTML dashboard for
U.S. gas-weighted degree day (GWDD) forecasts from GFS, ECE, GEFS.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parent
FORECAST_FILES = {
    "GFS":  WORKSPACE / "GFS_GWUS.csv",
    "ECE":  WORKSPACE / "ECE_GWUS.csv",
    "GEFS": WORKSPACE / "GEFS_GWUS.csv",
}
HISTORY_FILE = WORKSPACE / "GWUS_history.csv"
KDCA_FILE    = WORKSPACE / "ECE_KDCA_t2m_20200101_20251231.csv"
CSV_OUT      = WORKSPACE / "model_verification_results.csv"
KDCA_CSV_OUT = WORKSPACE / "kdca_extreme_events.csv"
HTML_OUT     = WORKSPACE / "weather_dashboard.html"

WINDOWS = ["1-5d", "6-10d", "11-15d"]
MODELS  = ["GFS", "ECE", "GEFS"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
REGIMES = ["Cold", "Normal", "Warm"]


# ---------- Section 1: load & normalize ----------

def season_of(month: int) -> str:
    if month in (12, 1, 2):  return "DJF"
    if month in (3, 4, 5):   return "MAM"
    if month in (6, 7, 8):   return "JJA"
    return "SON"


def assign_window(lead: int) -> str:
    if 1  <= lead <= 5:  return "1-5d"
    if 6  <= lead <= 10: return "6-10d"
    if 11 <= lead <= 15: return "11-15d"
    return ""


def half_year(month: int) -> str:
    # User-defined: Summer = Apr-Oct (4..10), Winter = Nov-Mar (11,12,1,2,3)
    return "Summer" if 4 <= month <= 10 else "Winter"


# Magnitude bins for actual GWDD — chosen to give roughly comparable populations
MAG_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 999)]
MAG_LABELS = ["<5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"]
def assign_mag(v: float) -> str:
    for (lo, hi), lab in zip(MAG_BINS, MAG_LABELS):
        if lo <= v < hi:
            return lab
    return MAG_LABELS[-1]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for model, path in FORECAST_FILES.items():
        df = pd.read_csv(path)
        df["ForecastDate"] = pd.to_datetime(df["ForecastDate"], errors="coerce")
        df["RunDate"]      = pd.to_datetime(df["RunDate"], errors="coerce")
        df = df.dropna(subset=["ForecastDate", "RunDate"])
        df["Model"] = model
        frames.append(df)
    forecasts = pd.concat(frames, ignore_index=True)

    history = pd.read_csv(HISTORY_FILE)
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    history = history.dropna(subset=["Date"])
    history = history.rename(columns={"Date": "ForecastDate", "GWDD": "Actual"})

    forecasts["LeadDays"] = (forecasts["ForecastDate"] - forecasts["RunDate"]).dt.days
    # Keep an all-leads frame (incl. lead 0) for sanity checks & extremes
    all_leads = forecasts[forecasts["LeadDays"].between(0, 15)].copy()
    all_leads = all_leads.merge(history, on="ForecastDate", how="inner")
    all_leads["Error"]  = all_leads["GWDD"] - all_leads["Actual"]
    all_leads["Month"]  = all_leads["ForecastDate"].dt.month
    all_leads["Year"]   = all_leads["ForecastDate"].dt.year
    all_leads["Season"] = all_leads["Month"].map(season_of)
    all_leads["HalfYear"] = all_leads["Month"].map(half_year)

    # Standard verification set: leads 1-15 with Window assignment
    merged = all_leads[all_leads["LeadDays"].between(1, 15)].copy()
    merged["Window"] = merged["LeadDays"].map(assign_window)
    return merged, history, all_leads


# ---------- Section 2: verification stats ----------

def safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_stats(df: pd.DataFrame) -> dict:
    err = df["Error"]
    abs_err = err.abs()
    chg1 = df["Chg1"].dropna()
    return {
        "N":          int(len(df)),
        "MAE":        float(abs_err.mean())          if len(df) else np.nan,
        "RMSE":       float(np.sqrt((err**2).mean())) if len(df) else np.nan,
        "Bias":       float(err.mean())              if len(df) else np.nan,
        "Corr":       safe_corr(df["GWDD"], df["Actual"]),
        "MeanAbsRev": float(chg1.abs().mean())       if len(chg1) else np.nan,
        "HitRate1":   float((abs_err <= 1.0).mean() * 100) if len(df) else np.nan,
        "HitRate2":   float((abs_err <= 2.0).mean() * 100) if len(df) else np.nan,
    }


def group_stats(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for vals, g in df.groupby(keys, sort=False):
        if not isinstance(vals, tuple):
            vals = (vals,)
        row = dict(zip(keys, vals))
        row.update(compute_stats(g))
        rows.append(row)
    return pd.DataFrame(rows)


def build_verification(df: pd.DataFrame) -> pd.DataFrame:
    mw      = group_stats(df, ["Model", "Window"]).assign(GroupType="ModelWindow")
    mwm     = group_stats(df, ["Model", "Window", "Month"]).assign(GroupType="ModelWindowMonth")
    mws     = group_stats(df, ["Model", "Window", "Season"]).assign(GroupType="ModelWindowSeason")
    out     = pd.concat([mw, mwm, mws], ignore_index=True)
    cols    = ["GroupType", "Model", "Window", "Month", "Season",
               "N", "MAE", "RMSE", "Bias", "Corr",
               "MeanAbsRev", "HitRate1", "HitRate2"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


# ---------- Section 3: pattern analysis ----------

def pattern_analysis(df: pd.DataFrame, history: pd.DataFrame, all_leads: pd.DataFrame) -> dict:
    mw = group_stats(df, ["Model", "Window"])

    # 1. Rankings by RMSE per window
    rankings = {}
    for w in WINDOWS:
        sub = mw[mw["Window"] == w].sort_values("RMSE")
        rankings[w] = [
            {"Model": r["Model"], "RMSE": round(r["RMSE"], 3), "MAE": round(r["MAE"], 3)}
            for _, r in sub.iterrows()
        ]

    # 2. Seasonal bias
    mws = group_stats(df, ["Model", "Window", "Season"])
    mws["Flag"] = mws["Bias"].abs() > 1.0
    seasonal = mws[["Model", "Window", "Season", "Bias", "Flag", "N"]].copy()

    # 2b. Monthly bias (Model × Window × Month)
    mwm = group_stats(df, ["Model", "Window", "Month"])
    mwm["Flag"] = mwm["Bias"].abs() > 1.0
    monthly = mwm[["Model", "Window", "Month", "Bias", "Flag", "N"]].copy()
    monthly["Month"] = monthly["Month"].astype(int)

    # 3. Convergence: RMSE per lead day per model
    convergence_rows = []
    for (model, lead), g in df.groupby(["Model", "LeadDays"]):
        convergence_rows.append({
            "Model": model,
            "LeadDays": int(lead),
            "RMSE": float(np.sqrt((g["Error"]**2).mean())),
            "MAE":  float(g["Error"].abs().mean()),
            "N":    int(len(g)),
        })
    convergence = pd.DataFrame(convergence_rows).sort_values(["Model", "LeadDays"])

    # 4. Drift: mean signed Chg1/2/4 per model
    drift_rows = []
    for model, g in df.groupby("Model"):
        drift_rows.append({
            "Model": model,
            "MeanChg1": float(g["Chg1"].dropna().mean()),
            "MeanChg2": float(g["Chg2"].dropna().mean()),
            "MeanChg4": float(g["Chg4"].dropna().mean()),
            "AbsChg1":  float(g["Chg1"].dropna().abs().mean()),
            "AbsChg2":  float(g["Chg2"].dropna().abs().mean()),
            "AbsChg4":  float(g["Chg4"].dropna().abs().mean()),
        })
    drift = pd.DataFrame(drift_rows)

    # 5. Divergence: |GFS - ECE| spread, smoothed per window
    div_rows = []
    pivot = (df[df["Model"].isin(["GFS", "ECE"])]
             .groupby(["ForecastDate", "Window", "Model"])["GWDD"]
             .mean().unstack("Model"))
    if "GFS" in pivot.columns and "ECE" in pivot.columns:
        pivot["Spread"] = (pivot["GFS"] - pivot["ECE"]).abs()
        pivot = pivot.dropna(subset=["Spread"]).reset_index()
        for w in WINDOWS:
            sub = pivot[pivot["Window"] == w].sort_values("ForecastDate")
            sub = sub.set_index("ForecastDate")
            sub["Rolling"] = sub["Spread"].rolling("30D").mean()
            for dt, row in sub.iterrows():
                div_rows.append({
                    "ForecastDate": dt.strftime("%Y-%m-%d"),
                    "Window": w,
                    "Spread": float(row["Spread"]),
                    "Rolling": float(row["Rolling"]) if pd.notna(row["Rolling"]) else None,
                })
    divergence = pd.DataFrame(div_rows)

    # 6. Regime performance — terciles from history actuals
    q33, q67 = history["Actual"].quantile([1/3, 2/3]).values
    def regime_of(v):
        if v <= q33: return "Cold"
        if v <= q67: return "Normal"
        return "Warm"
    df_r = df.copy()
    df_r["Regime"] = df_r["Actual"].map(regime_of)
    regime = group_stats(df_r, ["Model", "Window", "Regime"])
    # Normalized error: MAE as % of mean actual in that regime
    mean_actual = (df_r.groupby(["Model", "Window", "Regime"])["Actual"]
                       .mean().reset_index().rename(columns={"Actual": "MeanActual"}))
    regime = regime.merge(mean_actual, on=["Model", "Window", "Regime"])
    regime["MAEPct"] = 100.0 * regime["MAE"] / regime["MeanActual"]
    regime = regime[["Model", "Window", "Regime", "MAE", "MAEPct", "MeanActual", "RMSE", "Bias", "N"]]

    # 7. Extreme events — top/bottom 10% of history actuals
    p10, p90 = history["Actual"].quantile([0.10, 0.90]).values
    extremes = df[(df["Actual"] <= p10) | (df["Actual"] >= p90)].copy()
    ext_rows = []
    for (model, window), g in extremes.groupby(["Model", "Window"]):
        ext_rows.append({
            "Model": model,
            "Window": window,
            "MAE":  float(g["Error"].abs().mean()),
            "RMSE": float(np.sqrt((g["Error"]**2).mean())),
            "Bias": float(g["Error"].mean()),
            "N":    int(len(g)),
        })
    extremes_df = pd.DataFrame(ext_rows)

    # 8. Per-lead breakout (incl. lead 0 and lead 15 explicitly) — sanity-check view
    per_lead_rows = []
    for (model, lead), g in all_leads.groupby(["Model", "LeadDays"]):
        per_lead_rows.append({
            "Model": model,
            "LeadDays": int(lead),
            "N": int(len(g)),
            "MAE":  float(g["Error"].abs().mean()),
            "RMSE": float(np.sqrt((g["Error"]**2).mean())),
            "Bias": float(g["Error"].mean()),
            "Corr": safe_corr(g["GWDD"], g["Actual"]),
            "HitRate1": float((g["Error"].abs() <= 1.0).mean() * 100),
            "HitRate2": float((g["Error"].abs() <= 2.0).mean() * 100),
        })
    per_lead = pd.DataFrame(per_lead_rows).sort_values(["Model", "LeadDays"])

    # 9. Winter/Summer (half-year) bias per Model × Window
    halfyear = group_stats(df, ["Model", "Window", "HalfYear"])
    halfyear["Flag"] = halfyear["Bias"].abs() > 1.0
    halfyear = halfyear[["Model", "Window", "HalfYear", "Bias", "MAE", "RMSE", "N", "Flag"]]

    # 10. Bias by magnitude — bin actuals into GWDD magnitude buckets
    df_m = df.copy()
    df_m["MagBin"] = df_m["Actual"].map(assign_mag)
    mag_rows = []
    for (model, window, mag), g in df_m.groupby(["Model", "Window", "MagBin"]):
        mag_rows.append({
            "Model": model, "Window": window, "MagBin": mag,
            "N": int(len(g)),
            "MAE":  float(g["Error"].abs().mean()),
            "Bias": float(g["Error"].mean()),
            "MeanActual": float(g["Actual"].mean()),
            "MAEPct": float(100.0 * g["Error"].abs().mean() / g["Actual"].mean()) if g["Actual"].mean() > 0 else float("nan"),
        })
    mag_table = pd.DataFrame(mag_rows)

    # 11. Standardized anomaly — z-score Anomaly by month-of-year std across all rows
    a = all_leads.dropna(subset=["Anomaly"]).copy()
    a["MonthOfYear"] = a["ForecastDate"].dt.month
    month_std = a.groupby("MonthOfYear")["Anomaly"].std().rename("MonthStd").reset_index()
    a = a.merge(month_std, on="MonthOfYear")
    a["AnomalyZ"] = a["Anomaly"] / a["MonthStd"]
    a["AbsZ"] = a["AnomalyZ"].abs()
    extreme_forecasts = (a.sort_values("AbsZ", ascending=False)
                          .head(50)
                          [["Model","RunDate","ForecastDate","LeadDays","GWDD","Anomaly","AnomalyZ","Actual","Error"]]
                          .copy())

    # 12. Most extreme actuals (top/bottom from history)
    h_sorted = history.sort_values("Actual")
    extreme_actuals = pd.concat([
        h_sorted.head(15).assign(Side="Lowest"),
        h_sorted.tail(15).assign(Side="Highest").iloc[::-1],
    ])

    # 13. Biggest misses per lead day 10-15, per model — granular row dump
    miss_rows = []
    for lead in [10, 11, 12, 13, 14, 15]:
        for model in MODELS:
            g = all_leads[(all_leads["LeadDays"] == lead) & (all_leads["Model"] == model)].copy()
            if g.empty: continue
            g["AbsErr"] = g["Error"].abs()
            top = g.nlargest(10, "AbsErr")
            for _, r in top.iterrows():
                miss_rows.append({
                    "Model": model,
                    "LeadDays": int(lead),
                    "RunDate": r["RunDate"].strftime("%Y-%m-%d"),
                    "ForecastDate": r["ForecastDate"].strftime("%Y-%m-%d"),
                    "Forecast": float(r["GWDD"]),
                    "Actual":   float(r["Actual"]),
                    "Error":    float(r["Error"]),
                    "Anomaly":  float(r["Anomaly"]) if pd.notna(r["Anomaly"]) else None,
                })
    biggest_misses = pd.DataFrame(miss_rows)

    return {
        "rankings":    rankings,
        "seasonal":    seasonal,
        "monthly":     monthly,
        "convergence": convergence,
        "drift":       drift,
        "divergence":  divergence,
        "regime":      regime,
        "extremes":    extremes_df,
        "tercile_q33": float(q33),
        "tercile_q67": float(q67),
        "extreme_p10": float(p10),
        "extreme_p90": float(p90),
        "mw_summary":  mw,
        "per_lead":    per_lead,
        "halfyear":    halfyear,
        "magnitude":   mag_table,
        "extreme_forecasts": extreme_forecasts,
        "extreme_actuals":   extreme_actuals,
        "biggest_misses":    biggest_misses,
    }


# ---------- Section 4: insights text ----------

def build_insights(mw: pd.DataFrame, patterns: dict) -> list[str]:
    lines = []

    # Best model per window
    for w in WINDOWS:
        sub = mw[mw["Window"] == w].sort_values("RMSE")
        if not sub.empty:
            best = sub.iloc[0]
            worst = sub.iloc[-1]
            lines.append(
                f"At the {w} horizon, <b>{best['Model']}</b> is the most accurate "
                f"(RMSE {best['RMSE']:.2f}, MAE {best['MAE']:.2f}), "
                f"while {worst['Model']} trails (RMSE {worst['RMSE']:.2f})."
            )

    # Largest bias
    biggest = mw.loc[mw["Bias"].abs().idxmax()]
    direction = "high (forecasts cooler than reality)" if biggest["Bias"] > 0 else "low (forecasts warmer than reality)"
    lines.append(
        f"The largest systematic bias is <b>{biggest['Model']}</b> at {biggest['Window']}: "
        f"{biggest['Bias']:+.2f} GWDD on average — runs {direction}."
    )

    # Seasonal flags
    sflags = patterns["seasonal"][patterns["seasonal"]["Flag"]]
    if not sflags.empty:
        bits = [f"{r['Model']}/{r['Window']}/{r['Season']} ({r['Bias']:+.2f})"
                for _, r in sflags.iterrows()]
        lines.append("Seasonal bias warnings (|bias| &gt; 1 GWDD): " + ", ".join(bits) + ".")
    else:
        lines.append("No model exceeds ±1 GWDD seasonal bias in any window — no major seasonal red flags.")

    # Volatility (revision)
    drift = patterns["drift"].set_index("Model")
    most_vol  = drift["AbsChg1"].idxmax()
    least_vol = drift["AbsChg1"].idxmin()
    lines.append(
        f"<b>{most_vol}</b> is the most volatile model run-to-run "
        f"(mean |Chg1| = {drift.loc[most_vol, 'AbsChg1']:.2f}); "
        f"<b>{least_vol}</b> is the most stable ({drift.loc[least_vol, 'AbsChg1']:.2f})."
    )

    # Drift direction
    drift_lines = []
    for m in MODELS:
        if m in drift.index:
            d = drift.loc[m, "MeanChg1"]
            arrow = "trends higher" if d > 0.02 else ("trends lower" if d < -0.02 else "no consistent drift")
            drift_lines.append(f"{m} {arrow} ({d:+.3f})")
    lines.append("Run-to-run drift (mean signed Chg1): " + "; ".join(drift_lines) + ".")

    # Degradation from 1-5d to 11-15d
    deg_lines = []
    for m in MODELS:
        sub = mw[mw["Model"] == m]
        if sub["Window"].isin(["1-5d", "11-15d"]).all():
            r1  = sub[sub["Window"] == "1-5d"]["RMSE"].iloc[0]
            r15 = sub[sub["Window"] == "11-15d"]["RMSE"].iloc[0]
            deg_lines.append(f"{m} +{r15 - r1:.2f}")
    lines.append("RMSE degradation from 1-5d to 11-15d: " + ", ".join(deg_lines) + ".")

    # Regime weakness
    reg = patterns["regime"]
    weak = reg.loc[reg["MAE"].idxmax()]
    lines.append(
        f"The hardest-to-predict combo is <b>{weak['Model']}</b> at {weak['Window']} during "
        f"<b>{weak['Regime']}</b> regimes (MAE {weak['MAE']:.2f})."
    )

    # Extremes
    ext = patterns["extremes"]
    if not ext.empty:
        worst_ext = ext.loc[ext["MAE"].idxmax()]
        best_ext  = ext.loc[ext["MAE"].idxmin()]
        lines.append(
            f"On extreme-temperature days (top/bottom 10% of actuals), "
            f"<b>{best_ext['Model']}</b> at {best_ext['Window']} performs best (MAE {best_ext['MAE']:.2f}); "
            f"{worst_ext['Model']} at {worst_ext['Window']} performs worst ({worst_ext['MAE']:.2f})."
        )

    return lines


# ---------- Section 5: build forecast-vs-actual series ----------

def build_forecast_series(df: pd.DataFrame) -> dict:
    out = {}
    for m in MODELS:
        out[m] = {}
        for w in WINDOWS:
            sub = df[(df["Model"] == m) & (df["Window"] == w)]
            if sub.empty:
                out[m][w] = []
                continue
            agg = (sub.groupby("ForecastDate")
                       .agg(Forecast=("GWDD", "mean"),
                            Actual=("Actual", "first"))
                       .reset_index()
                       .sort_values("ForecastDate"))
            out[m][w] = [
                {"date": r["ForecastDate"].strftime("%Y-%m-%d"),
                 "forecast": round(float(r["Forecast"]), 3),
                 "actual":   round(float(r["Actual"]),   3)}
                for _, r in agg.iterrows()
            ]
    return out


# ---------- Section 6: HTML ----------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GWDD Model Verification Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
:root {
  --bg: #0d1117;
  --panel: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --muted: #8b949e;
  --gfs: #58a6ff;
  --ece: #f0883e;
  --gefs: #3fb950;
  --good: #3fb950;
  --bad:  #f85149;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
header {
  padding: 20px 28px;
  border-bottom: 1px solid var(--border);
}
header h1 { margin: 0; font-size: 20px; }
header .sub { color: var(--muted); margin-top: 4px; font-size: 13px; }
nav {
  display: flex;
  flex-wrap: wrap;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
}
nav button {
  background: transparent;
  border: 0;
  color: var(--muted);
  padding: 12px 18px;
  cursor: pointer;
  font: inherit;
  border-bottom: 2px solid transparent;
}
nav button:hover { color: var(--text); }
nav button.active {
  color: var(--text);
  border-bottom-color: var(--gfs);
}
main { padding: 24px 28px; max-width: 1280px; margin: 0 auto; }
section { display: none; }
section.active { display: block; }
section h2 { margin-top: 0; font-size: 18px; }
.note { color: var(--muted); margin: 0 0 16px; font-size: 13px; }
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 18px;
  margin-bottom: 18px;
}
.controls { display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
.controls label { color: var(--muted); display: flex; align-items: center; gap: 6px; font-size: 13px; }
.controls select, .controls input {
  background: #0d1117;
  color: var(--text);
  border: 1px solid var(--border);
  padding: 6px 10px;
  border-radius: 6px;
  font: inherit;
}
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; }
.cell { padding: 6px 10px; border-radius: 4px; display: inline-block; min-width: 60px; text-align: right; }
.chart-wrap { position: relative; height: 420px; }
ul.insights { padding-left: 20px; }
ul.insights li { margin-bottom: 10px; }
.flag-pill {
  display: inline-block;
  font-size: 11px;
  padding: 2px 8px;
  margin-left: 6px;
  background: rgba(248, 81, 73, 0.15);
  color: var(--bad);
  border: 1px solid rgba(248, 81, 73, 0.4);
  border-radius: 12px;
}
.legend-row { display: flex; gap: 18px; flex-wrap: wrap; margin: 0 0 12px; font-size: 12px; color: var(--muted); }
.legend-swatch { display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: middle; border-radius: 2px; }
</style>
</head>
<body>
<header>
  <h1>GWDD Model Verification Dashboard</h1>
  <div class="sub">U.S. Gas-Weighted Degree Day forecasts — GFS · ECE · GEFS · vs. observed history</div>
</header>
<nav id="tabnav"></nav>
<main>
  <section id="tab-scorecard" class="active">
    <h2>Model Scorecard</h2>
    <p class="note">Summary stats per model × lead-time window. Cells are tinted green (better than peers in the same window) to red (worse).</p>
    <div class="panel" id="scorecard-panel"></div>
  </section>

  <section id="tab-forecast">
    <h2>Forecast vs Actual</h2>
    <p class="note">Time series of mean forecast within the selected lead-time window, overlaid on observed GWDD with a ±1 GWDD band.</p>
    <div class="panel">
      <div class="controls">
        <label>Model: <select id="fa-model"></select></label>
        <label>Window: <select id="fa-window"></select></label>
        <label>From: <input type="date" id="fa-from"></label>
        <label>To: <input type="date" id="fa-to"></label>
      </div>
      <div class="chart-wrap"><canvas id="chart-fa"></canvas></div>
    </div>
  </section>

  <section id="tab-bias">
    <h2>Bias &amp; Seasonality</h2>
    <p class="note">Mean (forecast − actual). Positive = model forecasts higher GWDD than realized (cold bias on demand). Bars with |bias| &gt; 1 GWDD are red-bordered.</p>
    <div class="panel">
      <div class="controls">
        <label>Window: <select id="bias-window"></select></label>
        <label>View by:
          <select id="bias-grain">
            <option value="season">Season (DJF/MAM/JJA/SON)</option>
            <option value="month">Month (Jan–Dec)</option>
            <option value="halfyear">Half-year (Winter Nov–Mar / Summer Apr–Oct)</option>
          </select>
        </label>
      </div>
      <div class="chart-wrap"><canvas id="chart-bias"></canvas></div>
    </div>
  </section>

  <section id="tab-convergence">
    <h2>Model Convergence</h2>
    <p class="note">Average RMSE by lead day. Error typically grows as the lead time extends.</p>
    <div class="panel"><div class="chart-wrap"><canvas id="chart-conv"></canvas></div></div>
  </section>

  <section id="tab-revisions">
    <h2>Forecast Revisions</h2>
    <p class="note">Mean absolute run-over-run change. Higher = more volatile guidance; lower = more stable.</p>
    <div class="panel"><div class="chart-wrap"><canvas id="chart-rev"></canvas></div></div>
  </section>

  <section id="tab-divergence">
    <h2>Model Divergence (GFS vs ECE)</h2>
    <p class="note">Rolling 30-day average of |GFS − ECE| spread per ForecastDate, by window. Wider spread = more model disagreement.</p>
    <div class="panel"><div class="chart-wrap"><canvas id="chart-div"></canvas></div></div>
  </section>

  <section id="tab-regime">
    <h2>Regime Performance</h2>
    <p class="note"><b>Lower bars = better forecasts.</b> MAE expressed as a <b>percentage of the mean observed GWDD in that regime</b>, so cold/normal/warm groups are directly comparable despite their very different absolute magnitudes. Regimes are terciles of observed GWDD: Cold ≤ 9.12 (summer baseline), Normal 9.12–15.84 (shoulder), Warm > 15.84 (winter peak demand). The dashed line marks 15% — a rough "weak forecast" threshold; bars above it are unreliable.</p>
    <div class="panel">
      <div class="controls">
        <label>Window: <select id="regime-window"></select></label>
      </div>
      <div class="chart-wrap"><canvas id="chart-regime"></canvas></div>
    </div>
  </section>

  <section id="tab-leadcheck">
    <h2>Lead-Day Sanity Check</h2>
    <p class="note">All forecast stats broken out by integer lead day, <b>including lead 0 (nowcast — same-day "diagnosis") and lead 15 (longest horizon)</b>. Lead 0 should be near-perfect (model's own initialization); lead 15 should be the worst. Anything that breaks that pattern is a data issue worth investigating.</p>
    <div class="panel" id="leadcheck-panel"></div>
  </section>

  <section id="tab-magnitude">
    <h2>Bias by GWDD Magnitude</h2>
    <p class="note">Mean Bias (forecast − actual) and MAE binned by the <b>magnitude of the observed actual GWDD</b>. Tells you whether models systematically under- or over-shoot when actuals are small vs. large. A 15-GWDD miss when the actual was 35 (winter peak) is a different story than a 15-GWDD miss when the actual was 10 (shoulder).</p>
    <div class="panel">
      <div class="controls">
        <label>Window: <select id="mag-window"></select></label>
        <label>Metric: <select id="mag-metric"><option value="Bias">Bias</option><option value="MAE">MAE</option><option value="MAEPct">MAE %</option></select></label>
      </div>
      <div class="chart-wrap"><canvas id="chart-mag"></canvas></div>
      <div id="mag-table" style="margin-top:18px"></div>
    </div>
  </section>

  <section id="tab-kdca">
    <h2>KDCA — ECE Forecast Bias on Extreme Events</h2>
    <p class="note">Single-station ECE 51-member ensemble (00Z, 6-hour steps, 2020–2025) for Reagan National (KDCA). Temperatures converted °C→°F, then HDD = max(0, 65 − T<sub>F</sub>) and CDD = max(0, T<sub>F</sub> − 65). Lead 0 (the same-day "nowcast") serves as the observed proxy. <b>Bias = forecast<sub>L</sub> − actual<sub>0</sub></b>: <b>NEGATIVE bias = model UNDER-predicted</b> (forecast was warmer than reality → missed the cold for HDD, or cooler than reality → missed the heat for CDD). <b>POSITIVE bias = model OVER-predicted</b> (cold bias for HDD, hot bias for CDD). The interesting question: does the bias get worse on the days that actually matter — the coldest winter and hottest summer days?</p>
    <div class="panel" id="kdca-summary"></div>
    <div class="panel">
      <h3 style="margin-top:0">Bias Evolution by Lead Day</h3>
      <p class="note" style="margin-top:4px">Mean forecast bias by lead day (forecast − actual), split by category. <b>Negative = model under-predicted</b> (missed the cold/heat). <b>Positive = model over-predicted</b> (cold bias for HDD, hot bias for CDD). Watch the spread between the "all" and "extreme" lines — that's how much extra bias the model carries on the high-impact days.</p>
      <div class="controls">
        <label>Metric:
          <select id="kdca-metric">
            <option value="HDD_Bias">HDD Bias (cold metric)</option>
            <option value="CDD_Bias">CDD Bias (heat metric)</option>
            <option value="HDD_MAE">HDD MAE</option>
            <option value="CDD_MAE">CDD MAE</option>
          </select>
        </label>
      </div>
      <div class="chart-wrap"><canvas id="chart-kdca"></canvas></div>
    </div>
    <div class="panel">
      <h3 style="margin-top:0">Top Extreme Events — Forecast Evolution</h3>
      <p class="note" style="margin-top:4px">Top 25 events per category, showing how the forecast for that day looked at leads 1, 3, 5, 7, 10, 14 vs. the lead-0 realization. Bias columns = forecast − actual (negative = model under-predicted the extreme).</p>
      <div class="controls">
        <label>Category: <select id="kdca-evt-cat"><option value="cold">Extreme Cold (Winter)</option><option value="hot">Extreme Hot (Summer)</option></select></label>
      </div>
      <div id="kdca-events-table"></div>
    </div>
  </section>

  <section id="tab-blend">
    <h2>Blended Model (GFS + ECE + GEFS)</h2>
    <p class="note">For each lead day, the blend finds non-negative weights (summing to 1) for the three models plus a bias correction, chosen to minimize squared error. <b>All numbers here are out-of-sample.</b> Weights are refit at the start of every month using only forecasts whose valid date has already passed, then applied to that month's runs (walk-forward, expanding window, 12-month warm-up). Single models are scored on exactly the same rows, so the comparison is like-for-like. ECE stops at day 14, so lead 15 blends GFS + GEFS only.</p>
    <div class="panel" id="blend-summary"></div>
    <div class="panel">
      <h3 style="margin-top:0">Out-of-Sample Scorecard</h3>
      <p class="note" style="margin-top:4px">Candidate blending methods vs. each single model, by window. Colors rank rows within each window (green = best).</p>
      <div id="blend-scorecard"></div>
    </div>
    <div class="panel">
      <h3 style="margin-top:0">RMSE by Lead Day</h3>
      <p class="note" style="margin-top:4px">The dip at days 14–15 is a sampling artifact, not real skill: the source files contain far fewer forecasts at those leads (~150 vs ~640), and they fall mostly in summer, when GWDD varies little.</p>
      <div class="chart-wrap"><canvas id="chart-blend-lead"></canvas></div>
    </div>
    <div class="panel">
      <h3 style="margin-top:0">Blend Weights by Lead Day</h3>
      <p class="note" style="margin-top:4px">Current weights, fit on all verified history. These are the weights a live forecast would use today.</p>
      <div class="chart-wrap"><canvas id="chart-blend-weights"></canvas></div>
    </div>
  </section>

  <section id="tab-extreme">
    <h2>Extreme Events</h2>
    <p class="note">Granular row-level lookups — three sections: (1) most extreme forecasts (standardized anomaly, z-scored by month), (2) most extreme observed actuals (top &amp; bottom of the history), (3) biggest forecast misses per individual lead day 10–15.</p>

    <div class="panel">
      <h3 style="margin-top:0">Most Extreme Forecasts (top 50 by |z|)</h3>
      <p class="note" style="margin-top:4px">Anomaly standardized by month-of-year std-dev so summer/winter are comparable. Higher |z| = more unusual call.</p>
      <div id="extreme-forecasts-table"></div>
    </div>

    <div class="panel">
      <h3 style="margin-top:0">Most Extreme Observed Actuals</h3>
      <p class="note" style="margin-top:4px">Top 15 highest and 15 lowest GWDD days in the history file.</p>
      <div id="extreme-actuals-table"></div>
    </div>

    <div class="panel">
      <h3 style="margin-top:0">Biggest Misses — Per Lead Day 10–15</h3>
      <p class="note" style="margin-top:4px">Top 10 largest |Error| rows for the selected lead day. Drill into specific events.</p>
      <div class="controls">
        <label>Lead day: <select id="miss-lead"></select></label>
        <label>Model: <select id="miss-model"><option value="ALL">All models</option><option>GFS</option><option>ECE</option><option>GEFS</option></select></label>
      </div>
      <div id="biggest-misses-table"></div>
    </div>
  </section>

  <section id="tab-insights">
    <h2>Market Insights</h2>
    <div class="panel" id="insights-panel"></div>
  </section>
</main>

<script>
const DATA = __DATA_JSON__;
const COLORS = { GFS: '#58a6ff', ECE: '#f0883e', GEFS: '#3fb950', BLEND: '#d2a8ff' };

Chart.defaults.color = '#e6edf3';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

const TABS = [
  ['tab-scorecard',   'Scorecard'],
  ['tab-blend',       'Blended Model'],
  ['tab-forecast',    'Forecast vs Actual'],
  ['tab-bias',        'Bias & Seasonality'],
  ['tab-convergence', 'Convergence'],
  ['tab-revisions',   'Revisions'],
  ['tab-divergence',  'Divergence'],
  ['tab-regime',      'Regime Performance'],
  ['tab-leadcheck',   'Lead Check (0–15)'],
  ['tab-magnitude',   'Bias by Magnitude'],
  ['tab-extreme',     'Extreme Events'],
  ['tab-kdca',        'KDCA Bias (Extreme)'],
  ['tab-insights',    'Market Insights'],
];

function setupNav() {
  const nav = document.getElementById('tabnav');
  TABS.forEach(([id, label]) => {
    const btn = document.createElement('button');
    btn.textContent = label;
    btn.dataset.target = id;
    if (id === 'tab-scorecard') btn.classList.add('active');
    btn.addEventListener('click', () => {
      document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('main section').forEach(s => s.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(id).classList.add('active');
    });
    nav.appendChild(btn);
  });
}

function fillSelect(id, opts, defVal) {
  const el = document.getElementById(id);
  opts.forEach(v => {
    const o = document.createElement('option');
    o.value = v; o.textContent = v;
    if (v === defVal) o.selected = true;
    el.appendChild(o);
  });
  return el;
}

function lerpColor(t) {
  // t in [0,1]: 0 = green (good), 1 = red (bad)
  const g = { r: 63,  g: 185, b: 80  };
  const r = { r: 248, g: 81,  b: 73  };
  const mix = (a, b) => Math.round(a + (b - a) * t);
  return `rgba(${mix(g.r, r.r)}, ${mix(g.g, r.g)}, ${mix(g.b, r.b)}, 0.28)`;
}

function colorScale(values, isLowerBetter = true) {
  const finite = values.filter(v => isFinite(v));
  if (finite.length === 0) return values.map(() => 'transparent');
  const min = Math.min(...finite), max = Math.max(...finite);
  return values.map(v => {
    if (!isFinite(v) || max === min) return 'transparent';
    let t = (v - min) / (max - min);
    if (!isLowerBetter) t = 1 - t;
    return lerpColor(t);
  });
}

function renderScorecard() {
  const rows = DATA.scorecard;
  const windows = ['1-5d', '6-10d', '11-15d'];
  const models = ['GFS', 'ECE', 'GEFS'];
  const metrics = [
    ['MAE',      'MAE',         true],
    ['RMSE',     'RMSE',        true],
    ['Bias',     'Bias',        true],   // we'll use |Bias| for ranking color
    ['Corr',     'Correlation', false],
    ['HitRate1', '±1 Hit %',    false],
    ['HitRate2', '±2 Hit %',    false],
    ['MeanAbsRev', 'Mean |Chg1|', true],
    ['N',        'N',           null],
  ];

  let html = '<table><thead><tr><th>Model</th><th>Window</th>';
  metrics.forEach(([_, label]) => html += `<th>${label}</th>`);
  html += '</tr></thead><tbody>';

  // Build color scales per (window, metric)
  const scales = {};
  windows.forEach(w => {
    scales[w] = {};
    metrics.forEach(([key, _, lower]) => {
      if (lower === null) return;
      const vals = models.map(m => {
        const row = rows.find(r => r.Model === m && r.Window === w);
        if (!row) return NaN;
        return key === 'Bias' ? Math.abs(row[key]) : row[key];
      });
      scales[w][key] = colorScale(vals, lower);
    });
  });

  windows.forEach(w => {
    models.forEach((m, mi) => {
      const r = rows.find(x => x.Model === m && x.Window === w);
      html += `<tr><td><b style="color:${COLORS[m]}">${m}</b></td><td>${w}</td>`;
      metrics.forEach(([key, _, lower]) => {
        const v = r ? r[key] : null;
        if (v == null || !isFinite(v)) {
          html += '<td>—</td>';
          return;
        }
        const bg = (lower === null) ? 'transparent' : scales[w][key][mi];
        const fmt = (key === 'N') ? v : (key === 'HitRate1' || key === 'HitRate2') ? v.toFixed(1) : v.toFixed(3);
        html += `<td><span class="cell" style="background:${bg}">${fmt}</span></td>`;
      });
      html += '</tr>';
    });
  });
  html += '</tbody></table>';
  document.getElementById('scorecard-panel').innerHTML = html;
}

let chartFA = null;
function renderForecastActual() {
  const model = document.getElementById('fa-model').value;
  const window_ = document.getElementById('fa-window').value;
  const fromV = document.getElementById('fa-from').value;
  const toV   = document.getElementById('fa-to').value;
  let series = (DATA.forecastSeries[model] || {})[window_] || [];
  if (fromV) series = series.filter(p => p.date >= fromV);
  if (toV)   series = series.filter(p => p.date <= toV);

  const labels   = series.map(p => p.date);
  const forecast = series.map(p => p.forecast);
  const actual   = series.map(p => p.actual);
  const upper    = series.map(p => p.actual + 1);
  const lower    = series.map(p => p.actual - 1);

  const ctx = document.getElementById('chart-fa');
  if (chartFA) chartFA.destroy();
  chartFA = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Actual +1', data: upper, borderColor: 'rgba(255,255,255,0.05)', backgroundColor: 'rgba(255,255,255,0.10)', pointRadius: 0, borderWidth: 0, fill: '+1' },
        { label: 'Actual -1', data: lower, borderColor: 'rgba(255,255,255,0.05)', backgroundColor: 'rgba(255,255,255,0.10)', pointRadius: 0, borderWidth: 0, fill: false },
        { label: 'Actual',    data: actual, borderColor: '#ffffff', backgroundColor: '#ffffff', pointRadius: 0, borderWidth: 1.5, tension: 0.15 },
        { label: model + ' Forecast', data: forecast, borderColor: COLORS[model], backgroundColor: COLORS[model], pointRadius: 0, borderWidth: 1.8, tension: 0.15 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 }, grid: { color: '#1f242b' } },
        y: { title: { display: true, text: 'GWDD' }, grid: { color: '#1f242b' } }
      },
      plugins: {
        legend: { labels: { filter: (item) => !item.text.startsWith('Actual +') && !item.text.startsWith('Actual -') } },
        tooltip: { filter: (item) => !item.dataset.label.startsWith('Actual +') && !item.dataset.label.startsWith('Actual -') }
      }
    }
  });
}

let chartBias = null;
const MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function renderBias() {
  const w = document.getElementById('bias-window').value;
  const grain = document.getElementById('bias-grain').value;

  let labels, xTitle, lookup;
  if (grain === 'month') {
    labels = MONTH_LABELS;
    xTitle = 'Month';
    lookup = (m, idx) => {
      const monthNum = idx + 1;
      const row = DATA.monthly.find(r => r.Model === m && r.Window === w && r.Month === monthNum);
      return row ? row.Bias : null;
    };
  } else if (grain === 'halfyear') {
    labels = ['Winter (Nov–Mar)', 'Summer (Apr–Oct)'];
    xTitle = 'Half-year';
    const keys = ['Winter', 'Summer'];
    lookup = (m, idx) => {
      const row = DATA.halfyear.find(r => r.Model === m && r.Window === w && r.HalfYear === keys[idx]);
      return row ? row.Bias : null;
    };
  } else {
    labels = ['DJF','MAM','JJA','SON'];
    xTitle = 'Season';
    lookup = (m, idx) => {
      const row = DATA.seasonal.find(r => r.Model === m && r.Window === w && r.Season === labels[idx]);
      return row ? row.Bias : null;
    };
  }

  const datasets = ['GFS','ECE','GEFS'].map(m => {
    const data = labels.map((_, i) => lookup(m, i));
    const borders = data.map(v => v != null && Math.abs(v) > 1 ? '#f85149' : COLORS[m]);
    const widths  = data.map(v => v != null && Math.abs(v) > 1 ? 3 : 1);
    return {
      label: m, data,
      backgroundColor: COLORS[m] + 'cc',
      borderColor: borders,
      borderWidth: widths,
    };
  });

  const ctx = document.getElementById('chart-bias');
  if (chartBias) chartBias.destroy();
  chartBias = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        y: { title: { display: true, text: 'Mean Bias (Forecast − Actual, GWDD)' }, grid: { color: '#1f242b' } },
        x: { title: { display: true, text: xTitle }, grid: { color: '#1f242b' } }
      }
    }
  });
}

let chartConv = null;
function renderConvergence() {
  const leads = [];
  for (let i = 1; i <= 15; i++) leads.push(i);
  const datasets = ['GFS','ECE','GEFS'].map(m => {
    const data = leads.map(L => {
      const row = DATA.convergence.find(r => r.Model === m && r.LeadDays === L);
      return row ? row.RMSE : null;
    });
    return {
      label: m, data,
      borderColor: COLORS[m], backgroundColor: COLORS[m],
      tension: 0.25, borderWidth: 2, pointRadius: 3,
    };
  });
  const ctx = document.getElementById('chart-conv');
  if (chartConv) chartConv.destroy();
  chartConv = new Chart(ctx, {
    type: 'line',
    data: { labels: leads, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Lead Day' }, grid: { color: '#1f242b' } },
        y: { title: { display: true, text: 'RMSE (GWDD)' }, grid: { color: '#1f242b' } }
      }
    }
  });
}

let chartRev = null;
function renderRevisions() {
  const metrics = ['Chg1', 'Chg2', 'Chg4'];
  const datasets = ['GFS','ECE','GEFS'].map(m => {
    const row = DATA.drift.find(r => r.Model === m) || {};
    return {
      label: m,
      data: [row.AbsChg1 ?? null, row.AbsChg2 ?? null, row.AbsChg4 ?? null],
      backgroundColor: COLORS[m] + 'cc',
      borderColor: COLORS[m],
      borderWidth: 1,
    };
  });
  const ctx = document.getElementById('chart-rev');
  if (chartRev) chartRev.destroy();
  chartRev = new Chart(ctx, {
    type: 'bar',
    data: { labels: metrics, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        y: { title: { display: true, text: 'Mean Absolute Revision (GWDD)' }, grid: { color: '#1f242b' } },
        x: { grid: { color: '#1f242b' } }
      }
    }
  });
}

let chartDiv = null;
function renderDivergence() {
  const all = DATA.divergence;
  const windows = ['1-5d', '6-10d', '11-15d'];
  const palette = { '1-5d': '#58a6ff', '6-10d': '#f0883e', '11-15d': '#bc8cff' };
  // Build a sorted unique date axis from all rolling rows
  const dateSet = new Set(all.map(r => r.ForecastDate));
  const dates = [...dateSet].sort();
  const datasets = windows.map(w => {
    const lookup = {};
    all.filter(r => r.Window === w).forEach(r => { lookup[r.ForecastDate] = r.Rolling; });
    return {
      label: w + ' |GFS−ECE| (30d roll)',
      data: dates.map(d => lookup[d] ?? null),
      borderColor: palette[w], backgroundColor: palette[w],
      tension: 0.15, borderWidth: 1.8, pointRadius: 0, spanGaps: true,
    };
  });
  const ctx = document.getElementById('chart-div');
  if (chartDiv) chartDiv.destroy();
  chartDiv = new Chart(ctx, {
    type: 'line',
    data: { labels: dates, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 }, grid: { color: '#1f242b' } },
        y: { title: { display: true, text: 'Spread (GWDD)' }, grid: { color: '#1f242b' } }
      }
    }
  });
}

let chartRegime = null;
const thresholdLine = {
  id: 'thresholdLine',
  afterDraw(chart, args, opts) {
    const y = chart.scales.y;
    if (!y) return;
    const yPx = y.getPixelForValue(opts.value);
    if (yPx < y.top || yPx > y.bottom) return;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = opts.color || '#f85149';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(chart.chartArea.left, yPx);
    ctx.lineTo(chart.chartArea.right, yPx);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = opts.color || '#f85149';
    ctx.font = '11px sans-serif';
    ctx.fillText(opts.label || '', chart.chartArea.right - 90, yPx - 4);
    ctx.restore();
  }
};
Chart.register(thresholdLine);

function renderRegime() {
  const w = document.getElementById('regime-window').value;
  const regimes = ['Cold', 'Normal', 'Warm'];

  const datasets = ['GFS','ECE','GEFS'].map(m => {
    const data = regimes.map(reg => {
      const row = DATA.regime.find(r => r.Model === m && r.Window === w && r.Regime === reg);
      return row ? row.MAEPct : null;
    });
    const rawMAE = regimes.map(reg => {
      const row = DATA.regime.find(r => r.Model === m && r.Window === w && r.Regime === reg);
      return row ? row.MAE : null;
    });
    const meanAct = regimes.map(reg => {
      const row = DATA.regime.find(r => r.Model === m && r.Window === w && r.Regime === reg);
      return row ? row.MeanActual : null;
    });
    const borders = data.map(v => v != null && v > 15.0 ? '#f85149' : COLORS[m]);
    const widths  = data.map(v => v != null && v > 15.0 ? 2 : 1);
    return {
      label: m, data,
      backgroundColor: COLORS[m] + 'cc',
      borderColor: borders, borderWidth: widths,
      // stash raw values for the tooltip
      _rawMAE: rawMAE, _meanAct: meanAct,
    };
  });
  const ctx = document.getElementById('chart-regime');
  if (chartRegime) chartRegime.destroy();
  chartRegime = new Chart(ctx, {
    type: 'bar',
    data: { labels: regimes, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        thresholdLine: { value: 15.0, color: '#f85149', label: '15% (weak ▲)' },
        tooltip: { callbacks: { label: (ctx) => {
          const ds = ctx.dataset;
          const i = ctx.dataIndex;
          const pct  = ctx.parsed.y;
          const mae  = ds._rawMAE  ? ds._rawMAE[i]  : null;
          const meanA = ds._meanAct ? ds._meanAct[i] : null;
          if (mae == null) return `${ds.label}: ${pct.toFixed(2)}%`;
          return `${ds.label}: ${pct.toFixed(2)}% of mean   (raw MAE ${mae.toFixed(3)} GWDD on mean actual ${meanA.toFixed(2)})`;
        } } }
      },
      scales: {
        y: { title: { display: true, text: 'MAE as % of mean observed GWDD — lower = better' }, grid: { color: '#1f242b' }, beginAtZero: true,
             ticks: { callback: (v) => v + '%' } },
        x: { title: { display: true, text: 'Observed Regime (tercile of actual GWDD)' }, grid: { color: '#1f242b' } }
      }
    }
  });
}

function fmt(v, d=3) { return (v == null || !isFinite(v)) ? '—' : v.toFixed(d); }

function renderLeadCheck() {
  const leads = [];
  for (let i = 0; i <= 15; i++) leads.push(i);
  let html = '<table><thead><tr><th>Model</th><th>Lead</th><th>N</th><th>MAE</th><th>RMSE</th><th>Bias</th><th>Corr</th><th>±1 Hit%</th><th>±2 Hit%</th></tr></thead><tbody>';
  ['GFS','ECE','GEFS'].forEach(m => {
    leads.forEach(L => {
      const row = DATA.perLead.find(r => r.Model === m && r.LeadDays === L);
      if (!row) return;
      const highlight = (L === 0 || L === 15) ? ' style="background:rgba(88,166,255,0.08)"' : '';
      const tag = L === 0 ? ' <span style="color:#8b949e;font-size:11px">(nowcast)</span>'
                : L === 15 ? ' <span style="color:#8b949e;font-size:11px">(longest)</span>' : '';
      html += `<tr${highlight}><td><b style="color:${COLORS[m]}">${m}</b></td><td>${L}${tag}</td>`
            + `<td>${row.N}</td><td>${fmt(row.MAE)}</td><td>${fmt(row.RMSE)}</td>`
            + `<td>${row.Bias != null ? (row.Bias>=0?'+':'')+row.Bias.toFixed(3) : '—'}</td>`
            + `<td>${fmt(row.Corr)}</td><td>${fmt(row.HitRate1,1)}</td><td>${fmt(row.HitRate2,1)}</td></tr>`;
    });
  });
  html += '</tbody></table>';
  document.getElementById('leadcheck-panel').innerHTML = html;
}

let chartMag = null;
function renderMagnitude() {
  const w = document.getElementById('mag-window').value;
  const metric = document.getElementById('mag-metric').value;
  const labels = DATA.magLabels;
  const datasets = ['GFS','ECE','GEFS'].map(m => {
    const data = labels.map(lab => {
      const row = DATA.magnitude.find(r => r.Model === m && r.Window === w && r.MagBin === lab);
      return row ? row[metric] : null;
    });
    return {
      label: m, data,
      backgroundColor: COLORS[m] + 'cc',
      borderColor: COLORS[m], borderWidth: 1,
    };
  });
  const ctx = document.getElementById('chart-mag');
  if (chartMag) chartMag.destroy();
  const yTitle = metric === 'Bias' ? 'Mean Bias (GWDD)'
              : metric === 'MAE'  ? 'MAE (GWDD)'
              : 'MAE as % of mean actual';
  chartMag = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        tooltip: { callbacks: { label: (ctx) => {
          const row = DATA.magnitude.find(r => r.Model === ctx.dataset.label && r.Window === w && r.MagBin === labels[ctx.dataIndex]);
          if (!row) return ctx.dataset.label + ': —';
          return `${ctx.dataset.label}: ${metric}=${ctx.parsed.y.toFixed(2)} | N=${row.N} | mean actual=${row.MeanActual.toFixed(2)}`;
        }}}
      },
      scales: {
        y: { title: { display: true, text: yTitle }, grid: { color: '#1f242b' } },
        x: { title: { display: true, text: 'Observed GWDD magnitude bin' }, grid: { color: '#1f242b' } }
      }
    }
  });

  // Companion table with sample counts and all metrics
  let html = '<table><thead><tr><th>Model</th><th>Window</th><th>Bin</th><th>N</th><th>MeanActual</th><th>Bias</th><th>MAE</th><th>MAE %</th></tr></thead><tbody>';
  ['GFS','ECE','GEFS'].forEach(m => {
    labels.forEach(lab => {
      const r = DATA.magnitude.find(x => x.Model === m && x.Window === w && x.MagBin === lab);
      if (!r) return;
      html += `<tr><td><b style="color:${COLORS[m]}">${m}</b></td><td>${w}</td><td>${lab}</td>`
            + `<td>${r.N}</td><td>${fmt(r.MeanActual,2)}</td>`
            + `<td>${r.Bias>=0?'+':''}${r.Bias.toFixed(3)}</td>`
            + `<td>${fmt(r.MAE)}</td><td>${fmt(r.MAEPct,2)}%</td></tr>`;
    });
  });
  html += '</tbody></table>';
  document.getElementById('mag-table').innerHTML = html;
}

function renderExtremeForecasts() {
  let html = '<table><thead><tr><th>Model</th><th>RunDate</th><th>ForecastDate</th><th>Lead</th><th>Forecast</th><th>Anomaly</th><th>z-score</th><th>Actual</th><th>Error</th></tr></thead><tbody>';
  DATA.extremeForecasts.forEach(r => {
    const errStr = r.Error != null ? (r.Error>=0?'+':'') + r.Error.toFixed(2) : '—';
    html += `<tr><td><b style="color:${COLORS[r.Model]}">${r.Model}</b></td>`
          + `<td>${r.RunDate}</td><td>${r.ForecastDate}</td><td>${r.LeadDays}</td>`
          + `<td>${fmt(r.GWDD,2)}</td><td>${r.Anomaly>=0?'+':''}${r.Anomaly.toFixed(2)}</td>`
          + `<td>${r.AnomalyZ>=0?'+':''}${r.AnomalyZ.toFixed(2)}</td>`
          + `<td>${fmt(r.Actual,2)}</td><td>${errStr}</td></tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('extreme-forecasts-table').innerHTML = html;
}

function renderExtremeActuals() {
  let html = '<table><thead><tr><th>Side</th><th>Date</th><th>Actual GWDD</th></tr></thead><tbody>';
  DATA.extremeActuals.forEach(r => {
    html += `<tr><td>${r.Side}</td><td>${r.ForecastDate}</td><td>${r.Actual.toFixed(3)}</td></tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('extreme-actuals-table').innerHTML = html;
}

function renderBiggestMisses() {
  const lead = parseInt(document.getElementById('miss-lead').value, 10);
  const model = document.getElementById('miss-model').value;
  let rows = DATA.biggestMisses.filter(r => r.LeadDays === lead);
  if (model !== 'ALL') rows = rows.filter(r => r.Model === model);
  rows = rows.sort((a,b) => Math.abs(b.Error) - Math.abs(a.Error));
  let html = '<table><thead><tr><th>Model</th><th>RunDate</th><th>ForecastDate</th><th>Forecast</th><th>Actual</th><th>Error</th><th>|Error|</th><th>Anomaly</th></tr></thead><tbody>';
  rows.forEach(r => {
    const errSign = r.Error >= 0 ? '+' : '';
    html += `<tr><td><b style="color:${COLORS[r.Model]}">${r.Model}</b></td>`
          + `<td>${r.RunDate}</td><td>${r.ForecastDate}</td>`
          + `<td>${r.Forecast.toFixed(2)}</td><td>${r.Actual.toFixed(2)}</td>`
          + `<td>${errSign}${r.Error.toFixed(2)}</td><td>${Math.abs(r.Error).toFixed(2)}</td>`
          + `<td>${r.Anomaly == null ? '—' : (r.Anomaly>=0?'+':'') + r.Anomaly.toFixed(2)}</td></tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('biggest-misses-table').innerHTML = html;
}

let chartKdca = null;
const KDCA_CAT_COLORS = {
  All:                '#8b949e',
  Winter:             '#58a6ff',
  Summer:             '#f0883e',
  ExtremeColdWinter:  '#79c0ff',
  ExtremeHotSummer:   '#f85149',
};
const KDCA_CAT_LABELS = {
  All:                'All days',
  Winter:             'Winter (Nov–Mar)',
  Summer:             'Summer (Apr–Oct)',
  ExtremeColdWinter:  'Extreme Cold (top 10% HDD winter)',
  ExtremeHotSummer:   'Extreme Hot  (top 10% CDD summer)',
};

function renderKdcaSummary() {
  if (!DATA.kdca) {
    document.getElementById('kdca-summary').innerHTML = '<p class="note">KDCA dataset not loaded.</p>';
    return;
  }
  const s = DATA.kdca.summary;
  let html = '<table><tbody>';
  Object.entries(s).forEach(([k, v]) => {
    html += `<tr><td><b>${k}</b></td><td>${v == null ? '—' : v}</td></tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('kdca-summary').innerHTML = html;
}

function renderKdcaChart() {
  if (!DATA.kdca) return;
  const metric = document.getElementById('kdca-metric').value;
  const isCDD = metric.startsWith('CDD');
  const isMAE = metric.endsWith('MAE');
  const categoriesForMetric = isCDD
    ? ['All','Summer','ExtremeHotSummer']
    : ['All','Winter','ExtremeColdWinter'];

  const leads = Array.from({length: 15}, (_, i) => i);
  const datasets = categoriesForMetric.map(cat => {
    const data = leads.map(L => {
      const row = DATA.kdca.bias.find(r => r.Category === cat && r.Lead === L);
      return row ? row[metric] : null;
    });
    return {
      label: KDCA_CAT_LABELS[cat],
      data,
      borderColor: KDCA_CAT_COLORS[cat],
      backgroundColor: KDCA_CAT_COLORS[cat],
      borderWidth: cat.startsWith('Extreme') ? 2.5 : 1.8,
      pointRadius: 3,
      tension: 0.2,
    };
  });

  const ctx = document.getElementById('chart-kdca');
  if (chartKdca) chartKdca.destroy();
  const yTitle = isMAE
    ? `${isCDD ? 'CDD' : 'HDD'} MAE (°F·days)`
    : `${isCDD ? 'CDD' : 'HDD'} Bias = forecast − actual  (°F·days; negative = under-predicted)`;
  chartKdca = new Chart(ctx, {
    type: 'line',
    data: { labels: leads, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        thresholdLine: isMAE ? undefined : { value: 0, color: '#8b949e', label: 'zero bias' },
        tooltip: { callbacks: { label: (ctx) => {
          const row = DATA.kdca.bias.find(r => r.Lead === ctx.dataIndex && KDCA_CAT_LABELS[r.Category] === ctx.dataset.label);
          const n = row ? row.N : '?';
          return `${ctx.dataset.label}: ${ctx.parsed.y == null ? '—' : ctx.parsed.y.toFixed(2)} (N=${n})`;
        }}}
      },
      scales: {
        x: { title: { display: true, text: 'Lead Day (0 = nowcast/actual, 14 = 2 weeks out)' }, grid: { color: '#1f242b' } },
        y: { title: { display: true, text: yTitle }, grid: { color: '#1f242b' } }
      }
    }
  });
}

function renderKdcaEvents() {
  if (!DATA.kdca) return;
  const cat = document.getElementById('kdca-evt-cat').value;
  const rows = cat === 'cold' ? DATA.kdca.extremeCold : DATA.kdca.extremeHot;
  const metric = cat === 'cold' ? 'HDD' : 'CDD';
  const leads = [1, 3, 5, 7, 10, 14];
  let html = '<table><thead><tr>'
           + '<th>ValidDate</th><th>Month</th>'
           + `<th>${metric}<sub>0</sub></th>`;
  leads.forEach(L => { html += `<th>L${L}</th>`; });
  leads.forEach(L => { html += `<th>Bias L${L}</th>`; });
  html += '</tr></thead><tbody>';
  rows.forEach(r => {
    html += `<tr><td>${r.ValidDate}</td><td>${r.Month}</td>`
          + `<td><b>${r.Lead0.toFixed(2)}</b></td>`;
    leads.forEach(L => {
      const k = 'Lead' + String(L).padStart(2,'0');
      const v = r[k];
      html += `<td>${v == null ? '—' : v.toFixed(2)}</td>`;
    });
    leads.forEach(L => {
      const k = 'Bias' + String(L).padStart(2,'0');
      const v = r[k];
      if (v == null) { html += '<td>—</td>'; return; }
      const sign = v >= 0 ? '+' : '';
      const color = Math.abs(v) > 3 ? '#f85149' : (Math.abs(v) > 1.5 ? '#f0883e' : '#8b949e');
      html += `<td style="color:${color}">${sign}${v.toFixed(2)}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('kdca-events-table').innerHTML = html;
}

const BLEND_LABELS = {
  EQUAL: 'Equal average',
  INV_MSE: 'Inverse-MSE weights',
  WEIGHTED_BC: 'Optimized weights + bias corr.',
  WEIGHTED_BC_SEASONAL: 'Optimized weights + seasonal bias corr.',
};
let chartBlendLead = null, chartBlendWeights = null;

function renderBlendSummary() {
  const b = DATA.blend;
  let html = `<p style="margin-top:0">Test period: runs <b>${b.testStart}</b> to <b>${b.testEnd}</b> `
           + `(${b.nTest.toLocaleString()} forecasts). Selected method: <b style="color:${COLORS.BLEND}">${BLEND_LABELS[b.best]}</b>.</p>`;
  html += '<table><thead><tr><th>Window</th><th>Best single model</th><th>Single RMSE</th><th>Blend RMSE</th><th>RMSE improvement</th><th>Blend ±1 Hit %</th></tr></thead><tbody>';
  ['1-5d','6-10d','11-15d'].forEach(w => {
    const singles = ['GFS','ECE','GEFS'].map(m => b.windowStats.find(r => r.Model === m && r.Window === w)).filter(Boolean);
    const best = singles.reduce((a, c) => c.RMSE < a.RMSE ? c : a);
    const bl = b.windowStats.find(r => r.Model === 'BLEND' && r.Window === w);
    const imp = (best.RMSE - bl.RMSE) / best.RMSE * 100;
    html += `<tr><td>${w}</td><td><b style="color:${COLORS[best.Model]}">${best.Model}</b></td><td>${fmt(best.RMSE)}</td>`
          + `<td><b style="color:${COLORS.BLEND}">${fmt(bl.RMSE)}</b></td><td>${imp >= 0 ? '+' : ''}${imp.toFixed(1)}%</td><td>${fmt(bl.HitRate1, 1)}</td></tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('blend-summary').innerHTML = html;
}

function renderBlendScorecard() {
  const rows = DATA.blend.windowStats;
  const models = ['GFS','ECE','GEFS', ...Object.keys(BLEND_LABELS)];
  const metrics = [['RMSE','RMSE',true], ['MAE','MAE',true], ['Bias','Bias',true], ['Corr','Correlation',false],
                   ['HitRate1','±1 Hit %',false], ['MeanAbsRev','Mean |Chg1|',true], ['N','N',null]];
  let html = '<table><thead><tr><th>Model / Method</th><th>Window</th>';
  metrics.forEach(([_, label]) => html += `<th>${label}</th>`);
  html += '</tr></thead><tbody>';
  ['1-5d','6-10d','11-15d'].forEach(w => {
    const wr = models.map(m => rows.find(r => r.Model === m && r.Window === w));
    const scales = {};
    metrics.forEach(([key, _, lower]) => {
      if (lower !== null) scales[key] = colorScale(wr.map(r => r ? (key === 'Bias' ? Math.abs(r[key]) : r[key]) : NaN), lower);
    });
    models.forEach((m, i) => {
      const r = wr[i];
      if (!r) return;
      const name = COLORS[m] ? `<b style="color:${COLORS[m]}">${m}</b>`
                 : (m === DATA.blend.best ? `<b style="color:${COLORS.BLEND}">${BLEND_LABELS[m]} ★</b>` : BLEND_LABELS[m]);
      html += `<tr><td>${name}</td><td>${w}</td>`;
      metrics.forEach(([key, _, lower]) => {
        const v = r[key];
        if (v == null || !isFinite(v)) { html += '<td>—</td>'; return; }
        const bg = lower === null ? 'transparent' : scales[key][i];
        const txt = key === 'N' ? v : key === 'HitRate1' ? v.toFixed(1) : v.toFixed(3);
        html += `<td><span class="cell" style="background:${bg}">${txt}</span></td>`;
      });
      html += '</tr>';
    });
  });
  html += '</tbody></table><p class="note">★ = selected as the blend (lowest out-of-sample RMSE across all leads).</p>';
  document.getElementById('blend-scorecard').innerHTML = html;
}

function renderBlendLeadChart() {
  const leads = Array.from({length: 15}, (_, i) => i + 1);
  const datasets = ['GFS','ECE','GEFS','BLEND'].map(m => ({
    label: m,
    data: leads.map(L => {
      const r = DATA.blend.leadStats.find(x => x.Model === m && x.LeadDays === L);
      return r ? r.RMSE : null;
    }),
    borderColor: COLORS[m], backgroundColor: COLORS[m],
    borderWidth: m === 'BLEND' ? 3 : 1.6, pointRadius: m === 'BLEND' ? 3.5 : 2, tension: 0.2,
  }));
  if (chartBlendLead) chartBlendLead.destroy();
  chartBlendLead = new Chart(document.getElementById('chart-blend-lead'), {
    type: 'line',
    data: { labels: leads, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Lead Day' }, grid: { color: '#1f242b' } },
        y: { title: { display: true, text: 'RMSE (GWDD, out-of-sample)' }, grid: { color: '#1f242b' } }
      }
    }
  });
}

function renderBlendWeights() {
  const w = DATA.blend.weights;
  const datasets = ['GFS','ECE','GEFS'].map(m => ({
    label: m, data: w.map(r => r['w_' + m]), backgroundColor: COLORS[m], stack: 'w',
  }));
  if (chartBlendWeights) chartBlendWeights.destroy();
  chartBlendWeights = new Chart(document.getElementById('chart-blend-weights'), {
    type: 'bar',
    data: { labels: w.map(r => r.LeadDays), datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${(c.parsed.y * 100).toFixed(0)}%` } } },
      scales: {
        x: { stacked: true, title: { display: true, text: 'Lead Day' }, grid: { color: '#1f242b' } },
        y: { stacked: true, max: 1, title: { display: true, text: 'Weight' },
             ticks: { callback: v => (v * 100).toFixed(0) + '%' }, grid: { color: '#1f242b' } }
      }
    }
  });
}

function renderInsights() {
  const html = '<ul class="insights">' +
    DATA.insights.map(line => `<li>${line}</li>`).join('') +
    '</ul>';
  document.getElementById('insights-panel').innerHTML = html;
}

function init() {
  setupNav();

  // Forecast vs Actual selectors
  fillSelect('fa-model',  ['GFS','ECE','GEFS'], 'GFS');
  fillSelect('fa-window', ['1-5d','6-10d','11-15d'], '1-5d');
  const allDates = [];
  Object.values(DATA.forecastSeries).forEach(byW => Object.values(byW).forEach(arr => arr.forEach(p => allDates.push(p.date))));
  if (allDates.length) {
    allDates.sort();
    document.getElementById('fa-from').value = allDates[0];
    document.getElementById('fa-to').value = allDates[allDates.length - 1];
  }
  ['fa-model','fa-window','fa-from','fa-to'].forEach(id => {
    document.getElementById(id).addEventListener('change', renderForecastActual);
  });

  fillSelect('bias-window', ['1-5d','6-10d','11-15d'], '6-10d');
  document.getElementById('bias-window').addEventListener('change', renderBias);
  document.getElementById('bias-grain').addEventListener('change', renderBias);

  fillSelect('regime-window', ['1-5d','6-10d','11-15d'], '6-10d');
  document.getElementById('regime-window').addEventListener('change', renderRegime);

  fillSelect('mag-window', ['1-5d','6-10d','11-15d'], '1-5d');
  document.getElementById('mag-window').addEventListener('change', renderMagnitude);
  document.getElementById('mag-metric').addEventListener('change', renderMagnitude);

  fillSelect('miss-lead', ['10','11','12','13','14','15'], '15');
  document.getElementById('miss-lead').addEventListener('change', renderBiggestMisses);
  document.getElementById('miss-model').addEventListener('change', renderBiggestMisses);

  if (DATA.kdca) {
    document.getElementById('kdca-metric').addEventListener('change', renderKdcaChart);
    document.getElementById('kdca-evt-cat').addEventListener('change', renderKdcaEvents);
  }

  renderScorecard();
  renderForecastActual();
  renderBias();
  renderConvergence();
  renderRevisions();
  renderDivergence();
  renderRegime();
  renderLeadCheck();
  renderMagnitude();
  renderExtremeForecasts();
  renderExtremeActuals();
  renderBiggestMisses();
  if (DATA.kdca) {
    renderKdcaSummary();
    renderKdcaChart();
    renderKdcaEvents();
  }
  if (DATA.blend) {
    renderBlendSummary();
    renderBlendScorecard();
    renderBlendLeadChart();
    renderBlendWeights();
  } else {
    document.getElementById('blend-summary').innerHTML = '<p class="note">Blend not computed.</p>';
  }
  renderInsights();
}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""


def df_to_records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for c in out.select_dtypes(include=["datetime64[ns]"]).columns:
        out[c] = out[c].dt.strftime("%Y-%m-%d")
    return json.loads(out.to_json(orient="records"))


def load_chartjs_inline() -> str:
    local = WORKSPACE / "chart.umd.min.js"
    if local.exists():
        return local.read_text(encoding="utf-8")
    return ""


def write_dashboard(mw: pd.DataFrame, patterns: dict, forecast_series: dict, insights: list[str], kdca: dict | None = None, blend: dict | None = None) -> None:
    data = {
        "scorecard":      df_to_records(mw),
        "seasonal":       df_to_records(patterns["seasonal"]),
        "monthly":        df_to_records(patterns["monthly"]),
        "halfyear":       df_to_records(patterns["halfyear"]),
        "convergence":    df_to_records(patterns["convergence"]),
        "drift":          df_to_records(patterns["drift"]),
        "divergence":     df_to_records(patterns["divergence"]),
        "regime":         df_to_records(patterns["regime"]),
        "extremes":       df_to_records(patterns["extremes"]),
        "rankings":       patterns["rankings"],
        "forecastSeries": forecast_series,
        "insights":       insights,
        "tercile":        {"q33": patterns["tercile_q33"], "q67": patterns["tercile_q67"]},
        "extreme":        {"p10": patterns["extreme_p10"], "p90": patterns["extreme_p90"]},
        "perLead":          df_to_records(patterns["per_lead"]),
        "magnitude":        df_to_records(patterns["magnitude"]),
        "magLabels":        MAG_LABELS,
        "extremeForecasts": df_to_records(patterns["extreme_forecasts"]),
        "extremeActuals":   df_to_records(patterns["extreme_actuals"]),
        "biggestMisses":    df_to_records(patterns["biggest_misses"]),
    }
    if kdca is not None:
        data["kdca"] = {
            "bias":         df_to_records(kdca["bias"]),
            "extremeCold":  df_to_records(kdca["extreme_cold"]),
            "extremeHot":   df_to_records(kdca["extreme_hot"]),
            "summary":      kdca["summary"],
            "threshWinter": kdca["thresh_winter"],
            "threshSummer": kdca["thresh_summer"],
        }
    else:
        data["kdca"] = None
    if blend is not None:
        data["blend"] = {
            "best":        blend["best"],
            "windowStats": df_to_records(blend["window_stats"].astype({"Window": str})),
            "leadStats":   df_to_records(blend["lead_stats"]),
            "weights":     df_to_records(blend["weights"]),
            "testStart":   blend["test_start"],
            "testEnd":     blend["test_end"],
            "nTest":       blend["n_test"],
        }
    else:
        data["blend"] = None
    payload = json.dumps(data, allow_nan=False, default=str)
    html = HTML_TEMPLATE.replace("__DATA_JSON__", payload)
    chartjs = load_chartjs_inline()
    if chartjs:
        # Replace the cdn script tag with an inline copy so the file works offline
        cdn_tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>'
        inline_tag = "<script>\n" + chartjs + "\n</script>"
        html = html.replace(cdn_tag, inline_tag)
    HTML_OUT.write_text(html, encoding="utf-8")


# ---------- Section KDCA: ECE KDCA 6-hour temperature → HDD/CDD bias analysis ----------
#
# Source: ECE 51-member ensemble mean, 00Z run, 6-hour steps 0–360h, 2020–2025.
# Steps 0/6/12/18 = lead day 0 (nowcast, proxy for observed).
# Steps L*24, L*24+6, L*24+12, L*24+18 = lead day L (L=0..14).
# Lead 15 excluded (step 360 only — incomplete day).
# Convert °C → °F (T_F = T_C * 1.8 + 32). HDD = max(0, 65 - T_F); CDD = max(0, T_F - 65).
# Bias_L = HDD_lead_L - HDD_lead_0 (positive = cold bias = model over-predicts HDD).

KDCA_WINTER_MONTHS = {11, 12, 1, 2, 3}   # XZFGH per the project convention
KDCA_SUMMER_MONTHS = {4, 5, 6, 7, 8, 9, 10}


def load_kdca_evolution() -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    if not KDCA_FILE.exists():
        return None, None
    raw = pd.read_csv(KDCA_FILE)
    raw["step"] = raw["step"].astype(int)
    raw = raw.set_index("step")
    raw.columns = pd.to_datetime(raw.columns, format="%Y%m%d")

    import warnings
    records = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        for init_date in raw.columns:
            col = raw[init_date]
            for L in range(15):  # 0..14, exclude lead 15 (incomplete)
                steps = [L*24, L*24+6, L*24+12, L*24+18]
                temps_c = col.reindex(steps).values
                mean_c = np.nanmean(temps_c)
                if np.isnan(mean_c):
                    continue
                mean_f = mean_c * 1.8 + 32.0
                hdd = max(0.0, 65.0 - mean_f)
                cdd = max(0.0, mean_f - 65.0)
                records.append({
                    "ValidDate": init_date + pd.Timedelta(days=L),
                    "Lead":      L,
                    "HDD":       hdd,
                    "CDD":       cdd,
                })

    long_df = pd.DataFrame(records)
    evo_hdd = long_df.pivot(index="ValidDate", columns="Lead", values="HDD").sort_index()
    evo_cdd = long_df.pivot(index="ValidDate", columns="Lead", values="CDD").sort_index()
    return evo_hdd, evo_cdd


def kdca_analysis(evo_hdd: pd.DataFrame, evo_cdd: pd.DataFrame) -> dict:
    leads = list(range(15))
    # Keep only valid dates that have lead 0 + at least some other leads
    evo_hdd = evo_hdd.dropna(subset=[0])
    evo_cdd = evo_cdd.dropna(subset=[0])
    months = evo_hdd.index.month

    masks = {
        "All":           pd.Series(True, index=evo_hdd.index),
        "Winter":        pd.Series(months.isin(KDCA_WINTER_MONTHS), index=evo_hdd.index),
        "Summer":        pd.Series(months.isin(KDCA_SUMMER_MONTHS), index=evo_hdd.index),
    }
    # Extreme winter cold: top 10% of lead-0 HDD within winter months
    winter_idx = evo_hdd.index[months.isin(KDCA_WINTER_MONTHS)]
    if len(winter_idx) > 0:
        thresh_winter = evo_hdd.loc[winter_idx, 0].quantile(0.90)
        extreme_cold_idx = evo_hdd.index.isin(
            winter_idx[evo_hdd.loc[winter_idx, 0] >= thresh_winter]
        )
        masks["ExtremeColdWinter"] = pd.Series(extreme_cold_idx, index=evo_hdd.index)
    else:
        thresh_winter = float("nan")
        masks["ExtremeColdWinter"] = pd.Series(False, index=evo_hdd.index)

    # Extreme summer heat: top 10% of lead-0 CDD within summer months
    summer_idx = evo_cdd.index[months.isin(KDCA_SUMMER_MONTHS)]
    if len(summer_idx) > 0:
        thresh_summer = evo_cdd.loc[summer_idx, 0].quantile(0.90)
        extreme_hot_idx = evo_cdd.index.isin(
            summer_idx[evo_cdd.loc[summer_idx, 0] >= thresh_summer]
        )
        masks["ExtremeHotSummer"] = pd.Series(extreme_hot_idx, index=evo_hdd.index)
    else:
        thresh_summer = float("nan")
        masks["ExtremeHotSummer"] = pd.Series(False, index=evo_hdd.index)

    # Bias by lead, per category.
    # Convention: Bias_L = forecast - actual = HDD_lead_L - HDD_lead_0.
    #   NEGATIVE HDD bias = model UNDER-predicted HDD (missed the cold; forecast was warmer than reality).
    #   POSITIVE HDD bias = model OVER-predicted HDD (cold bias; forecast was colder than reality).
    bias_rows = []
    for cat, mask in masks.items():
        sub_hdd = evo_hdd.loc[mask]
        sub_cdd = evo_cdd.loc[mask]
        for L in leads:
            bias_h = (sub_hdd[L] - sub_hdd[0]).dropna()
            bias_c = (sub_cdd[L] - sub_cdd[0]).dropna()
            bias_rows.append({
                "Category":  cat,
                "Lead":      L,
                "N":         int(len(bias_h)),
                "HDD_Bias":  float(bias_h.mean()) if len(bias_h) else float("nan"),
                "HDD_MAE":   float(bias_h.abs().mean()) if len(bias_h) else float("nan"),
                "CDD_Bias":  float(bias_c.mean()) if len(bias_c) else float("nan"),
                "CDD_MAE":   float(bias_c.abs().mean()) if len(bias_c) else float("nan"),
            })
    bias_df = pd.DataFrame(bias_rows)

    # Per-extreme-event drill-down (HDD evolution for top winter cold; CDD for top summer hot)
    def event_table(evo: pd.DataFrame, idx_mask: pd.Series, metric_name: str, top_n: int = 25) -> pd.DataFrame:
        sub = evo.loc[idx_mask]
        if sub.empty: return pd.DataFrame()
        top = sub[0].nlargest(top_n)
        rows = []
        for vd, lead0_val in top.items():
            r = {"ValidDate": vd.strftime("%Y-%m-%d"),
                 "Month": vd.strftime("%b"),
                 "Lead0": float(lead0_val)}
            for L in [1, 3, 5, 7, 10, 14]:
                v = sub.loc[vd, L] if L in sub.columns else float("nan")
                r[f"Lead{L:02d}"] = float(v) if pd.notna(v) else None
                # Bias = forecast - actual (negative = model under-predicted; missed the extreme)
                r[f"Bias{L:02d}"] = float(v - lead0_val) if pd.notna(v) else None
            rows.append(r)
        out = pd.DataFrame(rows)
        out["Metric"] = metric_name
        return out

    extreme_cold_table = event_table(evo_hdd, masks["ExtremeColdWinter"], "HDD")
    extreme_hot_table  = event_table(evo_cdd, masks["ExtremeHotSummer"],  "CDD")

    # Aggregate stats
    summary = {
        "Date range": f"{evo_hdd.index.min().date()} to {evo_hdd.index.max().date()}",
        "Total valid dates": int(len(evo_hdd)),
        "Winter days": int(masks["Winter"].sum()),
        "Summer days": int(masks["Summer"].sum()),
        "Extreme cold winter days (top 10% HDD)": int(masks["ExtremeColdWinter"].sum()),
        "Extreme hot summer days (top 10% CDD)": int(masks["ExtremeHotSummer"].sum()),
        "Winter top-10% HDD threshold (F-days)": round(float(thresh_winter), 2) if not np.isnan(thresh_winter) else None,
        "Summer top-10% CDD threshold (F-days)": round(float(thresh_summer), 2) if not np.isnan(thresh_summer) else None,
    }

    return {
        "bias":             bias_df,
        "extreme_cold":     extreme_cold_table,
        "extreme_hot":      extreme_hot_table,
        "summary":          summary,
        "thresh_winter":    float(thresh_winter) if not np.isnan(thresh_winter) else None,
        "thresh_summer":    float(thresh_summer) if not np.isnan(thresh_summer) else None,
    }


# ---------- Section 7: console summary ----------

def print_summary(mw: pd.DataFrame, patterns: dict, n_rows: int, kdca: dict | None = None) -> None:
    print("=" * 68)
    print("GWDD MODEL VERIFICATION — TOP-LINE FINDINGS")
    print("=" * 68)
    print(f"Verified rows: {n_rows:,}")
    print()
    print("Best model per window (by RMSE):")
    for w in WINDOWS:
        sub = mw[mw["Window"] == w].sort_values("RMSE")
        if not sub.empty:
            r = sub.iloc[0]
            print(f"  {w:7s}  {r['Model']:5s}  RMSE={r['RMSE']:.3f}  MAE={r['MAE']:.3f}  "
                  f"Bias={r['Bias']:+.3f}  Corr={r['Corr']:.3f}  Hit±1={r['HitRate1']:.1f}%")
    print()
    biggest = mw.loc[mw["Bias"].abs().idxmax()]
    print(f"Largest bias: {biggest['Model']} {biggest['Window']}  Bias={biggest['Bias']:+.3f}")
    drift = patterns["drift"].set_index("Model")
    print(f"Most volatile (mean |Chg1|):   {drift['AbsChg1'].idxmax()}  "
          f"({drift['AbsChg1'].max():.3f})")
    print(f"Most stable   (mean |Chg1|):   {drift['AbsChg1'].idxmin()}  "
          f"({drift['AbsChg1'].min():.3f})")
    flagged = patterns["seasonal"][patterns["seasonal"]["Flag"]]
    if not flagged.empty:
        print()
        print("Seasonal bias flags (|bias| > 1):")
        for _, r in flagged.iterrows():
            print(f"  {r['Model']:5s} {r['Window']:7s} {r['Season']}  Bias={r['Bias']:+.3f}  N={r['N']}")
    else:
        print("No |bias| > 1 seasonal flags found.")
    print()
    print(f"Outputs written:")
    print(f"  {CSV_OUT}")
    print(f"  {HTML_OUT}")

    if kdca is not None:
        print()
        print("-" * 68)
        print("KDCA ECE bias by lead day (HDD F-days, Bias = forecast - actual; negative = under-predicted):")
        print(f"  Date range: {kdca['summary']['Date range']}")
        print(f"  Winter (Nov-Mar) days: {kdca['summary']['Winter days']}  |  "
              f"Extreme cold (top 10% HDD): {kdca['summary']['Extreme cold winter days (top 10% HDD)']}")
        print(f"  Summer (Apr-Oct) days: {kdca['summary']['Summer days']}  |  "
              f"Extreme hot  (top 10% CDD): {kdca['summary']['Extreme hot summer days (top 10% CDD)']}")
        print(f"  Thresholds: Winter HDD >= {kdca['summary']['Winter top-10% HDD threshold (F-days)']}  "
              f"Summer CDD >= {kdca['summary']['Summer top-10% CDD threshold (F-days)']}")
        print()
        print(f"  {'Lead':>4s}  {'AllHDDb':>8s}  {'WinHDDb':>8s}  {'XColdHDDb':>10s}  {'SumCDDb':>8s}  {'XHotCDDb':>9s}")
        bias = kdca["bias"]
        for L in range(15):
            def cell(cat, col):
                row = bias[(bias["Category"]==cat) & (bias["Lead"]==L)]
                if row.empty: return "  --"
                v = row.iloc[0][col]
                return f"{v:+.2f}" if pd.notna(v) else "  --"
            print(f"  {L:>4d}  {cell('All','HDD_Bias'):>8s}  {cell('Winter','HDD_Bias'):>8s}  "
                  f"{cell('ExtremeColdWinter','HDD_Bias'):>10s}  {cell('Summer','CDD_Bias'):>8s}  "
                  f"{cell('ExtremeHotSummer','CDD_Bias'):>9s}")
        print(f"  Output: {KDCA_CSV_OUT}")
    print("=" * 68)


# ---------- Main ----------

def main() -> None:
    df, history, all_leads = load_data()
    n_rows = len(df)

    verification = build_verification(df)
    verification.to_csv(CSV_OUT, index=False)

    patterns = pattern_analysis(df, history, all_leads)
    mw       = patterns["mw_summary"]
    insights = build_insights(mw, patterns)
    series   = build_forecast_series(df)

    # KDCA single-station forecast bias (separate dataset, additive)
    evo_hdd, evo_cdd = load_kdca_evolution()
    kdca = None
    if evo_hdd is not None:
        kdca = kdca_analysis(evo_hdd, evo_cdd)
        # Save extreme events to CSV for reference
        pd.concat(
            [kdca["extreme_cold"], kdca["extreme_hot"]],
            ignore_index=True
        ).to_csv(KDCA_CSV_OUT, index=False)

    # Blended GFS+ECE+GEFS forecast, walk-forward backtest (imported here: blend_model imports this module)
    from blend_model import print_blend_summary, run_blend, write_blend_outputs
    blend = run_blend(df)
    write_blend_outputs(blend)

    write_dashboard(mw, patterns, series, insights, kdca, blend)
    print_summary(mw, patterns, n_rows, kdca)
    print_blend_summary(blend)


if __name__ == "__main__":
    main()
