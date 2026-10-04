"""
Blended GWDD forecast: combines GFS, ECE and GEFS into one forecast.

Per lead day, fits non-negative model weights (summing to 1) plus a seasonal
bias correction. Every score is out-of-sample: weights are refit monthly on
an expanding window that only sees forecasts whose valid date has already
passed, then applied to the following month's runs.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from weather_verification import (
    WINDOWS, WORKSPACE, assign_window, compute_stats, half_year, load_data,
)

SINGLE_MODELS = ["GFS", "ECE", "GEFS"]
METHODS       = ["EQUAL", "INV_MSE", "WEIGHTED_BC", "WEIGHTED_BC_SEASONAL"]
GRID_STEP     = 0.05
MIN_TRAIN_MONTHS = 12

RESULTS_OUT   = WORKSPACE / "blend_results.csv"
FORECASTS_OUT = WORKSPACE / "blend_forecasts.csv"
WEIGHTS_OUT   = WORKSPACE / "blend_weights.csv"


def models_for_lead(lead: int) -> list[str]:
    # ECE runs stop at 14 days; lead 15 falls back to GFS + GEFS
    return SINGLE_MODELS if lead <= 14 else ["GFS", "GEFS"]


# ---------- Section 1: wide forecast table ----------

def build_wide(df: pd.DataFrame) -> pd.DataFrame:
    wide = (df.pivot_table(index=["RunDate", "ForecastDate", "LeadDays"],
                           columns="Model", values="GWDD")
              .reset_index())
    actual = df.groupby("ForecastDate")["Actual"].first()
    wide["Actual"] = wide["ForecastDate"].map(actual)
    for m in SINGLE_MODELS:
        if m not in wide.columns:
            wide[m] = np.nan

    keep = np.zeros(len(wide), dtype=bool)
    for lead in range(1, 16):
        req = models_for_lead(lead)
        keep |= (wide["LeadDays"] == lead).to_numpy() & wide[req].notna().all(axis=1).to_numpy()
    wide = wide[keep].copy()
    wide["HalfYear"] = wide["ForecastDate"].dt.month.map(half_year)
    wide["Window"]   = wide["LeadDays"].map(assign_window)
    return wide.sort_values(["RunDate", "LeadDays"]).reset_index(drop=True)


# ---------- Section 2: per-lead fitting ----------

def simplex_grid(k: int, step: float = GRID_STEP) -> np.ndarray:
    n = int(round(1 / step))
    pts = [c for c in product(range(n + 1), repeat=k) if sum(c) == n]
    return np.array(pts, dtype=float) / n


def fit_lead(train: pd.DataFrame, models: list[str], method: str) -> dict:
    X = train[models].to_numpy()
    y = train["Actual"].to_numpy()
    k = len(models)

    if method == "EQUAL":
        return {"w": np.full(k, 1 / k), "b": {"Summer": 0.0, "Winter": 0.0}}

    if method == "INV_MSE":
        inv = 1 / ((X - y[:, None]) ** 2).mean(axis=0)
        return {"w": inv / inv.sum(), "b": {"Summer": 0.0, "Winter": 0.0}}

    W = simplex_grid(k)
    R = y[:, None] - X @ W.T          # residual for every candidate weight vector

    if method == "WEIGHTED_BC":
        best = int(R.var(axis=0).argmin())
        b = float(R[:, best].mean())
        return {"w": W[best], "b": {"Summer": b, "Winter": b}}

    # WEIGHTED_BC_SEASONAL: separate intercept per half-year
    halves = train["HalfYear"].to_numpy()
    sse = np.zeros(W.shape[0])
    for h in ("Summer", "Winter"):
        Rh = R[halves == h]
        if len(Rh):
            sse += ((Rh - Rh.mean(axis=0)) ** 2).sum(axis=0)
    best = int(sse.argmin())
    overall = float(R[:, best].mean())
    b = {h: float(R[halves == h, best].mean()) if (halves == h).any() else overall
         for h in ("Summer", "Winter")}
    return {"w": W[best], "b": b}


def predict(rows: pd.DataFrame, models: list[str], fit: dict) -> np.ndarray:
    base = rows[models].to_numpy() @ fit["w"]
    return base + rows["HalfYear"].map(fit["b"]).to_numpy()


# ---------- Section 3: walk-forward backtest ----------

def walk_forward(wide: pd.DataFrame) -> pd.DataFrame:
    months = wide["RunDate"].dt.to_period("M")
    first_test = months.min() + MIN_TRAIN_MONTHS
    out = []
    for month in sorted(months[months >= first_test].unique()):
        cutoff = month.start_time
        test  = wide[months == month].copy()
        # Only forecasts whose valid date has passed are known at the cutoff
        train = wide[wide["ForecastDate"] < cutoff]
        assert train["ForecastDate"].max() < test["RunDate"].min(), "look-ahead leak"

        for lead, t in test.groupby("LeadDays"):
            models = models_for_lead(lead)
            tr = train[train["LeadDays"] == lead]
            for method in METHODS:
                test.loc[t.index, method] = predict(t, models, fit_lead(tr, models, method))
        out.append(test)
    return pd.concat(out, ignore_index=True)


def add_revisions(bt: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # Chg1 = change vs the previous day's run for the same valid date
    bt = bt.sort_values(["ForecastDate", "RunDate"]).copy()
    g = bt.groupby("ForecastDate")
    consecutive = g["RunDate"].diff().dt.days == 1
    for c in cols:
        bt[f"{c}_Chg1"] = g[c].diff().where(consecutive)
    return bt


def score(bt: pd.DataFrame, cols: list[str], keys: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        sub = bt.dropna(subset=[c])
        frame = pd.DataFrame({
            "GWDD": sub[c], "Actual": sub["Actual"],
            "Error": sub[c] - sub["Actual"], "Chg1": sub[f"{c}_Chg1"],
        })
        for k in keys:
            frame[k] = sub[k]
        groups = frame.groupby(keys, sort=True) if keys else [((), frame)]
        for vals, g in groups:
            vals = vals if isinstance(vals, tuple) else (vals,)
            rows.append({"Model": c, **dict(zip(keys, vals)), **compute_stats(g)})
    return pd.DataFrame(rows)


# ---------- Section 4: public entry point ----------

def run_blend(df: pd.DataFrame) -> dict:
    wide = build_wide(df)
    bt   = walk_forward(wide)

    overall_rmse = {m: float(np.sqrt(((bt[m] - bt["Actual"]) ** 2).mean())) for m in METHODS}
    best = min(overall_rmse, key=overall_rmse.get)
    bt["BLEND"] = bt[best]

    cols = SINGLE_MODELS + METHODS + ["BLEND"]
    bt = add_revisions(bt, cols)
    window_stats = score(bt, cols, ["Window"])
    window_stats["Window"] = pd.Categorical(window_stats["Window"], WINDOWS, ordered=True)
    window_stats = window_stats.sort_values(["Window", "Model"]).reset_index(drop=True)
    lead_stats = score(bt, SINGLE_MODELS + ["BLEND"], ["LeadDays"])

    # Current production weights: winning method fit on all verified history
    weights = []
    for lead in range(1, 16):
        models = models_for_lead(lead)
        fit = fit_lead(wide[wide["LeadDays"] == lead], models, best)
        w = dict.fromkeys(SINGLE_MODELS, 0.0) | dict(zip(models, map(float, fit["w"])))
        weights.append({"LeadDays": lead, **{f"w_{m}": w[m] for m in SINGLE_MODELS},
                        "Bias_Summer": fit["b"]["Summer"], "Bias_Winter": fit["b"]["Winter"]})

    return {
        "best": best,
        "overall_rmse": overall_rmse,
        "window_stats": window_stats,
        "lead_stats": lead_stats,
        "weights": pd.DataFrame(weights),
        "backtest": bt,
        "test_start": bt["RunDate"].min().strftime("%Y-%m-%d"),
        "test_end":   bt["RunDate"].max().strftime("%Y-%m-%d"),
        "n_test": int(len(bt)),
    }


def write_blend_outputs(res: dict) -> None:
    pd.concat([res["window_stats"].assign(GroupType="Window"),
               res["lead_stats"].assign(GroupType="Lead")],
              ignore_index=True).to_csv(RESULTS_OUT, index=False)
    fc_cols = ["RunDate", "ForecastDate", "LeadDays", "Window", *SINGLE_MODELS, "BLEND", "Actual"]
    (res["backtest"][fc_cols].sort_values(["RunDate", "LeadDays"])
        .to_csv(FORECASTS_OUT, index=False, float_format="%.3f"))
    res["weights"].to_csv(WEIGHTS_OUT, index=False, float_format="%.3f")


def print_blend_summary(res: dict) -> None:
    ws = res["window_stats"]
    print("-" * 68)
    print(f"BLENDED MODEL (out-of-sample, runs {res['test_start']} to {res['test_end']}, "
          f"N={res['n_test']:,})")
    print("  Method RMSE (all leads): " +
          "  ".join(f"{m}={v:.3f}" for m, v in res["overall_rmse"].items()))
    print(f"  Selected: {res['best']}")
    print(f"\n  {'Window':<8}{'Best single':<18}{'BLEND':>8}{'Improve':>10}{'Hit<=1':>9}")
    for w in WINDOWS:
        sub = ws[ws["Window"] == w].set_index("Model")
        single = sub.loc[SINGLE_MODELS, "RMSE"]
        bm, br = single.idxmin(), single.min()
        blend = sub.loc["BLEND"]
        print(f"  {w:<8}{bm:<6}RMSE={br:<7.3f}{blend['RMSE']:>8.3f}"
              f"{(br - blend['RMSE']) / br * 100:>9.1f}%{blend['HitRate1']:>8.1f}%")
    print("\n  Current weights by lead (GFS/ECE/GEFS, winter bias adj):")
    for _, r in res["weights"].iterrows():
        print(f"    L{int(r['LeadDays']):>2}  {r['w_GFS']:.2f}/{r['w_ECE']:.2f}/{r['w_GEFS']:.2f}"
              f"  {r['Bias_Winter']:+.2f}")
    print(f"  Outputs: {RESULTS_OUT.name}, {FORECASTS_OUT.name}, {WEIGHTS_OUT.name}")


def main() -> None:
    df, _, _ = load_data()
    res = run_blend(df)
    write_blend_outputs(res)
    print_blend_summary(res)


if __name__ == "__main__":
    main()
