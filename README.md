# Weather Model Forecast Verification

How accurate are the major weather models at forecasting U.S. heating and cooling demand, and how far ahead can you trust them?

This project scores three numerical weather prediction models (**GFS**, **ECMWF Ensemble (ECE)** and **GEFS**) on their forecasts of U.S. gas-weighted degree days (GWDD). GWDD is the weather measure that drives natural gas demand. One Python script turns the raw forecast files into verification statistics and a self-contained interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![pandas](https://img.shields.io/badge/pandas-numpy-150458) ![Chart.js](https://img.shields.io/badge/Chart.js-dashboard-ff6384)

## Key findings

From 43,846 forecasts checked against what actually happened (Oct 2022 – May 2026):

| Forecast window | Best model | RMSE | Correlation | Within ±1 GWDD |
|---|---|---|---|---|
| 1–5 days | GEFS | 1.18 | 0.992 | 65% |
| 6–10 days | ECE | 2.15 | 0.970 | 44% |
| 11–15 days | ECE | 3.08 | 0.931 | 37% |

- **GFS forecasts too few degree days (too warm) in winter.** In Dec–Feb its bias ranges from −1.4 to −1.75 GWDD across all forecast windows, which matters most when gas demand peaks.
- **ECE gives the most stable forecasts.** Its run-to-run revisions average 0.17 GWDD, against 1.28 for GFS, so its signal is less noisy to trade on.
- **Extreme events are underestimated at longer leads.** At Washington Reagan (KDCA), ECE forecasts too many heating degree days for extreme cold out to day 5, then flips to too few from day 7 on, reaching −6.2 HDD by day 14. Extreme heat follows the same pattern.

## The dashboard

`weather_dashboard.html` is a single file with no external dependencies (Chart.js is inlined), so it opens offline in any browser. A PDF export is in [`weather_dashboard.pdf`](weather_dashboard.pdf).

Sections:
- **Model scorecard:** RMSE, MAE, bias, correlation and hit rate by model and window
- **Forecast vs actual:** time series of each model against what verified
- **Bias & seasonality:** bias by month and season
- **Convergence & revisions:** how forecasts change as the valid date gets closer
- **GFS vs ECE divergence:** what happens when the two models disagree
- **Regime performance:** accuracy in cold, normal and warm conditions
- **Bias by magnitude:** whether errors grow on high-demand days
- **KDCA extreme events:** single-station HDD/CDD bias by lead day for the coldest and hottest 10% of days
- **Market insights:** plain-language takeaways generated from the stats

## How it works

`weather_verification.py` runs the full pipeline:

1. **Load & normalize:** reads each model's forecast file, computes lead time from run date to valid date, and matches each forecast to observed GWDD
2. **Verification stats:** RMSE, MAE, bias, correlation and hit rates, grouped by model, forecast window, month and season
3. **Pattern analysis:** revisions, convergence, model divergence, weather regimes and bias by magnitude
4. **KDCA analysis:** turns 6-hourly ECE temperature forecasts into daily HDD/CDD, builds a matrix of what each run predicted for each day by lead time, and isolates extreme events
5. **Output:** writes the results CSVs, renders the HTML dashboard and prints a console summary

## Run it

```bash
pip install pandas numpy
python weather_verification.py
```

It runs in about 3 seconds and writes:

| File | Contents |
|---|---|
| `weather_dashboard.html` | Interactive dashboard |
| `model_verification_results.csv` | Stats for each model, window, season and month |
| `kdca_extreme_events.csv` | KDCA extreme cold and heat events with forecast error by lead day |

## Data

| File | Description |
|---|---|
| `GFS_GWUS.csv`, `ECE_GWUS.csv`, `GEFS_GWUS.csv` | Daily 00Z model runs: forecast GWDD for U.S. gas-weighted demand, up to 15 days out |
| `GWUS_history.csv` | Observed daily GWDD |
| `ECE_KDCA_t2m_20200101_20251231.csv` | ECE 2 m temperature forecasts for KDCA, 6-hourly steps, 2020–2025 |
