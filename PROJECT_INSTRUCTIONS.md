# ECE KDCA Weather Analysis — Project Instructions
**Location:** `C:/Users/JohnnyWang/PycharmProjects/ICE Weather Analysis/`
**Python:** `C:/Python/Python312/python`
**Author:** Johnny Wang

---

## 1. Project Overview

This project analyzes the forecast skill of the **ECMWF ECE (Ensemble mean) model** for **KDCA (Reagan National Airport, Washington DC)** using **Heating Degree Days (HDD)** as the primary metric. It is used to assess how well the model predicts cold weather 1–14 days in advance, specifically for natural gas trading decisions during the **XZFGH winter season**.

**XZFGH Convention** — standard energy futures month codes for winter:

| Code | Month |
|------|-------|
| X    | November |
| Z    | December |
| F    | January  |
| G    | February |
| H    | March    |

---

## 2. Key Concepts

### Heating Degree Days (HDD)
- **Base temperature:** 65°F
- **Formula:** `HDD_F = max(0, 65 - daily_mean_temp_F)`  where `temp_F = temp_C × 9/5 + 32`
- Source data is in °C; conversion to °F is applied in every script before the HDD calculation
- **HDD_F = HDD_C × 1.8 exactly** — algebraic identity, not an approximation
- Higher HDD = colder day = more heating demand
- A day with mean temp of 30°F produces `65 - 30 = 35 HDD`

### ECE Model
- **Model:** ECMWF Ensemble, 51-member mean (not individual runs)
- **Run time:** 00Z initialization only (6:00 PM CST / 7:00 PM EST)
- **Resolution:** 0.5 degrees (~55 km)
- **Output:** 6-hourly steps (0h, 6h, 12h, 18h, ..., up to 360h = 15 days)
- **Known bias:** The ECE ensemble mean has a **cold bias** — it consistently forecasts more HDD than what actually verifies, especially at long lead times (lead 10–14)

### Lead Time
- **Lead N** = the forecast issued exactly N days before the valid date
- **Lead 0** = same-day realization (the model's own Day-0 output, used as proxy for observed)
- **Lead 14** = forecast issued 14 days before the event
- **Lead 15 is excluded** — the 00Z run only produces step 360h (one data point with no 366/372/378h steps), making the daily mean unreliable. All analysis uses leads 0–14 only.

### Evolution Matrix
The core data structure. For each valid date, we look up what each model run said about that day:
- `evo[2025-01-15][lead_07]` = the HDD forecast for Jan 15 that was issued on Jan 8 (7 days earlier)
- Rows = valid dates | Columns = lead_00 to lead_14

---

## 3. File Inventory

```
ICE Weather Analysis/
├── ECE_KDCA_t2m_20200101_20251231.csv      ← SOURCE DATA (do not modify)
├── ECE_KDCA_daily_HDD_F_2020_2025.csv      ← CONVERTED HDD_F table (base 65°F)
├── ECE_KDCA_cold_distribution_60events.csv ← 60 cold events detail table
├── KDCA_Forecast_vs_Actual_AllLeads.png    ← Chart 1: Scatter (all leads)
├── KDCA_Cold_Distribution_Evolution.png    ← Chart 2: Cold event histograms
├── generate_csv_outputs.py                 ← Script: raw → HDD CSV
├── scatter_all_leads.py                    ← Script: generates Chart 1
└── update_charts.py                        ← Script: generates Chart 2
```

---

## 4. Source Data Format

**File:** `ECE_KDCA_t2m_20200101_20251231.csv`
- **Rows:** Forecast step in hours (0, 6, 12, 18, 24, ..., 360) — 61 rows
- **Columns:** Model initialization dates (YYYYMMDD format) — 2,192 columns
- **Values:** 2m temperature in °C (ensemble mean of 51 members)
- **Date range:** 2020-01-01 to 2025-12-31

**How steps map to lead days:**

| Lead Day | Steps used |
|----------|------------|
| Lead 0   | 0h, 6h, 12h, 18h |
| Lead 1   | 24h, 30h, 36h, 42h |
| Lead N   | N×24h, N×24+6h, N×24+12h, N×24+18h |
| Lead 14  | 336h, 342h, 348h, 354h |
| Lead 15  | 360h only (EXCLUDED — incomplete) |

Each day's HDD = average of the available steps, then `max(0, 18.333 - mean_temp)`.

**Missing data:** 2 initialization dates are completely missing (2024-02-23, 2024-05-12); 29 more have partial steps. Python's `nanmean` silently uses available points. XZFGH impact: ~0.25% of cells.

---

## 5. Scripts — How to Run

**Always run from the project directory:**
```
cd "C:/Users/JohnnyWang/PycharmProjects/ICE Weather Analysis"
```

### Script 1: `generate_csv_outputs.py`
**Purpose:** Convert raw temperature source data to HDD CSV.
**Run when:** New source data is added.
```
C:/Python/Python312/python generate_csv_outputs.py
```
**Output:** `ECE_KDCA_daily_HDD_F_2020_2025.csv`
**Expected output on console:**
```
Init dates : 2020-01-01 to 2025-12-31  (2192 days)
Shape : (16, 2192)  (rows=lead_day, cols=init_dates)
DONE — 1 CSV file generated: ECE_KDCA_daily_HDD_F_2020_2025.csv
```

### Script 2: `update_charts.py`
**Purpose:** Generate the Cold Distribution Evolution chart.
**Run when:** Source data changes, or chart settings are adjusted.
```
C:/Python/Python312/python update_charts.py
```
**Output:** `KDCA_Cold_Distribution_Evolution.png`
**Expected console output:**
```
Cold events selected: 60
Saved: KDCA_Cold_Distribution_Evolution.png
```
> **Check:** The number of cold events should be **60**. If it changes, the source data has changed.

### Script 3: `scatter_all_leads.py`
**Purpose:** Generate the Forecast vs Actual scatter chart (all leads).
**Run when:** Source data changes, or chart settings are adjusted.
```
C:/Python/Python312/python scatter_all_leads.py
```
**Output:** `KDCA_Forecast_vs_Actual_AllLeads.png`
**Expected console output:**
```
Saved: KDCA_Forecast_vs_Actual_AllLeads.png
```

---

## 6. Chart Explanations & Verification Checklist

### Chart 1: `KDCA_Forecast_vs_Actual_AllLeads.png`

**What it shows:** 15 scatter plots (3 rows × 5 columns), one per lead time (Lead 14 → Day-0).
- **X-axis:** ECE model's HDD forecast issued N days before the valid date
- **Y-axis:** What actually realized on that day (Lead-0 HDD)
- **Each dot:** One XZFGH winter day (2020–2025)
- **Grey dashed line:** Perfect forecast (x = y)
- **Dark red line:** Best-fit regression

**Per-panel statistics:**
- **r** — Pearson correlation of forecast vs realized. Measures skill: how well the model rank-orders cold/warm days.
- **bias** — mean(realized − forecast). Negative = model too cold (over-forecasts HDD).
- **RMSE** — √mean((realized − forecast)²). Average error in HDD units.

**What to expect (sanity checks):**
- r should **increase** as lead decreases (Lead 14 ≈ 0.66, Lead 0 ≈ 1.00)
- bias should be **negative** at all leads (cold bias), approaching 0 near Day-0
- RMSE should **decrease** as lead decreases (Lead 14 ≈ 7.4 HDD°F, Lead 0 ≈ 0)
- The dark red regression line should fall **below** the grey dashed x=y line at most leads
- All 15 panels should be populated (no blank panels)
- Lead 15 should **not** appear anywhere

**Red flags:**
- r > 0.90 at Lead 14 → something is wrong (data overlap or look-ahead bias)
- Positive bias at long leads → would contradict the known cold bias
- Blank panel or missing panel → data loading issue

---

### Chart 2: `KDCA_Cold_Distribution_Evolution.png`

**What it shows:** 15 histogram panels (3 rows × 5 columns), tracking the **same 60 cold events** at every lead from Lead 14 to Day-0.

**How the 60 events are selected:**
1. Take all XZFGH days that have Lead-14 data
2. Compute model's own Lead-14 monthly mean and std (grouped by month)
3. Select days where `lead_14_HDD − model_monthly_mean > 1.5 × model_monthly_std`
4. This identifies days the model flagged as significantly cold 14 days out
5. Result: **60 events** (uses model mean as threshold, not observed, because model mean > observed mean due to cold bias → stricter threshold → more extreme cold events)

**X-axis:** HDD anomaly relative to the **observed** monthly normal (pooled 2020–2025 per month)
- X = 0 (dashed line) = monthly normal
- X > 0 = colder than normal
- X < 0 = warmer than normal

**Panel stats box:**
- `n=60` — all 60 events have complete data at every lead
- `>normal: XX%` — fraction of the 60 events still showing above-normal HDD at that lead

**What to expect (sanity checks):**
- Lead 14: `>normal` should be **100%** — these events were selected for being cold at lead 14
- `>normal` should **decrease** as lead approaches Day-0 (regression to mean)
- Day-0 (realized): `>normal` should be around **75–80%** — some events regress to normal
- All panels should show **n=60**
- Color should shift from **blue** (Lead 14, far out) to **red** (Day-0, realized)

**Red flags:**
- `Cold events selected:` prints anything other than **60** → data or threshold changed
- Lead 14 `>normal` is not 100% → selection logic broken
- n < 60 in any panel → some events are missing data at that lead

---

## 7. The 60 Cold Events CSV: `ECE_KDCA_cold_distribution_60events.csv`

This file contains the raw data behind Chart 2. Each row = one cold event day.

**Key columns:**

| Column | Description |
|--------|-------------|
| `valid_date` | The date in question |
| `month` | Month name |
| `obs_monthly_mean_hdd` | Observed HDD monthly normal (2020–2025 pooled) |
| `lead14_hdd` through `lead00_hdd` | Raw HDD forecast at each lead |
| `lead14_hdd_anom` through `lead00_hdd_anom` | HDD anomaly vs observed monthly mean (plotted in chart) |
| `realized_hdd_lead0` | Day-0 realization |

---

## 8. Known Limitations

| Limitation | Detail |
|------------|--------|
| 00Z only | No 12Z cross-check. Data is ~11 hours old by the time CWG delivers it |
| Ensemble mean compression | 51-member average suppresses extremes; cold extremes are underforecast |
| Cold bias | Model systematically over-predicts HDD, especially at leads 10–14 |
| Lead-15 excluded | Only step 360h exists (no 6h/12h/18h companions); excluded from all analysis |
| Missing init dates | 2024-02-23 and 2024-05-12 are completely absent from the source data |
| nanmean behavior | Partial step days (3 of 4 steps available) are silently averaged without warning |
| No bias correction | Raw model output is used; no post-processing adjustment applied |

---

## 9. Updating for New Data

When new source data arrives (e.g., adding 2026):

1. Replace `ECE_KDCA_t2m_20200101_20251231.csv` with the updated file
2. Update the filename reference in **all 3 scripts** (search for `20251231`)
3. Run all 3 scripts in order:
   ```
   python generate_csv_outputs.py
   python update_charts.py
   python scatter_all_leads.py
   ```
4. Verify console output: cold events count should still print (may differ slightly with new data)
5. Verify both charts open and look reasonable using the sanity checks in Section 6

---

## 10. Quick Verification Commands

Run all 3 scripts and check outputs in one go:
```
cd "C:/Users/JohnnyWang/PycharmProjects/ICE Weather Analysis"
C:/Python/Python312/python generate_csv_outputs.py
C:/Python/Python312/python update_charts.py
C:/Python/Python312/python scatter_all_leads.py
```

**Expected final file list:**
```
ECE_KDCA_t2m_20200101_20251231.csv      (source, ~654 KB, do not touch)
ECE_KDCA_daily_HDD_F_2020_2025.csv      (generated, ~465 KB, HDD in °F)
ECE_KDCA_cold_distribution_60events.csv (reference data, ~14 KB, HDD in °F)
KDCA_Cold_Distribution_Evolution.png    (chart, ~160 KB)
KDCA_Forecast_vs_Actual_AllLeads.png    (chart, ~1.2 MB)
generate_csv_outputs.py
scatter_all_leads.py
update_charts.py
```
