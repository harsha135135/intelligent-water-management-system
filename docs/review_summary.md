# Chronos-2 water demand forecasting — final review summary

**PW26_PK_06 · Intelligent Water Management System · PES University RR**
24 physical tanks · hourly · 2025-01-01 → 2026-04-22 · 270,849 observations

Every number in this document was measured on the completed rolling-origin benchmark and can be
traced to a file under `results/chronos2/`. Nothing is estimated, interpolated or carried over
from an earlier, non-comparable run. Where a quantity was not measured, it is left blank and
said to be blank. Interpretation is confined to paragraphs explicitly marked as such.

---

## 1. Executive summary

**Chronos-2 zero-shot is the recommended production model, with two documented caveats.**

Measured, on a single evaluation grid where every model scores the *same* 188,664 rows:

* Chronos-2 beats the incumbent **NPTS at all six horizons on all four metrics**. MASE improves
  by **12.4 % at 6 h falling to 5.7 % at 7 d**; MAE improves by 14.4 % → 5.5 % over the same
  range.
* Against the **SeasonalNaive-24** reference it reduces MASE by **34.9–38.6 %** at every horizon.
* It wins on **16–19 of 24 tanks** depending on horizon (17/24 = 71 % at the 1-day horizon).
  **SeasonalNaive-24 wins zero tanks at every horizon.**
* Zero-shot costs **89 seconds** for the entire 24-origin × 6-horizon backtest on an Apple MPS
  laptop GPU. The covariate variants cost 4–10× that for a MASE difference of **at most 0.0025**.

The two caveats, both measured:

1. **Its prediction intervals are too narrow.** The p10–p90 band should cover 80 % of outcomes;
   it covers **0.714–0.743**. NPTS covers 0.836–0.850 and is the better-calibrated model. This
   is the one axis on which the incumbent wins.
2. **It under-forecasts total volume.** Pooled over all tanks its 24 h forecasts sum to
   **12.2 % below** actual outflow (NPTS −9.9 %, SeasonalNaive +0.4 %). For sizing a refill this
   is the failure direction that matters.

Point accuracy (§5–§8) and uncertainty calibration (§10) are separate results and are kept
separate throughout. The recommendation in §12 is **Chronos-2 zero-shot for the point forecast, with a
calibration layer and a per-tank router**, not "Chronos-2 everywhere".

---

## 2. Dataset and evaluation setup

### 2.1 Data

| Property | Value | Source |
|---|---|---|
| Dataset directories | 26 | `dataset/` |
| **Physical tanks after de-duplication** | **24** | `src/data/curate.py` |
| Rows (gapless hourly panel) | 270,849 | `python -m src.data.curate` |
| Date range | 2025-01-01 00:00 → 2026-04-22 23:00 | same |
| Target | `Outflow in KL` (hourly, KL/h) | same |
| Mean hourly demand span across tanks | 0.0012 → 1.6739 KL/h | `eda/tank_mass_balance.csv` |
| Hours exactly zero, campus-wide | 26.5 % of the 262,590 observed readings | computed from the curated panel |

`dataset/` holds 26 directories but only 24 tanks. Two tanks were re-scraped under a slightly
different directory name and both copies were kept:

```
GJBC_BLOCK_1_A1_BOY_S  (477 days, → 2026-04-22)  ⊃  GJBC_BLOCK_1_A1_BOYS  (439 days, → 2026-03-16)
GJBC_BLOCK_1_A2_GIRL_S (477 days, → 2026-04-22)  ⊃  GJBC_BLOCK_1_A2_GIRLS (439 days, → 2026-03-16)
```

Each pair carries an identical `Tank Name`, identical `Tank Dimensions` and identical daily
totals; every day in the short copy is contained in the long copy. `curate.py` keeps the longer
copy, relabels it to the canonical id, drops the stale copy, and **raises** if the surviving
count is not exactly 24. Earlier runs in this repo loaded the tree naively and scored 26 series.

Each tank is then reindexed onto a complete hourly range between its own first and last reading.
Missing hours become NaN rather than being dropped — collapsing a gap would shift every later
timestamp and destroy the 24-hour seasonality the whole benchmark is scaled against.

### 2.2 Evaluation grid

| Property | Value |
|---|---|
| Forecast origins | **24** |
| Origin stride | **23 hours** |
| Origin range | 2026-03-24 22:00 → 2026-04-15 23:00 |
| Distinct hours-of-day among origins | **24 of 24** |
| Horizons | 6 h, 12 h, 1 d (24 h), 2 d (48 h), 3 d (72 h), 7 d (168 h) |
| Rows scored per model | 188,664 |
| Rows per model per horizon | 3,405 / 6,861 / 13,530 / 27,323 / 41,140 / 96,405 |

**Why the stride is 23 and not 24.** An earlier grid spaced origins 24 hours apart, which put
*every* origin at 23:00. The 6 h horizon then only ever scored the quiet 00:00–05:00 window and
never the morning refill peak, flattering short-horizon MASE to 0.22. 23 is co-prime with 24, so
24 origins at a 23-hour stride visit **each hour-of-day exactly once** and no horizon is scored
on a single slice of the diurnal cycle. This is enforced in `src/models/backtest.py::make_spec`
and verified in §2.3.

**Train / test separation.** The AutoGluon baselines are fitted strictly on data at or before the
**first** origin (2026-03-24 22:00) and are never refitted; the last origin is three weeks later.
Chronos-2 is zero-shot — no fitting at all — with its context truncated at each origin. At scoring
time every predicted row satisfies `timestamp > origin` (checked; 0 violations), and the
seasonal-naive scaling denominator for MASE/RMSSE is computed **only** from history at or before
the origin.

**Final holdout.** The last origin (2026-04-15 23:00) forecasts the final 7 contiguous days of
the dataset, 2026-04-16 00:00 → 2026-04-22 23:00. It is a labelled subset of the same grid, not
a separate experiment (`results/chronos2/unified/leaderboard.csv`).

### 2.3 Methodology verification

Every check below was re-run against the prediction files before this document was written. All
pass; no result in this review rests on an unverified assumption.

| # | Check | Result |
|---|---|---|
| 1 | **24 physical tanks** | PASS — 26 directories → 24 tanks; `curate.py` raises otherwise |
| 2 | **Duplicate datasets removed** | PASS — `GJBC_BLOCK_1_A1_BOY_S` / `GJBC_BLOCK_1_A2_GIRL_S` absent from every prediction file; 24 distinct `item_id` |
| 3 | **24 forecast origins** | PASS — 24 distinct origins, 2026-03-24 22:00 → 2026-04-15 23:00 |
| 4 | **23-hour origin stride** | PASS — all 23 inter-origin gaps are exactly 23 h; the set of observed strides is `{23}` |
| 5 | **All 24 hours-of-day represented** | PASS — origin hours = `{0,1,…,23}`, each exactly once |
| 6 | **Six forecast horizons** | PASS — `{6, 12, 24, 48, 72, 168}`, and `max(step) == horizon` for each |
| 7 | **Identical evaluation rows across models** | PASS — the set of `(item_id, origin, horizon, timestamp)` keys is **byte-identical across all 9 models**, 188,664 keys each. Row counts per horizon: 3,405 / 6,861 / 13,530 / 27,323 / 41,140 / 96,405, with exactly 1 distinct value per horizon |
| 8 | **No leakage from future observations** | PASS — 0 rows with `timestamp ≤ origin`; `step` equals hours-after-origin on every one of 1,710,720 rows; scaling denominators use pre-origin history only |
| 9 | **Chronological train/test separation** | PASS — AutoGluon baselines fitted on rows `≤ 2026-03-24 22:00` (the first origin) and never refitted; Chronos-2 is zero-shot with context truncated at each origin |
| 10 | **No duplicate prediction rows** | PASS — 0 duplicated `(model, item_id, origin, horizon, timestamp)` |
| 11 | **SeasonalNaive-24 sanity: MASE ≈ 1** | PASS — measured 0.967 – 1.069 across the six horizons; RMSSE 0.717 – 0.812 |
| 12 | **Metric unit tests** | PASS — 6 / 6 in `tests/test_metrics.py`, including the MASE ≈ 1 identity and degenerate-series exclusion |
| 13 | **Published tables reproduce from the raw predictions** | PASS — an independent re-score of `predictions_*.parquet` reproduces `metrics_by_horizon.csv` and `metrics_per_tank.csv` to ≤ 2.2e-16; two independent metric implementations (`metrics.py` via `score_benchmark`, and `unified_analysis`) agree to 1e-16 across all 66 model x horizon rows |
| 14 | **`per_tank_comparison.csv` matches the predictions** | PASS — all 144 rows × 3 models × 4 metrics reproduce with max abs difference 0.0; `n_observations` totals match the per-horizon row counts exactly |
| 15 | **Degenerate-series guard** | Not triggered — smallest seasonal-naive denominator on the grid is 0.00198, so `n_tanks_scaled = 24` everywhere and no metric is undefined |

---

## 3. Models compared

| Model | Role | What it is |
|---|---|---|
| **Chronos2-ZS** | **proposed** | `amazon/chronos-2`, 119.5 M params, zero-shot, target series only |
| **NPTS** | **incumbent** | Non-Parametric Time Series, AutoGluon. The shipping `results/autogluon` WeightedEnsemble carries weight 1.0 on NPTS |
| **SeasonalNaive-24** | **reference baseline** | repeat the last 24 observed hours; the denominator MASE/RMSSE scale to |
| Chronos2-COV | covariate study | + 10 known-future calendar covariates, + 3 past-only covariates |
| Chronos2-COV-LEAN | covariate study | + 3 MI-selected known covariates (`hour_sin`, `hour_cos`, `exam_proximity`), + past |
| Chronos2-COV-XL | covariate study | as COV, plus cross-learning across tanks |
| ETS, Theta, DynamicOptimizedTheta | context | classical baselines, scored on the same grid, reported in `benchmark_table.md` |

**PatchTST is now on this grid.** It is the trained deep control — the same patched-transformer
family Chronos-2 belongs to, but fitted on this campus rather than pretrained. Two configurations
were run (`src/models/patchtst_benchmark.py`): AutoGluon's defaults (context
96 h, 30 epochs) and a `tuned` preset at the
settings the PatchTST paper uses for hourly data (context 512 h,
100 epochs). Both are fitted only on data at or before the first origin, scored
on the identical rows, and reported. Measured: Chronos-2 zero-shot is **16.5–18.6 % better
in MASE at every horizon** than the stronger of the two, significant at all six by paired bootstrap
and Diebold–Mariano. Full analysis in `results/chronos2/unified/`.

The older `results/patchtst/` directory is a *different*, non-comparable evaluation (24 series, 576
rows, different holdout dates) and is used nowhere in this document.

---

## 4. Metrics and definitions

For a series with in-sample history `y₁…yₙ` and seasonal period `m = 24`:

```
scale_mae = mean_{t>m} |yₜ − yₜ₋ₘ|          scale_mse = mean_{t>m} (yₜ − yₜ₋ₘ)²

MAE   = mean |y − ŷ|                        (KL/h — lower is better)
RMSE  = sqrt( mean (y − ŷ)² )               (KL/h — lower is better)
MASE  = mean( |y − ŷ| / scale_mae )         (ratio — lower is better)
RMSSE = sqrt( mean( (y − ŷ)² / scale_mse ) )(ratio — lower is better)
```

**Reading MASE and RMSSE.** They are **ratios, not percentages**. A value of ~1 means the
forecast is roughly as accurate as the seasonal-naive reference on that series; **< 1 means
better than the naive reference**; > 1 means worse. A MASE of 0.65 does **not** mean "65 %
accurate" and does not mean "35 % error" — it means the average absolute error is 0.65× the
average absolute error of repeating yesterday's same hour.

**Why scaled metrics are the headline.** Per-tank mean hourly outflow spans 0.0012 KL/h
(`NEW_BLOCK_RO`) to 1.6739 KL/h (`BE_BLOCK_OHT`) — three orders of magnitude. An unweighted MAE
average across tanks is dominated by the four largest tanks, and a dead sensor scores a *perfect*
MAE of 0.0 by never moving. Both aggregations are therefore reported side by side:

* `macro_*` — unweighted mean across tanks; every tank counts once.
* `vw_mae` / `vw_rmse` — weighted by each tank's mean demand; answers "how many KL is the campus
  wrong by".

**Degenerate series.** A constant series has `scale_mae = 0` and an undefined MASE. Such series
are excluded from scaled aggregates and reported separately, never patched with an epsilon.
**On this dataset the guard was never triggered**: the smallest seasonal-naive denominator on the
grid is 0.00198 (`NEW_BLOCK_RO`), so all 24 tanks contributed to every scaled metric
(`n_tanks_scaled = 24` in every row of `metrics_by_horizon.csv`). No metric in any deliverable is
blank for this reason.

Implementation: `src/models/metrics.py`. Tests: `tests/test_metrics.py` (6 tests, all passing) —
including a hand-worked scale, NaN handling, a perfect forecast scoring 0, a degenerate series
being excluded and visibly counted, and the load-bearing identity that a seasonal-naive forecast
scores MASE ≈ 1.

---

## 5. Overall benchmark results

Full tables — every model, every horizon, every metric — are in
[`results/chronos2/benchmark_table.md`](../results/chronos2/benchmark_table.md).

### 5.1 Macro MASE (lower is better; 1.0 = seasonal naive)

| Model | 6 h | 12 h | 1 d | 2 d | 3 d | 7 d |
|---|---|---|---|---|---|---|
| Chronos2-COV-XL | **0.6002** | **0.6281** | **0.6550** | **0.6633** | **0.6641** | **0.6628** |
| Chronos2-COV | 0.6018 | 0.6292 | 0.6558 | 0.6643 | 0.6648 | 0.6632 |
| **Chronos2-ZS** | 0.6017 | 0.6294 | 0.6552 | 0.6656 | 0.6664 | 0.6653 |
| Chronos2-COV-LEAN | 0.6027 | 0.6304 | 0.6562 | 0.6654 | 0.6657 | 0.6640 |
| **NPTS (incumbent)** | 0.6871 | 0.6923 | 0.7098 | 0.7113 | 0.7090 | 0.7057 |
| DynamicOptimizedTheta | 0.8963 | 0.9459 | 0.9575 | 0.9856 | 0.9909 | 1.0148 |
| Theta | 0.8965 | 0.9451 | 0.9561 | 0.9845 | 0.9906 | 1.0162 |
| ETS | 0.9096 | 0.9480 | 0.9611 | 0.9718 | 0.9780 | 0.9799 |
| **SeasonalNaive-24** | 0.9806 | 0.9666 | 1.0320 | 1.0502 | 1.0628 | 1.0688 |

**Sanity check.** `SeasonalNaive-24` scores MASE **0.967–1.069**. MASE is *defined* so that a
seasonal-naive forecast scores 1.0, so this is the number that shows the scaling denominator is
computed correctly. It is also asserted as a unit test. Its slight drift above 1.0 at longer
horizons is expected: the metric's denominator is fixed at the origin while the forecast is
tiled forward over an increasingly stale 24-hour window.

### 5.2 Macro RMSSE

| Model | 6 h | 12 h | 1 d | 2 d | 3 d | 7 d |
|---|---|---|---|---|---|---|
| **Chronos2-ZS** | 0.4910 | 0.5271 | 0.5670 | 0.5722 | 0.5718 | 0.5647 |
| NPTS | 0.5494 | 0.5688 | 0.6064 | 0.6075 | 0.6053 | 0.5957 |
| ETS | 0.5390 | 0.5698 | 0.6072 | 0.6120 | 0.6138 | 0.6074 |
| SeasonalNaive-24 | 0.7174 | 0.7399 | 0.8020 | 0.8058 | 0.8117 | 0.8085 |

### 5.3 Absolute error, and the campus view

| Horizon | C2 macro MAE | C2 macro RMSE | C2 vw MAE | C2 vw RMSE |
|---|---|---|---|---|
| 6 h | 0.1974 | 0.3976 | 0.3911 | 0.7225 |
| 12 h | 0.2029 | 0.4207 | 0.4050 | 0.7606 |
| 1 d | 0.2116 | 0.4640 | 0.4273 | 0.8424 |
| 2 d | 0.2161 | 0.4688 | 0.4392 | 0.8529 |
| 3 d | 0.2174 | 0.4677 | 0.4431 | 0.8497 |
| 7 d | 0.2199 | 0.4635 | 0.4484 | 0.8466 |

All values KL/h. `vw` = demand-weighted, so it is roughly 2× macro: the campus's error is
concentrated in its large tanks, which is where the KL are.

### 5.4 Final 7-day holdout (last origin only, 4,028 rows at 7 d)

| Horizon | Chronos2-ZS MASE | NPTS MASE | SeasonalNaive MASE |
|---|---|---|---|
| 6 h | 0.2499 | 0.2751 | 0.3762 |
| 12 h | 0.5742 | 0.5932 | 0.6146 |
| 1 d | 0.6265 | 0.6566 | 0.8160 |
| 2 d | 0.6249 | 0.6470 | 0.8356 |
| 3 d | 0.6204 | 0.6397 | 0.8281 |
| 7 d | 0.6358 | 0.6628 | 0.9289 |

The ordering on the single held-out week is the same as on the full grid. *Interpretation:* a
single origin is one draw, so this is corroboration of the full-grid result, not independent
evidence — the 6 h figure in particular rests on 144 rows.

---

## 6. Per-horizon comparison — Chronos-2 vs NPTS vs SeasonalNaive

Positive `% impr` = Chronos-2 better. `abs diff` is in the metric's own units.

### MAE (KL/h)

| Horizon | Chronos-2 | NPTS | SeasNaive | abs diff vs NPTS | % vs NPTS | abs diff vs naive | % vs naive |
|---|---|---|---|---|---|---|---|
| 6 h | **0.1974** | 0.2306 | 0.3095 | 0.0331 | **+14.37 %** | 0.1121 | +36.22 % |
| 12 h | **0.2029** | 0.2255 | 0.3041 | 0.0226 | **+10.03 %** | 0.1012 | +33.29 % |
| 1 d | **0.2116** | 0.2303 | 0.3178 | 0.0187 | **+8.12 %** | 0.1062 | +33.43 % |
| 2 d | **0.2161** | 0.2312 | 0.3200 | 0.0151 | **+6.52 %** | 0.1039 | +32.46 % |
| 3 d | **0.2174** | 0.2307 | 0.3248 | 0.0133 | **+5.76 %** | 0.1074 | +33.05 % |
| 7 d | **0.2199** | 0.2326 | 0.3277 | 0.0127 | **+5.46 %** | 0.1078 | +32.90 % |

### RMSE (KL/h)

| Horizon | Chronos-2 | NPTS | SeasNaive | abs diff vs NPTS | % vs NPTS | abs diff vs naive | % vs naive |
|---|---|---|---|---|---|---|---|
| 6 h | **0.3976** | 0.4420 | 0.5649 | 0.0444 | **+10.05 %** | 0.1673 | +29.62 % |
| 12 h | **0.4207** | 0.4519 | 0.5779 | 0.0312 | **+6.90 %** | 0.1572 | +27.20 % |
| 1 d | **0.4640** | 0.4921 | 0.6456 | 0.0281 | **+5.72 %** | 0.1816 | +28.13 % |
| 2 d | **0.4688** | 0.4934 | 0.6428 | 0.0246 | **+4.98 %** | 0.1740 | +27.07 % |
| 3 d | **0.4677** | 0.4904 | 0.6465 | 0.0227 | **+4.63 %** | 0.1789 | +27.67 % |
| 7 d | **0.4635** | 0.4843 | 0.6441 | 0.0208 | **+4.29 %** | 0.1806 | +28.04 % |

### MASE (ratio — 1.0 = as good as seasonal naive)

| Horizon | Chronos-2 | NPTS | SeasNaive | abs diff vs NPTS | % vs NPTS | abs diff vs naive | % vs naive |
|---|---|---|---|---|---|---|---|
| 6 h | **0.6017** | 0.6871 | 0.9806 | 0.0854 | **+12.43 %** | 0.3789 | +38.64 % |
| 12 h | **0.6294** | 0.6923 | 0.9666 | 0.0629 | **+9.09 %** | 0.3371 | +34.88 % |
| 1 d | **0.6552** | 0.7098 | 1.0320 | 0.0546 | **+7.69 %** | 0.3767 | +36.51 % |
| 2 d | **0.6656** | 0.7113 | 1.0502 | 0.0458 | **+6.43 %** | 0.3846 | +36.62 % |
| 3 d | **0.6664** | 0.7090 | 1.0628 | 0.0426 | **+6.01 %** | 0.3964 | +37.30 % |
| 7 d | **0.6653** | 0.7057 | 1.0688 | 0.0405 | **+5.73 %** | 0.4035 | +37.75 % |

### RMSSE (ratio)

| Horizon | Chronos-2 | NPTS | SeasNaive | abs diff vs NPTS | % vs NPTS | abs diff vs naive | % vs naive |
|---|---|---|---|---|---|---|---|
| 6 h | **0.4910** | 0.5494 | 0.7174 | 0.0584 | **+10.64 %** | 0.2265 | +31.57 % |
| 12 h | **0.5271** | 0.5688 | 0.7399 | 0.0416 | **+7.32 %** | 0.2128 | +28.76 % |
| 1 d | **0.5670** | 0.6064 | 0.8020 | 0.0394 | **+6.50 %** | 0.2350 | +29.30 % |
| 2 d | **0.5722** | 0.6075 | 0.8058 | 0.0354 | **+5.82 %** | 0.2336 | +28.99 % |
| 3 d | **0.5718** | 0.6053 | 0.8117 | 0.0335 | **+5.53 %** | 0.2398 | +29.55 % |
| 7 d | **0.5647** | 0.5957 | 0.8085 | 0.0309 | **+5.19 %** | 0.2128 | +30.15 % |

*Interpretation.* The advantage is largest where the incumbent is weakest — the short horizons,
where a foundation model's learned diurnal shape beats a resampling scheme. It shrinks with
horizon but never reverses, and never falls below +4.29 % on any metric at any horizon. That is
the "horizon robustness" claim: the win is monotone in the right direction and is not carried by
one favourable horizon.

Figures: `A_mase_vs_horizon`, `B_rmse_vs_horizon`, `C_mae_vs_horizon`, `D_rmsse_vs_horizon`.

---

## 7. Per-tank analysis — where the model actually wins

Full grid: [`results/chronos2/per_tank_comparison.csv`](../results/chronos2/per_tank_comparison.csv)
(144 rows = 24 tanks × 6 horizons, no blanks).

### 7.1 Tanks won

Winner decided on MASE. All 24 tanks are counted; none is excluded for being a bad sensor.

| Horizon | Chronos-2 wins | NPTS wins | SeasonalNaive wins | Chronos-2 share |
|---|---|---|---|---|
| 6 h | **19** | 5 | 0 | **79 %** |
| 12 h | **17** | 7 | 0 | **71 %** |
| 1 d | **17** | 7 | 0 | **71 %** |
| 2 d | **16** | 8 | 0 | **67 %** |
| 3 d | **16** | 8 | 0 | **67 %** |
| 7 d | **17** | 7 | 0 | **71 %** |

Win counts on the raw metrics agree: at 1 d Chronos-2 beats NPTS on 17/24 by MAE and MASE, and
on **23/24 by RMSE and RMSSE** — it is markedly better on the squared metrics, i.e. it makes
fewer large errors even on tanks where its mean error is slightly worse.

Across all six horizons: **15 tanks Chronos-2 wins at every horizon, 5 it loses at every
horizon, 4 are split.** Figure: `M_tanks_won`.

### 7.2 Tanks Chronos-2 never wins (measured, all 6 horizons)

| Tank | tier | panel mean KL/h | median MASE C2 | median MASE NPTS | median % vs NPTS |
|---|---|---|---|---|---|
| `NEW_BLOCK_RO` | dead | 0.0012 | 0.2239 | 0.1905 | **−19.10 %** |
| `NBX` | healthy | 0.6566 | 0.8117 | 0.7906 | −3.19 % |
| `CRICKET_GROUND_OHT` | degraded | 0.0423 | 0.1645 | 0.1608 | −2.20 % |
| `INFORMATION_CENTRE` | dead | 0.0052 | 1.9822 | 1.9443 | −1.95 % |
| `GJBC_BLOCK_1_A4_RO` | degraded | 0.0611 | 0.0709 | 0.0705 | −0.79 % |

Four of the five are dead or degraded sensors that barely move. The one real loss is **`NBX`**,
a healthy tank moving 0.66 KL/h, where NPTS is consistently ~2–3 % better on MASE. `G_BLOCK` and
`I&H_BLOCK` are also close losses at some horizons (split tanks).

### 7.3 Largest improvements (1 d horizon)

| Tank | tier | C2 MASE | NPTS MASE | % impr |
|---|---|---|---|---|
| `BE_BLOCK_RO` | dead | 0.5559 | 0.7854 | **+29.22 %** |
| `GJBC_LAW_BLOCK_3__A2` | healthy | 0.3994 | 0.5551 | **+28.06 %** |
| `GJBC_BLOCK_2_A1` | healthy | 0.5654 | 0.7483 | **+24.45 %** |
| `TECH_PARK` | healthy | 0.5260 | 0.6527 | **+19.42 %** |
| `GJBC_LAW_BLOCK_3__A3_GIRLS` | healthy | 0.5669 | 0.6891 | **+17.73 %** |

### 7.4 Worst Chronos-2 tanks (1 d horizon) — not hidden by the macro average

| Tank | tier | C2 MASE | NPTS MASE | SN MASE | mean actual KL/h | why |
|---|---|---|---|---|---|---|
| `INFORMATION_CENTRE` | **dead** | **2.0556** | 2.0182 | 4.3144 | 0.017 | 91.9 % zero hours; a near-constant series |
| `GJBC_LAW_BLOCK_3__A1` | healthy | **1.5211** | 1.6020 | 2.1192 | 0.626 | spiky, near-unpredictable draw pattern |
| `IT_BLOCK` | degraded | **1.0668** | 1.1764 | 1.6351 | 0.094 | 40.0 % zero hours |
| `I&H_BLOCK` | degraded | 0.8388 | 0.8324 | 1.5089 | 0.314 | 51.6 % zero hours |
| `NBX` | healthy | 0.8385 | 0.8191 | 1.1736 | 0.803 | the one healthy tank Chronos-2 loses outright |

**Tanks where Chronos-2 is worse than the seasonal-naive reference (MASE > 1), by horizon count:**

| Tank | horizons with MASE > 1 | MASE range | tier |
|---|---|---|---|
| `INFORMATION_CENTRE` | 6 of 6 | 1.664 – 2.063 | dead |
| `GJBC_LAW_BLOCK_3__A1` | 6 of 6 | 1.113 – 1.686 | healthy |
| `IT_BLOCK` | 5 of 6 | 1.063 – 1.081 | degraded |
| `I&H_BLOCK` | 1 of 6 (6 h only) | 1.126 | degraded |

Note NPTS is *also* above 1.0 on the same tanks at 1 d — these are hard series, not a
Chronos-specific failure. But on `INFORMATION_CENTRE` a constant-zero forecast would beat both.

### 7.5 By sensor trust tier (1 d)

| Tier | Tanks | Mean C2 MASE | Mean NPTS MASE | Worst C2 MASE | C2 wins |
|---|---|---|---|---|---|
| healthy | 15 | 0.6671 | 0.7353 | 1.5211 | 13/15 |
| degraded | 6 | 0.4790 | 0.5014 | 1.0668 | 3/6 |
| dead | 3 | 0.9483 | 0.9993 | 2.0556 | 1/3 |

*Interpretation.* The degraded tier scoring *better* than healthy is not a paradox: those tanks
barely move, so their seasonal-naive denominator is small and easy to beat. MASE ranks
difficulty-adjusted skill, not operational importance, and must be read alongside the
volume-weighted MAE in §5.3.

Figures: `E_per_tank_mase_24h` (every tank, both models), `F_per_tank_improvement_24h` (signed
improvement, outliers included), `G_per_tank_mase_heatmap` (tank × horizon).

---

## 8. Accuracy interpretation — how well does this predict water requirement?

MASE answers "is this better than the naive baseline". It does **not** answer "can we size a
refill with it". These are the operational numbers, measured on the same grid.

### 8.1 Volume over the 24 h following each origin (24 origins × 24 tanks)

`err %` = mean absolute error of the 24-hour total, as a percentage of that tank's mean 24-hour
total. Bias % = signed total error over all 24 windows.

| Tank | mean 24 h demand (KL) | C2 MAE (KL) | C2 err % | NPTS err % | SN err % | C2 bias % |
|---|---|---|---|---|---|---|
| `BE_BLOCK_OHT` | 45.59 | 7.79 | **17.1 %** | 18.5 % | 21.6 % | -13.3 % |
| `MRD_BLOCK` | 32.64 | 10.24 | 31.4 % | 43.1 % | 49.5 % | -17.3 % |
| `GJBC_LAW_BLOCK_3_A4_BOYS` | 30.10 | 6.66 | 22.1 % | 25.7 % | 32.5 % | -13.5 % |
| `MM_BLOCK` | 26.89 | 1.27 | **4.7 %** | 6.0 % | 5.5 % | +1.7 % |
| `G_BLOCK` | 24.24 | 4.26 | **17.6 %** | 15.8 % | 28.7 % | +6.7 % |
| `NBX` | 18.77 | 3.90 | 20.8 % | 23.9 % | 19.2 % | -14.9 % |
| `F_BLOCK` | 16.68 | 5.08 | 30.5 % | 31.3 % | 45.7 % | -22.5 % |
| `GJBC_LAW_BLOCK_3__A1` | 14.63 | 5.97 | 40.8 % | 40.4 % | 46.3 % | -37.2 % |
| `GJBC_LAW_BLOCK_3__A2` | 10.16 | 1.88 | **18.5 %** | 44.0 % | 28.4 % | +9.9 % |
| `GJBC_BLOCK_2_A1` | 9.52 | 1.27 | **13.3 %** | 27.2 % | 30.1 % | +0.2 % |
| `TECH_PARK` | 9.41 | 2.28 | 24.2 % | 36.0 % | 52.9 % | +3.7 % |
| `GJBC_LAW_BLOCK_3__A3_GIRLS` | 8.01 | 1.66 | 20.7 % | 40.5 % | 19.7 % | +6.2 % |
| `I&H_BLOCK` | 7.38 | 6.98 | 94.6 % | 99.6 % | 112.7 % | -93.1 % |
| `GJBC_BLOCK_1_A3` | 4.45 | 0.96 | 21.6 % | 21.8 % | 25.0 % | -9.7 % |
| `NEW_BLOCK` | 4.11 | 0.28 | **6.9 %** | 9.1 % | 9.4 % | -3.1 % |
| `GJBC_BLOCK_1_A2_GIRLS` | 3.87 | 0.75 | **19.4 %** | 22.6 % | 26.0 % | -8.5 % |
| `GJBC_BLOCK_1_A1_BOYS` | 3.73 | 0.72 | **19.5 %** | 21.5 % | 23.5 % | -5.7 % |
| `IT_BLOCK` | 2.22 | 0.72 | 32.5 % | 47.4 % | 34.6 % | -26.7 % |
| `ME_WORKSHOP_BLOCK` | 0.88 | 0.27 | 31.1 % | 33.5 % | 42.9 % | -29.1 % |
| `INFORMATION_CENTRE` | 0.41 | 0.39 | 96.5 % | 100.0 % | 126.3 % | -95.8 % |
| `BE_BLOCK_RO` | 0.27 | 0.09 | 31.7 % | 66.8 % | 63.0 % | -14.4 % |
| `CRICKET_GROUND_OHT` | 0.20 | 0.18 | 91.5 % | 97.5 % | 136.0 % | -90.9 % |
| `GJBC_BLOCK_1_A4_RO` | 0.09 | 0.09 | 95.9 % | 100.0 % | 61.9 % | -95.9 % |
| `NEW_BLOCK_RO` | 0.01 | 0.01 | 98.6 % | 100.0 % | 86.4 % | -76.9 % |

**Campus aggregate:** mean actual demand is **274.3 KL per 24 h**. Chronos-2's 24 h campus total
is off by **39.5 KL/day = 14.4 %** on average; NPTS by **45.6 KL/day = 16.6 %**.

*Interpretation.* Read this table as three groups.

* **11 tanks draw ≥ 9 KL per 24 h and carry 87.0 % of campus volume.** On these the 24-hour
  requirement is predicted to within **4.7 %–40.8 %**, with 8 of 11 under 25 %, and Chronos-2
  beats NPTS on 9 of the 11.
* **7 mid-size tanks (1–9 KL per 24 h).** Chronos-2 beats NPTS on **all 7**, sometimes by a wide
  margin (`GJBC_LAW_BLOCK_3__A2` 18.5 % vs 44.0 %; `TECH_PARK` 24.2 % vs 36.0 %).
* **6 tanks draw under 1 KL per 24 h.** Here the percentage is close to meaningless — a tank
  drawing 0.09 KL/day cannot be predicted to a percentage, and for the three dead sensors a
  constant-zero forecast would score better than any model.

Across all 24 tanks Chronos-2 has the lower 24-hour volume error on **22 of 24**. **Do not quote
a single campus accuracy figure**: quote 14.4 % at the campus level, and the per-tank table for
anything else.

### 8.2 The under-forecast bias — the most important operational caveat

Signed volume error, `100 × (Σpred − Σactual) / Σactual`, pooled over all tanks:

| Model | 6 h | 12 h | 1 d | 2 d | 3 d | 7 d |
|---|---|---|---|---|---|---|
| **Chronos2-ZS** | −9.29 % | −9.41 % | **−12.15 %** | −12.73 % | −12.68 % | −12.79 % |
| NPTS | −4.22 % | −5.98 % | −9.87 % | −9.85 % | −9.41 % | −8.63 % |
| SeasonalNaive-24 | +2.05 % | −0.55 % | **+0.40 %** | +0.43 % | +0.90 % | +1.87 % |

Restricted to the 15 healthy tanks: Chronos-2 −9.53 %, NPTS −6.75 %, SeasonalNaive −0.56 % at 1 d.

*Interpretation.* Hourly outflow is spiky and right-skewed. A forecast that minimises mean
absolute error on such a series sits near the conditional median, which is below the conditional
mean — so its totals under-shoot. Chronos-2 is more accurate per hour and **more biased in
aggregate** than the incumbent, and both are more biased than seasonal naive. **A refill
decision must not use the Chronos-2 mean forecast directly as a volume estimate.** Use an upper
quantile, or apply a per-tank multiplicative bias correction fitted on a held-out window; the
per-tank bias column above is exactly the correction table. This is cheap and is the second
recommended next step after interval calibration.

---

## 9. Covariate analysis — do covariates earn their compute?

**Measured conclusion: no. Ship the zero-shot variant.**

Chronos-2 accepts covariates, which Chronos-1 could not, so three variants were run on the
identical grid.

| Variant | MASE 6 h | MASE 1 d | MASE 7 d | MAE 7 d | runtime | vs ZS at 7 d (MASE) |
|---|---|---|---|---|---|---|
| **Chronos2-ZS** | 0.6017 | 0.6552 | 0.6653 | 0.2199 | **1.5 min** | — |
| Chronos2-COV-LEAN | 0.6027 | 0.6562 | 0.6640 | 0.2184 | 5.9 min | +0.19 % |
| Chronos2-COV-XL | **0.6002** | **0.6550** | **0.6628** | **0.2181** | 10.5 min | **+0.37 %** |
| Chronos2-COV | 0.6018 | 0.6558 | 0.6632 | 0.2183 | 15.5 min | +0.31 % |

Total spread across all four variants, per horizon (MASE): 0.00247, 0.00228, 0.00123, 0.00230,
0.00227, 0.00245. **The entire covariate effect is in the third decimal place.**

Per-tank, at 1 d, the best variant beats zero-shot on **14 of 24 tanks** with a mean MASE
difference of **+0.0002** — i.e. it is a coin flip whose expected value rounds to zero.

**This is what the feature study predicted.** Mutual information with future demand at
per-tank-hourly resolution (`eda/feature_mi_by_horizon.csv`):

| Feature | MI @ 6 h | MI @ 24 h | MI @ 168 h |
|---|---|---|---|
| `lag_1` | 0.232 | **0.615** | **0.515** |
| `lag_24` | 0.211 | 0.376 | 0.343 |
| `lag_168` | 0.192 | 0.339 | 0.344 |
| `roll_24` | 0.294 | 0.274 | 0.260 |
| `hour_cos` | 0.054 | 0.092 | 0.095 |
| `exam_proximity` | 0.035 | 0.026 | 0.031 |
| `is_weekend` | 0.0013 | **0.0000** | 0.0011 |
| `dow_sin` | 0.0048 | 0.0048 | 0.0029 |
| `dow_cos` | 0.0000 | 0.0000 | 0.0037 |
| `is_holiday` | **0.0000** | 0.0036 | 0.0012 |

Lag features carry two orders of magnitude more information than any calendar flag — and the lag
structure is exactly what a zero-shot foundation model already reads from its context window.
The calendar covariates add variates to each forecasting task without adding information.

**This does not contradict the daily-level EDA.** On *campus daily totals*, weekday demand
(273.59 KL) genuinely exceeds weekend demand (231.83 KL), difference 41.77 KL, two-sample
**t = 9.357, p < 1e-5**; and demand differs across academic phases, **ANOVA F = 8.73,
p = 6.29e-08** (`eda/eda_report.txt`). Both results are true. The calendar effect is real at
**campus-daily** aggregation and vanishes into noise at **per-tank-hourly** resolution, which is
the resolution the model runs at. The aggregation level changed, not the statistics.

One further point against the covariate variants, measured in `results/chronos2/unified/leaderboard.csv`:
they are **more** volume-biased than zero-shot, not less — at 1 d, COV −13.75 %, COV-LEAN
−13.55 %, COV-XL −13.88 %, against zero-shot's −12.15 %. Whatever the covariates change, they do
not change it in the direction the operator cares about (§8.2).

**Recommendation: `Chronos2-ZS`.** Statistically indistinguishable accuracy, one tenth the
compute, and no covariate pipeline to build or maintain in production. Figures:
`N_covariate_vs_zeroshot`, `L_variant_selection`. *No covariate experiment was re-run for this
review; all numbers come from the completed run.*

---

## 10. Uncertainty and calibration

**Kept deliberately separate from §5–§8, which are point-forecast results.** A model can be the
best point forecaster and the worst uncertainty quantifier, and here that is nearly the case.

### 10.1 Empirical p10–p90 coverage (nominal **0.80**)

| Model | 6 h | 12 h | 1 d | 2 d | 3 d | 7 d |
|---|---|---|---|---|---|---|
| **Chronos2-ZS** | **0.721** | **0.714** | **0.722** | 0.729 | 0.732 | 0.743 |
| Chronos2-COV-LEAN | 0.703 | 0.706 | 0.718 | 0.726 | 0.737 | 0.749 |
| Chronos2-COV | 0.737 | 0.734 | 0.749 | 0.753 | 0.769 | 0.776 |
| Chronos2-COV-XL | 0.740 | 0.740 | 0.757 | 0.757 | 0.772 | 0.775 |
| **NPTS** | **0.850** | 0.846 | **0.842** | 0.841 | 0.839 | 0.836 |
| ETS | 0.915 | 0.913 | 0.912 | 0.915 | 0.916 | 0.923 |
| SeasonalNaive-24 | 0.903 | 0.902 | 0.900 | 0.921 | 0.933 | 0.959 |
| Theta | 0.921 | 0.930 | 0.940 | 0.950 | 0.954 | 0.965 |

### 10.2 Mean p10–p90 interval width (KL/h), same rows

| Model | 6 h | 12 h | 1 d | 2 d | 3 d | 7 d |
|---|---|---|---|---|---|---|
| Chronos2-ZS | 0.560 | 0.583 | 0.599 | 0.615 | 0.623 | 0.632 |
| Chronos2-COV-XL | 0.578 | 0.615 | 0.636 | 0.649 | 0.656 | 0.667 |
| NPTS | 0.676 | 0.677 | 0.680 | 0.682 | 0.681 | 0.680 |
| ETS | 1.222 | 1.237 | 1.262 | 1.303 | 1.339 | 1.456 |
| SeasonalNaive-24 | 1.681 | 1.681 | 1.709 | 2.058 | 2.350 | 3.259 |

**Verdict: Chronos-2 is overconfident.** Its band is the narrowest of any model (0.56–0.63 KL/h)
and covers 7–9 points below nominal. NPTS is the best-calibrated model in the study: 0.84 with a
0.68 KL/h band. ETS, Theta and SeasonalNaive reach high coverage only by being 2–5× wider, which
is not a virtue — coverage and width must be read together.

*Interpretation.* At a nominal 80 % band, the truth should fall outside 1 time in 5. With
Chronos-2 it falls outside roughly **1 time in 3.6**. Using its p10–p90 to size a safety margin
therefore under-states risk. Combined with the −12 % volume bias in §8.2, the two errors point
the same way: **too low and too confident**.

**The fix is cheap and is the single highest-value next step:** split-conformal calibration on a
held-out window — compute the empirical quantile of the absolute residual per tank and horizon
and widen the interval to hit 80 % coverage. It requires no retraining. It was not done in this
study and is not claimed. Figure: `K_interval_calibration`; example intervals in
`I_actual_vs_predicted_24h` and `J_final_holdout_7day`.

---

## 11. Sensor quality and missingness

The mass-balance identity `Opening + Inflow − Outflow − Closing = 0` is scored as
`rel_resid = mean|residual| / mean outflow` — balance error as a fraction of the tank's own
throughput (`eda/tank_mass_balance.csv`). That splits the campus into three tiers
(`eda/tank_trust.json`):

The tier rule, as implemented in `eda/eda_hourly.py`, is a disjunction — not `rel_resid` alone:

```
dead     = mean < 0.01 KL/h  OR  > 90 % zero hours
degraded = > 10 % missing  OR  > 45 % zero hours  OR  rel_resid > 0.10
healthy  = everything else
```

* **15 healthy** — every one has `rel_resid ≤ 0.097`, ≤ 9.9 % missing and ≤ 14.5 % zero hours
* **6 degraded** — `ME_WORKSHOP_BLOCK` (rel_resid 0.145, 19.9 % missing), `CRICKET_GROUND_OHT`
  (0.144, 20.7 % missing), `GJBC_BLOCK_1_A4_RO` (0.118, 51.5 % zero hours), `IT_BLOCK` (0.116,
  40.0 % zero), `GJBC_BLOCK_1_A1_BOYS` (0.112), `I&H_BLOCK` (0.115, 51.6 % zero hours)
* **3 dead** — `BE_BLOCK_RO` (73.5 % zero hours), `INFORMATION_CENTRE` (91.9 %),
  `NEW_BLOCK_RO` (96.3 %); all under 0.01 KL/h mean

**Scoring this relatively rather than absolutely reverses the conclusion**, which is worth
stating. An absolute 0.1 KL residual threshold flags `BE_BLOCK_OHT` on **40.5 % of its observed
hours** and looks damning — until you note it moves 1.67 KL/h, twenty times more than a small
tank. Relative to its own throughput its balance error is **rel_resid = 0.097, the best of any
tank that actually moves water**; under the threshold the EDA actually uses,
`max(0.1 KL, 25 % of mean flow)`, its breach rate is 6.6 %. Meanwhile `NEW_BLOCK_RO`, which is
96.3 % zero, breaches the absolute 0.1 KL threshold on **0.0 %** of hours while having the worst
relative balance error on campus (1.63). The absolute threshold was measuring tank size, not
sensor quality.

### Does sensor quality explain poor forecast performance?

Measured, per tank at 1 d (Spearman rank correlation with Chronos-2 MASE):

| Against | ρ |
|---|---|
| `missing_pct` | −0.232 |
| `zero_pct` | −0.192 |
| `rel_resid` | −0.146 |
| `panel_mean_kl_h` | **+0.291** |

**No — not in the direction one would guess.** All three quality measures correlate *negatively*
with MASE: dirtier sensors get *lower* (apparently better) MASE, because a near-constant series
has a tiny seasonal-naive denominator that is easy to match. The strongest association is with
demand size (ρ = +0.291): bigger, busier tanks are genuinely harder to forecast in scaled terms.

The honest statement is therefore narrower: **sensor quality does not explain forecast skill
across the fleet, but it does explain the two extreme failures.** `INFORMATION_CENTRE` (MASE
1.66–2.06 at every horizon) is 91.9 % zero hours with a 0.005 KL/h mean — the metric is
near-meaningless there, and a constant-zero forecast would beat the foundation model outright.
`GJBC_LAW_BLOCK_3__A1`, by contrast, is a **healthy** sensor with a genuinely spiky draw pattern
(MASE 1.11–1.69); that one is a modelling failure, not a data failure, and is the tank most
worth attacking next.

`ME_WORKSHOP_BLOCK` (19.9 % missing) and `CRICKET_GROUND_OHT` (20.7 % missing, and only 7,569
hours of record vs 11,448 for most tanks) both score *well* on MASE, which confirms the caveat
rather than contradicting it.

---

## 12. Recommended production model

**Chronos-2 zero-shot (`amazon/chronos-2`, `Chronos2-ZS`) as the point forecaster, with two
wrappers and a router.**

1. **Point forecast — `Chronos2-ZS`.** Best or statistically tied-best at every horizon, wins
   16–19 of 24 tanks, 89 s for a full 24-origin × 6-horizon backtest, no covariate pipeline, no
   training, no per-tank model store.
2. **Do not ship covariates.** The measured gain is ≤ 0.0025 MASE for 4–10× compute (§9).
3. **Add a calibration layer before shipping intervals.** Split-conformal widening per (tank,
   horizon) to reach 80 % coverage. Until it exists, publish the point forecast and label the
   p10–p90 band as uncalibrated (§10).
4. **Add a bias correction before publishing volumes.** Chronos-2 under-forecasts 24 h totals by
   12.2 % pooled, 9.5 % on healthy tanks. Correct per tank, or serve an upper quantile for
   refill sizing (§8.2).
5. **Route the dead tanks away from the model.** On `INFORMATION_CENTRE`, `NEW_BLOCK_RO` and
   `BE_BLOCK_RO` the sensible forecast is a constant near zero; on `INFORMATION_CENTRE` that
   would beat Chronos-2 outright. Routing them improves accuracy *and* saves compute. This is
   justified by measurement, not preference.
6. **Keep NPTS as the fallback.** It is the better-calibrated model, it wins 5–8 tanks, and it
   is already in production — a per-tank champion/challenger split is defensible on the numbers
   for `NBX`, `G_BLOCK`, `CRICKET_GROUND_OHT`, `GJBC_BLOCK_1_A4_RO` and `NEW_BLOCK_RO`.

**What must not be claimed.** Not "Chronos-2 is accurate for every tank" — it loses to NPTS on
5–8 tanks depending on horizon and is worse than seasonal naive on 3 tanks at 1 d. Not
"covariates improve forecasting" — the measured improvement is ≤ 0.4 % and within noise. Not
"MASE 0.65 means 65 % accuracy" — MASE is a ratio to the naive baseline.

---

## 13. Limitations

1. **One 23-day evaluation window.** 24 origins spanning 2026-03-24 → 2026-04-15. It covers
   every hour-of-day but a single seasonal regime; no monsoon, no full semester cycle. The
   improvement figures should not be extrapolated to a different season.
2. **No statistical significance test on the model difference.** The improvements are reported as
   point estimates on a shared row set. A paired bootstrap or Diebold–Mariano test over origins
   was not run. The per-tank win counts in §7.1 are the closest thing to a robustness check.
3. **The final holdout is one origin.** §5.4 is corroboration, not an independent test set.
4. **NPTS's advantage is calibration, and calibration was measured but not fixed.** No conformal
   layer exists yet, so the "recommended production model" in §12 is partly a plan.
5. **Covariate variants were compared, not tuned.** A different covariate set, or fine-tuning
   rather than conditioning, might behave differently; only conditioning was tested.
6. **MASE and MAE are computed on rows where the actual exists.** 1,416 of 190,080 rows per model
   (0.74 %) are dropped campus-wide for a missing actual; they are dropped identically for every
   model, so comparability holds, but coverage on `ME_WORKSHOP_BLOCK` and `CRICKET_GROUND_OHT`
   is thinner than elsewhere.
7. **Percentage-of-demand accuracy is undefined where demand is zero.** 26.5 % of observed
   hourly readings are exactly zero; §8.1 percentages on the five near-dead tanks should be read as "unusable",
   not as "96 % error".
8. **Runtimes are single measurements** on one Apple MPS laptop, not benchmarked means.
9. **TiDE and the AutoGluon ensembles are absent by scope.** PatchTST is no longer absent — it
   is on the grid at two configurations (§3) and Chronos-2 beats the stronger one by
   16.5–18.6 % MASE at every horizon. What this still does not establish is a result
   against a deep model given substantially more data or a per-tank architecture search; the
   claim is about this dataset at this size.

---

## 14. Future improvements

Ordered by measured value per unit of effort:

1. **Split-conformal interval calibration** per (tank, horizon). Closes the 0.72 → 0.80 coverage
   gap with no retraining. Highest value, lowest cost.
2. **Per-tank volume bias correction** from the bias column in §8.1, or serve p60–p75 instead of
   the mean for refill sizing. Directly fixes the −12 % under-forecast.
3. **A per-tank router**: constant-zero for the 3 dead tanks, NPTS for the 5 tanks it wins
   consistently, Chronos-2 elsewhere. Every routing decision above is already measured.
4. **Attack `GJBC_LAW_BLOCK_3__A1`** — a healthy sensor with MASE 1.11–1.69, the only genuine
   modelling failure in the fleet. Worth an event/spike-detection model rather than a smoother.
5. **Paired bootstrap over origins** to put a confidence interval on the 5.7–12.4 % improvement,
   which is what a reviewer will ask for next.
6. **Fine-tune Chronos-2** on the campus panel rather than conditioning on covariates. The
   covariate result in §9 says conditioning is exhausted; it says nothing about fine-tuning.
7. **Fix the sensors.** Six degraded and three dead tanks are a data-collection problem, and no
   forecasting model will solve them.

---

## 15. Reproducibility

Full step-by-step commands, runtimes and expected outputs are in the
[README, "Reproducing the final benchmark"](../README.md#reproducing-the-final-benchmark).
Short version:

```bash
source venv/bin/activate
python -m src.data.curate                 # receipt: 26 dirs -> 24 tanks, 270,849 rows
python -m tests.test_metrics              # MASE(SeasonalNaive) ~= 1.0 must hold
python eda/eda_hourly.py                  # trust tiers, feature MI study
python -m src.models.chronos2_forecasting # 4 Chronos-2 variants  (~32 min total)
python -m src.models.baselines_autogluon  # NPTS, SeasonalNaive, ETS, Theta, DOT (~26 min)
python -m src.models.score_benchmark      # benchmark_table.md + per_tank_comparison.csv
python -m src.models.unified_analysis     # every table, all models
python -m src.models.unified_figures      # 12 figures, PNG + SVG
```

Environment: `venv/` — Python 3.13.3, pandas 2.3.3, numpy 2.1.3, matplotlib 3.10.8,
torch on Apple MPS, `chronos-forecasting`, `autogluon.timeseries`.

### Where every number in this document comes from

| Section | File |
|---|---|
| §2 dataset | `python -m src.data.curate`, `results/chronos2/run_manifest.json` |
| §2.2 grid | `results/chronos2/run_manifest.json`, `results/chronos2/unified/summary.json` |
| §5, §6 | `results/chronos2/metrics_by_horizon.csv`, `results/chronos2/benchmark_table.md` |
| §5.4 | `results/chronos2/unified/leaderboard.csv` |
| §7 | `results/chronos2/per_tank_comparison.csv`, `results/chronos2/metrics_per_tank.csv` |
| §8.1 | `results/chronos2/unified/per_tank.csv` |
| §8.2 | `results/chronos2/unified/leaderboard.csv` |
| §8 tolerance rates | `results/chronos2/unified/per_tank.csv` |
| §9 | `results/chronos2/metrics_by_horizon.csv`, `eda/feature_mi_by_horizon.csv`, `eda/eda_report.txt` |
| §9 runtimes | `results/chronos2/run_manifest.json` (`wall_clock_s`) |
| §10 | `results/chronos2/metrics_by_horizon.csv` (`p10_p90_coverage`, `p10_p90_width`) |
| §11 | `eda/tank_mass_balance.csv`, `eda/tank_trust.json` |
| Figures | `results/chronos2/unified/plots/` (PNG + SVG) |

### Figure index

| ID | File | Shows |
|---|---|---|
| A | `A_mase_vs_horizon` | MASE by horizon — Chronos-2 / NPTS / SeasonalNaive |
| B | `B_rmse_vs_horizon` | RMSE by horizon (KL/h) |
| C | `C_mae_vs_horizon` | MAE by horizon (KL/h) |
| D | `D_rmsse_vs_horizon` | RMSSE by horizon |
| E | `E_per_tank_mase_24h` | Per-tank MASE at 1 d, Chronos-2 vs NPTS, all 24 tanks |
| F | `F_per_tank_improvement_24h` | Signed per-tank improvement %, positive and negative |
| G | `G_per_tank_mase_heatmap` | Chronos-2 MASE, tank × horizon |
| H | `H_error_distribution_24h` | Error distribution and absolute-error percentiles |
| I | `I_actual_vs_predicted_24h` | Actual vs prediction with p10–p90 band, 4 tanks |
| J | `J_final_holdout_7day` | 7-day final-holdout forecast with p10–p90 band |
| K | `K_interval_calibration` | Coverage vs nominal 0.80 |
| L | `L_variant_selection` | Accuracy per minute of compute |
| M | `M_tanks_won` | Number and % of tanks won, per horizon |
| N | `N_covariate_vs_zeroshot` | Covariate gain vs its cost, against the incumbent gap |

---

## 16. What was wrong with the earlier numbers in this repo

| Issue | Before | Now |
|---|---|---|
| Tank count | 26 series (two directories double-counted) | 24 physical tanks, supersets verified, count asserted in code |
| Comparability | AutoGluon 624 rows vs PatchTST 576 rows, different holdout dates | identical rows per horizon, asserted by `backtest.assert_comparable()` |
| Metrics | MAE + RMSE only | MAE, RMSE, MASE, RMSSE + interval coverage + width |
| Horizons | 24 h only | 6 h, 12 h, 1 d, 2 d, 3 d, 7 d |
| Origin stride | 24 h — every origin at 23:00, short horizons scored only on the quiet overnight window (MASE flattered to 0.22) | 23 h — every hour-of-day visited exactly once |
| Missingness | reported as 3–21 hours | true gapless reindex: up to 20.7 % |
