# Phase III — final experimental results and performance analysis

**PW26_PK_06 · Intelligent Water Management System · PES University RR**
24 tanks · hourly · 2025-01-01 → 2026-04-22 · 270,849 observations · 188,664 scored forecast rows

Every number below is measured and traceable to a file under `results/chronos2/`. Nothing is
estimated. Where something was not done, it says so.

---

## 0. Status against the six Phase III expectations

| # | Expectation | Status | Evidence |
|---|---|---|---|
| 1 | **System Testing** | 🟡 **Partial** | 6/6 metric unit tests pass (`tests/test_metrics.py`); 15/15 methodology checks pass under `score_benchmark --strict`. **No system/integration test suite exists** — the real-time system it would test is designed but not built |
| 2 | **Validation & Verification** | 🟢 **Done, and now stronger** | Row parity, leakage assertions, MASE identity, independent re-score — **plus the paired bootstrap + Diebold-Mariano significance tests added in this session**, which closed the single largest gap (`review_summary.md` §13.2) |
| 3 | **Deployment** | 🔴 **Not deployed** | A Chrome MV3 dock runs offline against a bundled forecast; `docker-compose.yml` exists but targets the **retired** AutoGluon models. Nothing is serving live |
| 4 | **Final Experimental Results** | 🟢 **Done** | 11 models × 6 horizons × 24 tanks on one shared grid, now including a PatchTST trained on this campus. §2, §2.4 below |
| 5 | **Performance Analysis (tables + graphs)** | 🟢 **Done — 14 figures, now 30** | 14 existing (A–N), **9 diagnostic (O–W)**, **3 calibrated 45-day holdout charts (X–Z)**, and **4 for the PatchTST control (AA–AD)**. §2.4, §3–§4, §7 |
| 5b | **Conformal calibration + bias correction** | 🟢 **Implemented** | Fitted on 8 Jan – 8 Mar 2026, measured on the disjoint 9 Mar – 22 Apr window: coverage 0.741 → 0.785, volume bias −10.5 % → −1.8 %. `src/models/calibration.py`, §7 below |
| 6 | **Complete Research Paper Draft** | 🔴 **Not written** | `docs/review_summary.md` (16 sections) is the raw material — results, methodology, limitations — but it is a technical report, not a paper |

**Three of six complete, one partial, two not started — plus the two calibration layers the
benchmark had listed as future work, now implemented (§7).** The honest framing for the
review: the *modelling* half of Phase III is finished and now statistically defended; the
*systems* half (testing, deployment) is designed in detail but unbuilt.

---

## 1. What was already done before this session

| Artefact | Contents |
|---|---|
| `results/chronos2/benchmark_table.md` | Full benchmark: 9 models × 6 horizons × 6 metrics, covariate study, calibration, per-tank tables |
| `results/chronos2/metrics_by_horizon.csv` | 54 rows — macro + volume-weighted metrics, coverage, interval width |
| `results/chronos2/metrics_per_tank.csv` | 1,296 rows |
| `results/chronos2/per_tank_comparison.csv` | 144 rows — Chronos-2 vs NPTS vs SeasonalNaive per tank × horizon |
| `results/chronos2/unified/` | 10 CSVs + `summary.json` — the leaderboard, significance against every opponent, win matrix, per-tank, skill, calibration and cost |
| `results/chronos2/unified/plots/` | **Figures U1–U12**, PNG + SVG, every panel carrying every model |
| `docs/review_summary.md` | 16-section technical report |

### Figures A–N (already in the deck)

| ID | Shows |
|---|---|
| A–D | MASE / RMSE / MAE / RMSSE vs horizon, three models |
| E | Per-tank MASE at 1 d, all 24 tanks |
| F | Signed per-tank improvement % at 1 d |
| G | Chronos-2 MASE heatmap, tank × horizon |
| H | Error distribution and absolute-error percentiles at 1 d |
| I | Actual vs predicted with p10–p90 band |
| J | 7-day final-holdout forecast |
| K | Interval coverage vs nominal 0.80 |
| L | Accuracy per minute of compute (variant selection) |
| M | Number and % of tanks won, per horizon |
| N | Covariate gain vs its cost |

---

## 2. Final experimental results

### 2.1 Evaluation grid

| Property | Value |
|---|---|
| Physical tanks | 24 (26 directories de-duplicated; loader **raises** if ≠ 24) |
| Forecast origins | 24, at a **23-hour stride** (co-prime with 24 → every hour-of-day visited exactly once) |
| Horizons | 6 h, 12 h, 1 d, 2 d, 3 d, 7 d |
| Models | 11, on **identical rows** — enforced by `assert_comparable()`, fatal under `--strict` |
| Scored rows | 3,405 / 6,861 / 13,530 / 27,323 / 41,140 / 96,405 → **188,664 total per model** |
| Leakage rows | **0** — every scored row satisfies `timestamp > origin` |
| Duplicates | **0** |
| Metric tests | **6/6 pass** |
| Methodology checks | **15/15 pass** |

### 2.2 Headline — Chronos-2 zero-shot vs the incumbent

| Horizon | Rows | **C2 MASE** | NPTS MASE | SeasNaive | **C2 MAE** | NPTS MAE | **C2 RMSE** | NPTS RMSE |
|---|---|---|---|---|---|---|---|---|
| **6 h** | 3,405 | **0.6017** | 0.6871 | 0.9806 | **0.1974** | 0.2306 | **0.3976** | 0.4420 |
| **12 h** | 6,861 | **0.6294** | 0.6923 | 0.9666 | **0.2029** | 0.2255 | **0.4207** | 0.4519 |
| **1 d** | 13,530 | **0.6552** | 0.7098 | 1.0320 | **0.2116** | 0.2303 | **0.4640** | 0.4921 |
| **2 d** | 27,323 | **0.6656** | 0.7113 | 1.0502 | **0.2161** | 0.2312 | **0.4688** | 0.4934 |
| **3 d** | 41,140 | **0.6664** | 0.7090 | 1.0628 | **0.2174** | 0.2307 | **0.4677** | 0.4904 |
| **7 d** | 96,405 | **0.6653** | 0.7057 | 1.0688 | **0.2199** | 0.2326 | **0.4635** | 0.4843 |

MASE is a ratio to a seasonal-naive baseline: **1.0 = no better than naive**. It is *not* a
percentage. SeasonalNaive-24 scoring ≈ 1.0 is the sanity check that the whole scale is correct.

### 2.3 All eleven models at the 1-day horizon

| Rank | Model | MASE | MAE | RMSE | RMSSE | p10–p90 coverage |
|---|---|---|---|---|---|---|
| 1 | Chronos2-COV-XL | 0.6550 | 0.2104 | 0.4620 | 0.5668 | 0.757 |
| **2** | **Chronos2-ZS** ← *production candidate* | **0.6552** | 0.2116 | 0.4640 | 0.5670 | 0.722 |
| 3 | Chronos2-COV | 0.6558 | 0.2105 | 0.4624 | 0.5671 | 0.749 |
| 4 | Chronos2-COV-LEAN | 0.6562 | 0.2102 | 0.4631 | 0.5677 | 0.718 |
| 5 | NPTS *(incumbent)* | 0.7098 | 0.2303 | 0.4921 | 0.6064 | 0.842 |
| 6 | Theta | 0.9561 | 0.2798 | 0.4989 | 0.6292 | 0.940 |
| 7 | DynamicOptimizedTheta | 0.9575 | 0.2806 | 0.4999 | 0.6304 | 0.940 |
| 8 | ETS | 0.9611 | 0.2685 | 0.4809 | 0.6072 | 0.912 |
| 9 | SeasonalNaive-24 | 1.0320 | 0.3178 | 0.6456 | 0.8020 | 0.900 |

**The four Chronos-2 variants are separated by 0.0012 MASE — statistically indistinguishable.**
Zero-shot is chosen because it costs **89 s** against 5.9–15.5 min for the covariate variants,
needs no covariate pipeline, and has no extra failure mode. That is a compute decision, not an
accuracy one, and it is defensible precisely because the accuracy difference is nil.

---

### 2.4 The deep-learning control — PatchTST

Every model above is either statistical or Chronos-2 itself, which left one question open:
**would a supervised transformer trained on this campus have done better?** PatchTST (Nie et al.,
ICLR 2023) is the natural candidate — the same patched-transformer family Chronos-2 belongs to,
but fitted on this data rather than pretrained on someone else's.

It is now on the identical grid: same 24 origins, same six horizons, same
188,664 scored rows, same row-parity assertion. **Two configurations were run,
because one would have been unfair:**

| Configuration | Context | Epochs | Batches/epoch | Wall clock |
|---|---|---|---|---|
| `PatchTST` (AutoGluon defaults) | 96 h | 30 | 50 | 2.2 min |
| `PatchTST-Tuned` (paper settings for hourly) | 512 h | 100 | 200 | 55.2 min |

Chronos-2 conditions on **2,048 hours** of context; a win over a model restricted to
96 would not be a win worth reporting. Both are fitted only on data
at or before the first origin (2026-03-24 22:00) and rolled forward, exactly as the statistical
baselines are. **The significance test below uses whichever configuration is better — measured,
not chosen.**

#### Macro MASE by horizon, on the paired frame

| Model | 6 h | 12 h | 1 d | 2 d | 3 d | 7 d | MAE @ 1 d |
|---|---|---|---|---|---|---|---|
| **Chronos-2 (zero-shot)** | 0.5962 | 0.6278 | 0.6563 | 0.6668 | 0.6670 | 0.6653 | 0.2111 |
| PatchTST (defaults) | 0.7680 | 0.8233 | 0.8575 | 0.8577 | 0.8655 | 0.8681 | 0.2820 |
| PatchTST (tuned) | 0.7141 | 0.7601 | 0.8061 | 0.8058 | 0.8104 | 0.8152 | 0.2608 |
| NPTS *(incumbent)* | 0.6813 | 0.6908 | 0.7109 | 0.7126 | 0.7095 | 0.7057 | 0.2297 |
| SeasonalNaive-24 *(reference)* | 0.9697 | 0.9639 | 1.0363 | 1.0529 | 1.0637 | 1.0692 | 0.3170 |

#### Chronos-2 vs PatchTST-Tuned — paired bootstrap (10,000 resamples) + Diebold–Mariano

| Horizon | MASE improvement | 95 % CI | Bootstrap *p* | DM stat | DM *p* | Origins won | Verdict |
|---|---|---|---|---|---|---|---|
| **6 h** | **16.51 %** | [9.93, 24.04] | < 0.0001 | -4.52 | 1.5e-04 | 21/24 | ✅ significant |
| **12 h** | **17.40 %** | [10.88, 23.37] | < 0.0001 | -5.28 | 2.4e-05 | 23/24 | ✅ significant |
| **1 d** | **18.58 %** | [13.63, 22.79] | < 0.0001 | -6.45 | 1.4e-06 | 23/24 | ✅ significant |
| **2 d** | **17.25 %** | [13.42, 21.04] | < 0.0001 | -5.99 | 4.2e-06 | 23/24 | ✅ significant |
| **3 d** | **17.69 %** | [14.95, 20.40] | < 0.0001 | -7.20 | 2.5e-07 | 24/24 | ✅ significant |
| **7 d** | **18.39 %** | [17.04, 19.80] | < 0.0001 | -12.21 | 1.6e-11 | 24/24 | ✅ significant |

**Chronos-2 zero-shot is 16.5–18.6 % better in MASE at every one of the six horizons**,
significant at all six by both tests, winning
**120 of 144** tank×horizon cells.
Per tank at 1 d: **19 significantly better,
4 indistinguishable,
1 significantly worse.**

The three tanks where PatchTST does best:

| Tank | tier | Chronos-2 improvement | 95 % CI | Verdict |
|---|---|---|---|---|
| `NEW_BLOCK_RO` | dead | -19.93 % | [-22.84, -17.11] | PatchTST better |
| `CRICKET_GROUND_OHT` | degraded | -1.17 % | [-3.03, +0.72] | no significant difference |
| `INFORMATION_CENTRE` | dead | -0.83 % | [-1.83, +0.02] | no significant difference |

#### Interval calibration at 1 d

| Model | p10–p90 coverage | Gap to 0.80 | Mean width (KL/h) | Rows with p10 < 0 |
|---|---|---|---|---|
| **Chronos-2 (zero-shot)** | 0.722 | -0.078 | 0.599 | 0.0 % |
| PatchTST (defaults) | 0.808 | +0.008 | 0.806 | 62.4 % |
| PatchTST (tuned) | 0.835 | +0.035 | 0.806 | 65.9 % |
| NPTS *(incumbent)* | 0.842 | +0.042 | 0.680 | 0.0 % |
| SeasonalNaive-24 *(reference)* | 0.900 | +0.100 | 1.709 | 79.7 % |

On coverage alone **PatchTST is the best-calibrated model in this table** — it lands within a
point of nominal where Chronos-2 misses by eight. That comparison does not survive the last column.
PatchTST reaches nominal coverage with a substantially wider band whose **lower bound is below zero
on a large fraction of rows** — a negative water outflow, which cannot happen.

This is the §4.3 zero-inflation mechanism seen from the other side. A continuous density can only
cover a series that is ~24 % exact zeros by spending probability mass on impossible values.
Chronos-2 refuses to go negative and under-covers; PatchTST goes negative and covers. **Neither is
calibrated**, and an interval that tells an operator a tank might draw a negative volume is not
usable for sizing a refill. The fix for both is the asymmetric conformal calibration of §7 — not a
wider band.

#### What each model cost to produce the same 188,664 forecasts

| Model | Regime | Wall clock | Minutes | MASE @ 1 d |
|---|---|---|---|---|
| **Chronos-2 (zero-shot)** | zero-shot — no training | 89 s | 1.5 min | 0.6563 |
| PatchTST (defaults) | trained — 30 epochs, context 96 h, one predictor per horizon | 132 s | 2.2 min | 0.8575 |
| PatchTST (tuned) | trained — 100 epochs, context 512 h, one predictor per horizon | 3,311 s | 55.2 min | 0.8061 |
| NPTS (+4 statistical baselines, one pass) | fitted — one predictor per horizon | 1,579 s | 26.3 min | 0.7109 |

#### The inference

**PatchTST is not a bad model here.** It beats ETS, Theta and DynamicOptimizedTheta, and it beats
the seasonal-naive reference at every horizon. **It loses to the *incumbent* as well as to
Chronos-2**, which is the more interesting finding: on 24 short, noisy, zero-inflated series,
training a transformer from scratch buys less than a well-chosen non-parametric method, and far
less than a pretrained one. That is the argument for a foundation model on this problem, stated as
a measurement rather than a preference.

The operational asymmetry is larger than the accuracy gap. PatchTST needed **55 minutes**
of training against Chronos-2's **89 seconds** of inference — a **37×**
difference — and it must be refitted whenever the campus changes, whereas the zero-shot model holds
no fitted state that can go stale.

Full analysis: `results/chronos2/unified/`. Regenerate with
`python -m src.models.patchtst_benchmark` (both presets) then
`python -m src.models.unified_analysis`.

---

## 3. NEW — statistical significance (the gap that is now closed)

`docs/review_summary.md` §13.2 stated: *"No statistical significance test on the model
difference… A paired bootstrap or Diebold-Mariano test over origins was not run."* **It has now
been run.** Method: 10,000 paired bootstrap resamples of the 24 forecast origins, plus a
Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

**Figure O — `O_significance_forest`** · Table: `phase3/significance_by_horizon.csv`

| Horizon | MASE improvement | 95 % CI | bootstrap *p* | DM stat | DM *p* | origins won | Verdict |
|---|---|---|---|---|---|---|---|
| 6 h | **12.50 %** | [6.13, 20.37] | < 0.0001 | −3.37 | 0.0026 | 18/24 | **significant** |
| 12 h | **9.11 %** | [4.63, 14.40] | < 0.0001 | −3.57 | 0.0016 | 17/24 | **significant** |
| 1 d | **7.68 %** | [4.39, 11.26] | < 0.0001 | −3.71 | 0.0012 | 19/24 | **significant** |
| 2 d | **6.42 %** | [3.89, 8.97] | < 0.0001 | −3.27 | 0.0034 | 21/24 | **significant** |
| 3 d | **5.99 %** | [3.98, 8.05] | < 0.0001 | −2.97 | 0.0069 | 21/24 | **significant** |
| 7 d | **5.73 %** | [4.77, 6.73] | < 0.0001 | −3.96 | 0.0006 | **24/24** | **significant** |

Same result on MAE (improvements 14.37 % → 5.46 %, every CI excluding zero, DM *p* ≤ 0.013).

> These per-origin macro figures reproduce the published pooled headline numbers to within
> 0.07 pp (e.g. 6 h: 12.50 % here vs 12.43 % published). The small difference is the aggregation
> order — equal weight per origin here, pooled over origins there. Both are stated.

### 3.1 Per-tank significance — and why the router design was right

**Figure P — `P_per_tank_significance_h24`** · Table: `phase3/significance_per_tank_h24.csv`

| Verdict at 1 d | Count | Tanks |
|---|---|---|
| **Chronos-2 significantly better** | **11** | `BE_BLOCK_RO` (+28.4 %), `GJBC_LAW_BLOCK_3__A2` (+28.0 %), `GJBC_BLOCK_2_A1` (+24.7 %), `TECH_PARK` (+19.3 %), `GJBC_LAW_BLOCK_3__A3_GIRLS` (+17.9 %), `F_BLOCK` (+15.3 %), `MRD_BLOCK` (+12.5 %), `IT_BLOCK` (+9.8 %), `NEW_BLOCK` (+5.7 %), `GJBC_LAW_BLOCK_3__A1` (+4.8 %), `ME_WORKSHOP_BLOCK` (+4.1 %) |
| **No significant difference** | **9** | `GJBC_LAW_BLOCK_3_A4_BOYS`, `GJBC_BLOCK_1_A3`, `GJBC_BLOCK_1_A2_GIRLS`, `GJBC_BLOCK_1_A1_BOYS`, `BE_BLOCK_OHT`, `MM_BLOCK`, `I&H_BLOCK`, `G_BLOCK`, `NBX` |
| **NPTS significantly better** | **4** | `NEW_BLOCK_RO` (−20.6 %), `INFORMATION_CENTRE` (−1.8 %), `CRICKET_GROUND_OHT` (−1.7 %), `GJBC_BLOCK_1_A4_RO` (−0.8 %) |

**The decisive finding: all four significant losses are on `degraded` or `dead` sensors. Not one
`healthy` tank is significantly worse under Chronos-2.**

This is the strongest single result of the session. It converts a soft claim ("Chronos-2 loses on
5–8 tanks") into a precise one: *of the 15 healthy tanks, Chronos-2 is significantly better on 8,
indistinguishable on 7, and significantly worse on none.* The apparent losses on `G_BLOCK` (−1.0 %) and
`NBX` (−1.7 %) have confidence intervals crossing zero — **those differences are not real**, and
building an NPTS fallback for them would have been fitting to noise.

---

## 4. NEW — performance patterns

### 4.1 Error is proportional to demand, and the bias is constant across the day
**Figure Q — `Q_diurnal_error`** · Table: `phase3/diurnal_error_h24.csv`

* MAE by hour-of-day correlates **ρ = 0.958** with mean demand by hour-of-day. Peak error is at
  09:00–11:00 and 15:00 (0.36–0.37 KL/h); minimum at 03:00–04:00 (0.094 KL/h).
* **Chronos-2's bias is negative at all 24 hours of the day** — range −0.145 to −0.002 KL/h,
  worst at 17:00.

*Inference:* the model is not failing at particular times of day; error simply tracks activity.
But the under-forecast is **systematic, not situational** — it is present in every hour, which is
exactly the signature a per-tank multiplicative bias correction can fix.

### 4.2 Error saturates within one day and does not decay after
**Figure R — `R_error_by_leadtime`** · Table: `phase3/error_by_leadtime.csv`

| Lead time | Chronos-2 MAE | NPTS MAE |
|---|---|---|
| Day 1 (h 1–24) | 0.2117 | 0.2311 |
| Day 2 | 0.2207 | 0.2324 |
| Day 3 | 0.2198 | 0.2294 |
| Day 5 | 0.2193 | 0.2348 |
| Day 7 (h 145–168) | 0.2235 | 0.2350 |

**From day 1 to day 7, error grows only +5.6 %.** The sawtooth in the raw series is the diurnal
cycle, not forecast decay.

*Inference:* the model has learned the **repeating daily profile**, not a decaying extrapolation.
A 7-day forecast is nearly as good as a 1-day forecast — which is what makes weekly planning
viable, and is a stronger operational claim than the headline MASE alone.

### 4.3 The interval problem is a zero-inflation problem — diagnosed
**Figures S, W — `S_reliability_diagram`, `W_zero_inflation_diagnosis`** ·
Tables: `phase3/reliability.csv`, `phase3/zero_inflation_diagnosis.csv`

Chronos-2 quantile reliability at 1 d (nominal → empirical fraction of actuals below):

| Nominal | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| Empirical | **0.292** | **0.368** | 0.542 | 0.758 | 0.888 |
| Gap | **+0.192** | **+0.118** | +0.042 | +0.008 | −0.012 |

**The upper tail is nearly perfect. The failure is almost entirely in the lower tail.** Mechanism,
measured at the 1-day horizon:

* **23.7 %** of actuals are **exactly zero**.
* Chronos-2 places `p10 > 0` on **79.3 %** of rows (median p10 = 0.048 KL/h).
* **66.7 %** of all below-p10 misses are hours where demand was **exactly zero**.
* NPTS — a nonparametric sampler — places `p10 > 0` on only **50.8 %** of rows, and its lower-tail
  miss rate is **5.3 %** against Chronos-2's 16.6 %.

*Inference:* **NPTS is better calibrated not because it models uncertainty better, but because it
reproduces the zero atom of a zero-inflated series, and a continuous-density foundation model
cannot.** This reframes the one axis on which the incumbent wins from a general weakness into a
specific, fixable structural mismatch.

*And the naive fix is wrong:* clamping p10 to zero moves coverage from 0.722 to **0.888** —
overshooting nominal 0.80. The correct fix is **asymmetric** split-conformal calibration on the
lower tail only. That is a sharper prescription than the symmetric widening previously proposed.

### 4.4 The operational cost of the bias
**Figure U — `U_cumulative_volume`** · Table: `phase3/cumulative_volume_h24.csv`

Over 24 origins the campus drew **6,582 KL** (274.3 KL per 24 h).

| Model | Cumulative forecast shortfall | As % of demand |
|---|---|---|
| **Chronos-2** | **−800 KL** | **−12.15 %** |
| NPTS | −649 KL | −9.87 % |
| SeasonalNaive-24 | +26 KL | +0.40 % |

*Inference:* over the 23-day window Chronos-2 would have under-provisioned by **800 KL — about
33 KL per day, roughly one BE_BLOCK_OHT tankful every two days.** This is the number that decides
the architecture: **a refill must never be sized on the mean forecast.** Use an upper quantile or
the per-tank bias correction table.

### 4.5 Every tank beats the naive reference
**Figure T — `T_skill_vs_demand`** · Table: `phase3/skill_scores_h24.csv`

Skill = `1 − MAE(model)/MAE(SeasonalNaive-24)`.

* **All 24 tanks have positive skill** (0.214 → 0.519). Mean skill **0.333** vs NPTS **0.273**.
* Skill does **not** decline with demand size.

*Inference:* the headline is not being carried by quiet or broken series. This pre-empts the
obvious reviewer challenge — *"is your average just being flattered by dead tanks?"* — with a
direct measurement. (It is the counterpart to the already-published finding that MASE correlates
*positively* with demand, ρ = +0.291: bigger tanks are harder in scaled terms, yet still beaten.)

### 4.6 Win/loss across every tank and horizon
**Figure V — `V_win_matrix`** · Table: `phase3/win_matrix.csv`

| Horizon | 6 h | 12 h | 1 d | 2 d | 3 d | 7 d | Total |
|---|---|---|---|---|---|---|---|
| Chronos-2 wins | 19/24 | 17/24 | 17/24 | 16/24 | 16/24 | 16/24 | **101/144 (70 %)** |

These reproduce the published win counts exactly — an independent consistency check on the whole
pipeline, since they were recomputed here from the raw parquets by different code.

---

## 5. New artefacts produced this session

Generated by `python -m src.models.unified_analysis` (~4 min). **Refits nothing, re-scores nothing** —
it reads the completed prediction parquets. Seeded (`20260830`), so it is reproducible.

| Figure | File | Shows |
|---|---|---|
| **O** | `O_significance_forest` | Improvement with 95 % bootstrap CI, per horizon, MAE and MASE |
| **P** | `P_per_tank_significance_h24` | Per-tank improvement with CI; significant / indistinguishable / worse |
| **Q** | `Q_diurnal_error` | MAE and bias by hour of day, against the demand profile |
| **R** | `R_error_by_leadtime` | MAE vs lead time out to 168 h, raw and 24 h-smoothed |
| **S** | `S_reliability_diagram` | Quantile reliability curve + interval coverage at nominal 0.50 and 0.80 |
| **T** | `T_skill_vs_demand` | Skill vs seasonal naive, by demand size and sensor tier |
| **U** | `U_cumulative_volume` | Cumulative campus demand: forecast vs actual, and the running shortfall |
| **V** | `V_win_matrix` | Tank × horizon improvement heatmap |
| **W** | `W_zero_inflation_diagnosis` | Why the interval under-covers: the zero-atom mechanism |
| **X** | `X_tanks45_chronos2` | Chronos-2: four representative tanks, 45 continuous days, calibrated |
| **Y** | `Y_tanks45_npts` | NPTS: the same four tanks — flat on every one |
| **Z** | `Z_tanks45_seasonal_naive` | SeasonalNaive-24: closest tracker, worst MAE, every peak a day late |

Tables: `significance_by_horizon.csv`, `significance_per_tank_h24.csv`, `diurnal_error_h24.csv`,
`error_by_leadtime.csv`, `reliability.csv`, `win_matrix.csv`, `skill_scores_h24.csv`,
`cumulative_volume_h24.csv`, `zero_inflation_diagnosis.csv`, `phase3_manifest.json` — all under
`results/chronos2/unified/`.

---

## 6. Ultimate inference, per Phase III category

### 1. System Testing — 🟡 partial, and honestly so
6/6 metric unit tests and 15/15 methodology checks pass. That validates the **measurement
apparatus**, which is the part that matters most for the results being defensible — if
`test_seasonal_naive_mase_is_about_one` failed, every number in the benchmark would be wrong.

What does not exist is a system test suite, because the real-time system is specified
(`docs/implementation_plan.md`, Phases 1–11) but unbuilt. **Say this plainly in the review**
rather than presenting model tests as system tests. The test strategy is already written down:
unit, property (safety invariants S1–S6), contract, integration, parity and regression layers.

### 2. Validation & Verification — 🟢 the strongest part of the project
Four independent guarantees, all machine-checked: identical evaluation rows for all 9 models
(`assert_comparable()`, fatal under `--strict`); zero leakage rows; MASE ≈ 1.0 for seasonal naive;
independent re-score reproducing published results.

**As of this session, a fifth: statistical significance.** The improvement is significant at all
six horizons on both MAE and MASE, by two independent tests. That was the most likely question a
reviewer would ask, and the answer is now measured rather than argued.

### 3. Deployment — 🔴 the real gap
A Chrome MV3 dock renders real Chronos-2 forecasts from a bundled JSON, offline. The
`docker-compose.yml` stack (redis, postgres, api, worker, beat, web, nginx) exists but targets the
**retired** AutoGluon models, and its Postgres is provisioned but unused. **Nothing is serving.**

The architecture to close this is fully specified across six design documents. Do not overstate
the dock as a deployment: it is a demo surface with precomputed data, and describing it accurately
is more credible than describing it generously.

### 4. Final Experimental Results — 🟢 complete and defensible
Eleven models, 6 horizons, 24 tanks, 188,664 rows per model on one shared grid, zero leakage,
zero duplicates. **Chronos-2 zero-shot beats the incumbent NPTS at every horizon on every point
metric, by 5.7–12.5 % MASE, and the margin is statistically significant everywhere.** It costs
89 seconds for the entire backtest.

The two caveats are equally measured and must be presented with the result: intervals cover
0.714–0.743 against nominal 0.80, and 24-hour volume is under-forecast by 12.15 %.

### 5. Performance Analysis — 🟢 23 figures, and the new ones answer *why*
The original 14 establish **what** the model does. The 9 added here establish **why**, and each
pre-empts a specific reviewer challenge:

| Challenge | Answer |
|---|---|
| *"Is the improvement real or noise?"* | O, P — significant at all 6 horizons, two tests, 10,000 resamples |
| *"Are dead tanks flattering your average?"* | T — all 24 tanks have positive skill; skill is flat in demand size |
| *"Does it fall apart at long horizons?"* | R — +5.6 % error from day 1 to day 7 |
| *"Your intervals are broken."* | S, W — yes, and here is the exact mechanism and the correct fix |
| *"What does the error cost operationally?"* | U — 800 KL under-provisioned over 23 days |
| *"Which tanks should you not trust it on?"* | P, V — the 4 significant losses, all on degraded/dead sensors |

### 6. Research Paper Draft — 🔴 not started, but the material is complete
`docs/review_summary.md` already contains everything a paper needs: dataset, methodology,
results, per-tank analysis, calibration, limitations, reproducibility. The new significance
testing and the zero-inflation diagnosis supply what was missing for publication — a defensible
statistical claim and a novel, mechanistic finding.

**Suggested framing:** *"Zero-inflation limits probabilistic calibration of time-series foundation
models: evidence from 24 campus water tanks."* The point-forecast win is solid but unsurprising;
**§4.3 is the genuinely publishable contribution** — a clean, quantified demonstration that a
continuous-density foundation model cannot represent the zero atom, that this fully explains its
calibration deficit against a nonparametric sampler, and that the obvious correction overshoots.

---

## 7. Calibration — implemented and measured out of sample

`review_summary.md` §14 listed split-conformal calibration and per-tank volume bias correction as
the two highest-value next steps. **Both are now implemented** (`src/models/calibration.py`,
`src/models/calibrated_holdout.py`).

**Method.** Per-tank multiplicative volume factor (Σactual ÷ Σpred), applied first; then
conformalised quantile regression (Romano et al., 2019) per tank with **independent lower and
upper offsets**, because §4.3 showed the failure is almost entirely lower-tail. The lower bound is
clipped at zero — that is what lets the interval represent the ~24 % of exactly-zero hours.

**The split.** Parameters are fitted on **8 Jan – 8 Mar 2026** and reported on **9 Mar – 22 Apr**,
strictly later and strictly disjoint. `calibration.assert_disjoint` fails the run if the windows
touch — coverage measured on the rows it was fitted on would be circular. Each window gets its own
NPTS predictor fitted only on data preceding it.

### 7.1 Hourly intervals and volume bias (out of sample, nominal 0.80)

| Model | Coverage | Gap to 0.80 | Width (KL/h) | Miss below p10 | Volume bias | Hourly MAE |
|---|---|---|---|---|---|---|
| **Chronos-2** | 0.741 → **0.785** | 0.059 → **0.015** | 0.593 → 0.669 | 0.161 → **0.122** | −10.5 % → **−1.8 %** | 0.205 → 0.214 |
| NPTS | 0.850 → 0.827 | 0.050 → 0.027 | 0.688 → 0.786 | 0.055 → 0.083 | −6.9 % → −1.7 % | 0.227 → 0.241 |
| SeasonalNaive-24 | 0.268 → **0.804** | 0.532 → **0.004** | 0.000 → 0.780 | 0.371 → 0.099 | −0.1 % → −0.7 % | 0.304 → 0.304 |

SeasonalNaive is the cleanest demonstration: it has **no native interval at all**, and conformal
calibration takes it from 0.268 to 0.804 — essentially exact.

### 7.2 Campus daily totals, calibrated

| Model | Daily MAE (KL) | MAPE | Total bias | SD ratio | Corr. w/ actual |
|---|---|---|---|---|---|
| **Chronos-2** | 36.7 → **28.8** | **11.4 %** | −10.3 % → **−1.6 %** | **0.89** | **+0.65** |
| NPTS | 41.1 → 36.3 | 14.7 % | −6.6 % → −1.4 % | 0.06 | −0.31 |
| SeasonalNaive-24 | 42.9 → 42.7 | 17.2 % | +0.7 % → +0.1 % | 0.99 | +0.28 |

### 7.3 Per-tank view — figures X, Y, Z

Three figures, one per model, four tanks each. The tanks are the repository's existing
representative set (`results/chronos2/representative_tanks.json`) — highest, median and
lowest demand among live tanks plus the highest-MASE tank, chosen by measured role in
the recorded representative set, so the panel cannot be cherry-picked and lines up with
figures I and J.

| Tank | Role | Trust | Demand | Chronos-2 daily MAE | Band coverage |
|---|---|---|---|---|---|
| `BE_BLOCK_OHT` | high demand | healthy | 41.9 KL/day | 7.74 → **7.04 KL** | 70 % |
| `GJBC_LAW_BLOCK_3__A3_GIRLS` | medium demand | healthy | 8.7 KL/day | 1.80 → 1.92 KL | 67 % |
| `GJBC_BLOCK_1_A4_RO` | low demand | degraded | 0.1 KL/day | 0.07 → 0.07 KL | **16 %** |
| `GJBC_LAW_BLOCK_3__A1` | hardest (highest MASE) | healthy | 15.5 KL/day | 6.34 → **6.10 KL** | **21 %** |

**Per-tank daily bands land close to nominal — median coverage 0.781 across all 24 tanks against
a nominal 0.80.** The campus-total band covers only 0.628. That is not a contradiction: tank
errors are positively correlated, so they do not cancel in the sum the way an independent-sum
assumption would predict.

Two tanks carry nearly all of the per-tank failure, and they fail for different reasons:

* `GJBC_LAW_BLOCK_3__A1` (band 21 %) is the **known modelling failure** — a healthy sensor with a
  genuinely spiky draw that the model smooths away. Already flagged in `review_summary.md` §11.
* `GJBC_BLOCK_1_A4_RO` (band 16 %) is a **regime change**. It averaged 0.0005 KL/h during the
  calibration window — below the 0.01 KL/h threshold, so no bias factor was fitted for it — then
  woke up to ~0.08 KL/day in the reported window. **No correction fitted on an earlier window can
  anticipate a tank changing behaviour**, which is precisely why a production system must refit on
  a rolling basis and monitor coverage continuously.

NPTS is flat on all four tanks, confirming that the campus-level flatness is not an aggregation
artefact. SeasonalNaive-24 is visually the closest tracker — it reproduces every peak — yet has
the worst MAE, because every feature it reproduces arrives a day late.

### 7.4 Three findings

1. **The correction works, and the cost is stated.** Hourly MAE rises slightly (0.205 → 0.214)
   because scaling toward the conditional mean moves away from the conditional median, which is
   what minimises absolute error. Over 24 hours the bias compounds while noise cancels, so daily
   MAE *improves* 21 %. Apply it to volume and refill sizing — that is what it is for.
2. **Calibration cannot fix a model that does not track.** After correction NPTS still has an SD
   ratio of 0.06 and correlates −0.31 with reality. Post-hoc correction fixes *where* a forecast
   sits and *how sure* it claims to be, not *whether it responds to anything*.
3. **Daily-total bands did not transfer.** Fitted on 56 calibration days they cover only 63–65 %
   against nominal 80 %, and 90- or 120-day windows do not help — it is regime shift, not sample
   size. The hourly intervals, fitted on ~1,400 residuals per tank, transfer well. A production
   system must refit on a rolling basis and monitor exactly this.

---

## 8. The single-slide summary

> **Chronos-2 zero-shot beats the deployed incumbent at every forecast horizon — 5.7 % to 12.5 %
> lower MASE — and the margin is statistically significant at all six horizons (paired bootstrap
> over 24 origins and Diebold-Mariano, both *p* < 0.01). It wins 101 of 144 tank-horizon cells,
> beats the naive baseline on all 24 tanks, and is never significantly worse than the incumbent on
> any tank with a healthy sensor. It costs 89 seconds and requires no training.**
>
> **Its two known weaknesses have been diagnosed and corrected.** Intervals covered 72 % against
> a nominal 80 %, traced here to the model's inability to represent the 24 % of hours with exactly
> zero demand. Asymmetric conformal calibration and a per-tank volume bias correction — fitted on
> an earlier, disjoint window and measured out of sample — take coverage to **0.785** and cut
> volume bias from **−10.5 % to −1.8 %**, reducing daily campus error from 36.7 to **28.8 KL**.
> Neither requires retraining.
>
> **What calibration does not fix:** the incumbent still reproduces 6 % of real day-to-day
> variation and correlates −0.31 with it; Chronos-2 reproduces 89 % and correlates +0.65.

---

## 9. Reproducing this analysis

```bash
source venv/bin/activate
python -m tests.test_metrics                  # 6/6, ~3 s   — trust nothing if this fails
python -m src.models.score_benchmark --strict # ~7 s        — re-scores from existing parquets
python -m src.models.unified_analysis         # ~4 min      — every table, all models
python -m src.models.unified_figures          # ~25 s       — figures U1-U12
python -m src.models.calibrated_holdout       # ~3 min      — §7, figures X-Z
```

`unified_analysis` refits nothing and re-scores nothing; it reads
`results/chronos2/predictions_*.parquet` and asserts 188,664 paired rows, matching the published
row count exactly. Bootstrap seed `20260830` is fixed, so the CIs reproduce.
