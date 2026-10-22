# Final analysis — refresh study

Every number here is read from `results/refresh_2026-10-22/` and, for the earlier window, `results/chronos2/`, by `reports/build_final_analysis.py`. Regenerate it rather than editing it.

## 1. The data

* **Snapshot.** `dataset_2026-10-22`, fetched from Waltr's API: 520,936 hourly rows, 24 tanks, 2024-02-21 → 2026-10-21. The published study used 2025-01-01 → 2026-04-22.
* **Waltr revised its history.** Re-downloaded, the published period's flows differ: campus outflow 124,824 → 118,812 KL (-4.8 %), 56,534 of 270,849 tank-hours changed, level readings unchanged on all but 3. Most of the change zeroes small flows: the share of exactly-zero outflow hours rises on 19 of 24 tanks by more than 5 points (largest: TECH_PARK 11 → 71 %, F_BLOCK 6 → 58 %, GJBC_BLOCK_2_A1 10 → 50 %). The whole refresh study uses the revised figures; the published study keeps the original ones, so the two are not comparable in level (`audit/revisions_vs_published.csv`).
* **What the revision means.** Waltr now reports zero inflow and zero outflow in many hours whose level moved, so the forecast target — Waltr's reported outflow — leaves out small consumption that the original figures counted. That is what the dashboard and the extension show, so it is what the models are scored on, but the refresh study forecasts a slightly different quantity than the published one.
* **Sensor trust tiers.** The tier test is a mass balance (level change = inflow − outflow); on the revised flows it fails by construction and would mark 18 of 24 tanks degraded. Tiers here are the published ones, assessed on the original self-consistent data (`eda/TRUST_NOTE.md`).
* **Tanks that stop early:** INFORMATION_CENTRE (last reading 2026-10-18 15:00), MRD_BLOCK (last reading 2026-10-19 22:00). Their missing hours are dropped from every model's score alike.
* **Calendar.** The academic calendar covers Jan 2025 – May 2026 only. The five models that read it — Chronos2-COV, Chronos2-COV-LEAN, Chronos2-COV-XL, PatchTST, PatchTST-Tuned — see no academic flags for 2024 or after May 2026, so their refresh results are marked † and kept out of the deployment decision.
* **Benchmark-only models (‡).** TimesFM 3.0's weights are licensed for internal benchmarking and research only, not for production systems or end users, so it is scored as a reference point and is never a deployment candidate (`timesfm3_manifest.json`).

## 2. The evaluation window

The published protocol unchanged: 24 origins at a 23-hour stride, 22 Sep 2026 22:00 → 14 Oct 2026 23:00, six horizons, 183,165 scored rows per model, identical for all 14 models (`score_benchmark --strict`). The published window's origins ran 24 Mar → 15 Apr 2026.

## 3. Leaderboard on the new window — macro MASE (rank)

| Model | 6 h | 12 h | 1 d | 2 d | 3 d | 7 d | p10–p90 coverage, 1 d |
|---|---|---|---|---|---|---|---|
| Timestral-209M-ZS | 0.5796 (1) | 0.6536 (1) | 0.6801 (1) | 0.7045 (1) | 0.7173 (1) | 0.7397 (1) | 0.869 |
| Chronos2-ZS | 0.5942 (2) | 0.6684 (2) | 0.6939 (2) | 0.7178 (4) | 0.7313 (2) | 0.7546 (4) | 0.751 |
| Chronos2-COV-LEAN † | 0.6075 (5) | 0.6770 (3) | 0.6963 (3) | 0.7161 (2) | 0.7314 (3) | 0.7512 (2) | 0.760 |
| Chronos2-COV † | 0.6091 (6) | 0.6795 (5) | 0.6981 (4) | 0.7167 (3) | 0.7329 (4) | 0.7522 (3) | 0.791 |
| Chronos2-COV-XL † | 0.6066 (3) | 0.6783 (4) | 0.6992 (5) | 0.7201 (5) | 0.7358 (5) | 0.7548 (5) | 0.752 |
| TimesFM3-ZS ‡ | 0.6068 (4) | 0.6837 (6) | 0.7069 (6) | 0.7276 (6) | 0.7394 (6) | 0.7549 (6) | 0.846 |
| TimesFM2.5-ZS | 0.6421 (7) | 0.7047 (7) | 0.7268 (7) | 0.7472 (7) | 0.7589 (7) | 0.7757 (7) | 0.822 |
| NPTS | 0.8135 (10) | 0.8462 (8) | 0.8306 (8) | 0.8394 (8) | 0.8470 (8) | 0.8560 (8) | 0.892 |
| DynamicOptimizedTheta | 0.8114 (9) | 0.9381 (10) | 0.9603 (9) | 0.9931 (9) | 1.0167 (10) | 1.0504 (11) | 0.928 |
| Theta | 0.8112 (8) | 0.9381 (9) | 0.9603 (10) | 0.9932 (10) | 1.0170 (11) | 1.0514 (12) | 0.928 |
| ETS | 0.8574 (11) | 0.9540 (11) | 0.9753 (11) | 1.0021 (11) | 1.0223 (12) | 1.0387 (10) | 0.877 |
| SeasonalNaive | 0.9245 (12) | 0.9925 (13) | 0.9954 (12) | 1.0369 (13) | 1.0826 (14) | 1.0996 (13) | 0.880 |
| PatchTST † | 0.9377 (13) | 0.9899 (12) | 0.9990 (13) | 1.0217 (12) | 1.0343 (13) | 1.0116 (9) | 0.797 |
| PatchTST-Tuned † | 1.1155 (14) | 1.1575 (14) | 1.0032 (14) | 1.1662 (14) | 1.0141 (9) | 1.1557 (14) | 0.824 |

PatchTST-Tuned has a MASE above 1 at 6 of 6 horizons, scores worse than the SeasonalNaive model at 5 of 6, and swings between 1.00 and 1.17 from one horizon to the next. Its horizons are fitted separately, so that points at unstable training on the revised, zero-heavy target, beyond the missing calendar. `old_vs_new/OLD_VS_NEW.md` follows this up: the March-fitted network, run unchanged on these origins, has the lower MASE at 6 of 6 horizons (6 with a 95 % bootstrap interval excluding zero).

Lowest MASE per horizon: 6 h Timestral-209M-ZS; 12 h Timestral-209M-ZS; 1 d Timestral-209M-ZS; 2 d Timestral-209M-ZS; 3 d Timestral-209M-ZS; 7 d Timestral-209M-ZS.
Among the deployment candidates (zero-shot, no calendar): 6 h Timestral-209M-ZS; 12 h Timestral-209M-ZS; 1 d Timestral-209M-ZS; 2 d Timestral-209M-ZS; 3 d Timestral-209M-ZS; 7 d Timestral-209M-ZS.

## 4. Is the difference real? (new window)

Paired bootstrap over the origins and Diebold–Mariano, read together; a result only one test supports is the weaker claim.

* **Chronos2-ZS vs Timestral-209M-ZS.** Chronos2-ZS's MASE is higher than Timestral-209M-ZS's at every horizon; in Timestral-209M-ZS's favour both tests agree at 1 d, 2 d and 7 d and one test only at 12 h and 3 d.
* **Chronos2-ZS vs TimesFM2.5-ZS.** Chronos2-ZS's MASE is lower than TimesFM2.5-ZS's at every horizon; in Chronos2-ZS's favour both tests agree at 6 h, 12 h, 1 d, 2 d, 3 d and 7 d.
* **Timestral-209M-ZS vs TimesFM2.5-ZS.** Timestral-209M-ZS's MASE is lower than TimesFM2.5-ZS's at every horizon; in Timestral-209M-ZS's favour both tests agree at 6 h, 12 h, 1 d, 2 d and 7 d and one test only at 3 d.
* **Chronos2-ZS vs NPTS.** Chronos2-ZS's MASE is lower than NPTS's at every horizon; in Chronos2-ZS's favour both tests agree at 6 h, 12 h, 1 d, 2 d, 3 d and 7 d.
* **Timestral-209M-ZS vs NPTS.** Timestral-209M-ZS's MASE is lower than NPTS's at every horizon; in Timestral-209M-ZS's favour both tests agree at 6 h, 12 h, 1 d, 2 d, 3 d and 7 d.
* **Chronos2-ZS vs Chronos2-COV-XL †.** Chronos2-ZS's MASE is lower than Chronos2-COV-XL's at every horizon; neither test separates them at any horizon.
* **Chronos2-ZS vs TimesFM3-ZS ‡.** Chronos2-ZS's MASE is lower than TimesFM3-ZS's at every horizon; in Chronos2-ZS's favour both tests agree at no horizon and one test only at 6 h and 12 h.
* **Timestral-209M-ZS vs TimesFM3-ZS ‡.** Timestral-209M-ZS's MASE is lower than TimesFM3-ZS's at every horizon; in Timestral-209M-ZS's favour both tests agree at 6 h, 12 h, 1 d, 2 d and 3 d and one test only at 7 d.
* **TimesFM3-ZS ‡ vs TimesFM2.5-ZS.** TimesFM3-ZS's MASE is lower than TimesFM2.5-ZS's at every horizon; in TimesFM3-ZS's favour both tests agree at 6 h and 7 d.

## 5. Does it replicate? Published window vs new window

Ranks, not MASE levels, are compared: the revised data and longer history change the scale. Spearman correlation of the mean-rank leaderboards: **0.851** (14 models, p = 0.00012). The published window's ranks come from its re-analysis with TimesFM 3.0 added (`window1_timesfm3/`), which reproduces every published row unchanged; ranks below TimesFM 3.0 move down one place.

| Model | mean rank, published | mean rank, new |
|---|---|---|
| Timestral-209M-ZS | 1.00 | 1.00 |
| Chronos2-ZS | 4.17 | 2.67 |
| Chronos2-COV-LEAN † | 4.50 | 3.00 |
| Chronos2-COV † | 3.33 | 4.17 |
| Chronos2-COV-XL † | 2.00 | 4.50 |
| TimesFM3-ZS ‡ | 7.33 | 5.67 |
| TimesFM2.5-ZS | 6.17 | 7.00 |
| NPTS | 7.50 | 8.33 |
| DynamicOptimizedTheta | 12.17 | 9.67 |
| Theta | 11.83 | 10.00 |
| ETS | 12.00 | 11.00 |
| PatchTST † | 10.00 | 12.00 |
| SeasonalNaive | 14.00 | 12.83 |
| PatchTST-Tuned † | 9.00 | 13.17 |

Head-to-head verdicts in each window:

* **Chronos2-ZS vs Timestral-209M-ZS** — published: Chronos2-ZS's MASE is higher than Timestral-209M-ZS's at every horizon; in Timestral-209M-ZS's favour both tests agree at no horizon and one test only at 3 d and 7 d. New: Chronos2-ZS's MASE is higher than Timestral-209M-ZS's at every horizon; in Timestral-209M-ZS's favour both tests agree at 1 d, 2 d and 7 d and one test only at 12 h and 3 d.
* **Chronos2-ZS vs TimesFM2.5-ZS** — published: Chronos2-ZS's MASE is lower than TimesFM2.5-ZS's at every horizon; in Chronos2-ZS's favour both tests agree at 7 d and one test only at 6 h, 12 h, 1 d, 2 d and 3 d. New: Chronos2-ZS's MASE is lower than TimesFM2.5-ZS's at every horizon; in Chronos2-ZS's favour both tests agree at 6 h, 12 h, 1 d, 2 d, 3 d and 7 d.
* **Timestral-209M-ZS vs TimesFM2.5-ZS** — published: Timestral-209M-ZS's MASE is lower than TimesFM2.5-ZS's at every horizon; in Timestral-209M-ZS's favour both tests agree at 3 d and 7 d and one test only at 6 h, 12 h, 1 d and 2 d. New: Timestral-209M-ZS's MASE is lower than TimesFM2.5-ZS's at every horizon; in Timestral-209M-ZS's favour both tests agree at 6 h, 12 h, 1 d, 2 d and 7 d and one test only at 3 d.
* **Chronos2-ZS vs NPTS** — published: Chronos2-ZS's MASE is lower than NPTS's at every horizon; in Chronos2-ZS's favour both tests agree at 6 h, 12 h, 1 d, 2 d, 3 d and 7 d. New: Chronos2-ZS's MASE is lower than NPTS's at every horizon; in Chronos2-ZS's favour both tests agree at 6 h, 12 h, 1 d, 2 d, 3 d and 7 d.
* **Chronos2-ZS vs TimesFM3-ZS ‡** — published: Chronos2-ZS's MASE is lower than TimesFM3-ZS's at every horizon; neither test separates them at any horizon. New: Chronos2-ZS's MASE is lower than TimesFM3-ZS's at every horizon; in Chronos2-ZS's favour both tests agree at no horizon and one test only at 6 h and 12 h.
* **Timestral-209M-ZS vs TimesFM3-ZS ‡** — published: Timestral-209M-ZS's MASE is lower than TimesFM3-ZS's at every horizon; in Timestral-209M-ZS's favour both tests agree at no horizon and one test only at 1 d, 2 d, 3 d and 7 d. New: Timestral-209M-ZS's MASE is lower than TimesFM3-ZS's at every horizon; in Timestral-209M-ZS's favour both tests agree at 6 h, 12 h, 1 d, 2 d and 3 d and one test only at 7 d.

## 6. The operational view — daily campus volume

45 consecutive daily forecasts, 07 Sep → 21 Oct 2026; corrections fitted on the disjoint 09 Jul → 06 Sep 2026, which probably spans the 2026 summer break (the 2026 calendar dates are not available), a regime the corrections then carry into term time.

| Model | daily MAE raw → calibrated (KL) | volume bias raw → calibrated | hourly p10–p90 coverage raw → calibrated |
|---|---|---|---|
| Chronos2-ZS | 33.8 → 24.4 | -10.6 % → +2.5 % | 0.774 → 0.831 |
| Timestral-209M-ZS | 37.7 → 24.9 | -12.7 % → +0.8 % | 0.871 → 0.883 |
| TimesFM2.5-ZS | 38.8 → 25.0 | -13.0 % → +1.5 % | 0.839 → 0.847 |
| TimesFM3-ZS ‡ | 36.6 → 24.4 | -11.9 % → +2.4 % | 0.848 → 0.857 |
| NPTS | 47.9 → 40.6 | -11.3 % → -2.1 % | 0.892 → 0.885 |
| SeasonalNaive-24 | 39.9 → 40.2 | +1.3 % → +2.1 % | 0.438 → 0.791 |

Lowest calibrated daily MAE: Chronos2-ZS.

## 7. Compute

| Model | wall clock (s) | device | same-CPU (s) |
|---|---|---|---|
| Chronos2-ZS | 64.1 | mps | 72.3 |
| TimesFM2.5-ZS | 166.7 | cpu | 166.7 |
| Timestral-209M-ZS | 144.2 | cpu | 144.2 |
| TimesFM3-ZS ‡ | 143.6 | cpu | 143.6 |
| Chronos2-COV-XL † | 837.9 | mps |  |
| NPTS | 621.3 |  |  |
| PatchTST-Tuned † | 3711.0 |  |  |

Timestral sizes on the same rows (off the roster, untested):

| Size | 6 h | 12 h | 1 d | 2 d | 3 d | 7 d | wall clock (s) |
|---|---|---|---|---|---|---|---|
| 5M | 0.6222 | 0.6994 | 0.7206 | 0.7461 | 0.7575 | 0.7705 | 23 |
| 28M | 0.5955 | 0.6687 | 0.6958 | 0.7194 | 0.7320 | 0.7510 | 38 |
| 123M | 0.5880 | 0.6606 | 0.6863 | 0.7099 | 0.7226 | 0.7440 | 89 |
| 209M | 0.5796 | 0.6536 | 0.6801 | 0.7045 | 0.7173 | 0.7397 | 144 |

## 8. Interpretation — which model to deploy

*This section interprets the measurements above; the decision is the team's.*

* **Hourly accuracy: Timestral 209M, in both windows.** It has the lowest macro MASE at every horizon in the published window and again here. Against Chronos-2 zero-shot the lead was separated by both tests at no horizon in the published window and at 1 d, 2 d and 7 d in this one, so it replicated and strengthened. It also has the lower volume-weighted MAE at every horizon, so the lead holds on the high-demand tanks, and the lower 1-day MAE on 19 of 24 tanks.
* **Campus volume: Chronos-2 zero-shot, narrowly.** Timestral under-forecasts total volume more (-15.3 % vs -12.0 % at 1 d), so before correction Chronos-2's daily campus MAE is lower (33.8 vs 37.7 KL). After the per-tank volume correction the two are 0.5 KL apart (24.4 vs 24.9); the daily totals have no significance test, so neither is established as better there.
* **Cost.** Both are cheap enough to refresh every hour: the full 144-call backtest took 144 s for Timestral and 72 s for Chronos-2 on the same CPU, i.e. about 1.0 s per 24-tank forecast. Timestral needs no GPU; Chronos-2 can also use Apple MPS.
* **TimesFM 3.0 — benchmark only (‡).** Ranks on the new window: 6 h #4, 12 h #6, 1 d #6, 2 d #6, 3 d #6, 7 d #6; on the published window: 6 h #7, 12 h #8, 1 d #8, 2 d #8, 3 d #7, 7 d #6. Against Timestral: Timestral-209M-ZS's MASE is lower than TimesFM3-ZS's at every horizon; in Timestral-209M-ZS's favour both tests agree at 6 h, 12 h, 1 d, 2 d and 3 d and one test only at 7 d. Against Chronos-2 zero-shot: Chronos2-ZS's MASE is lower than TimesFM3-ZS's at every horizon; in Chronos2-ZS's favour both tests agree at no horizon and one test only at 6 h and 12 h. Calibrated daily campus MAE 24.4 KL (raw 36.6). Same-CPU backtest 144 s. Whatever it scores, its weights' licence rules it out of the extension, and Google publishes no fine-tuning recipe for 3.0; TimesFM 2.5 (Apache-2.0) is the TimesFM that could be fine-tuned and shipped.
* **Caveats.** Timestral is a preview release (weights under a modified MIT licence: no rights above USD 20M monthly revenue). The new window is one more 23-day window, and it forecasts Waltr's revised flow figures.

**Recommendation.** For the extension's per-tank hourly forecast, Timestral 209M is now the better-supported model: its lead over Chronos-2 replicated on a second window and is established by both tests at three horizons. Ship it with the per-tank volume correction, which matters more for it than for Chronos-2, and keep Chronos-2 zero-shot as the fallback and challenger. If the extension leads with daily campus volume rather than per-tank hours, the two are tied after correction and either serves.
