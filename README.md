# Intelligent Water Management System — Predictive Analytics for a Sustainable Campus

**Project ID:** PW26_PK_06
**Institution:** PES University, Ring Road Campus
**Guides:** Ms. Preet Kanwal · Mr. Prasad B Honnavalli
**Team:** Abhay Patil · Amogh E M · Harshavardhan M · Viraj Ved Shankar

Hourly water-demand forecasting for the **24 distribution tanks** on the PES University RR campus,
from the campus's Waltr tank sensors. Fourteen forecasting models are compared on exactly the
same data, in two separate evaluation windows six months apart, and the best one feeds the
**Waltr forecast dock**, a browser extension that shows each tank's forecast.

---

## Final results

### Recommendation

| Use | Model | Why |
|---|---|---|
| **Per-tank hourly forecast** (shipped in the extension) | **Timestral 209M**, zero-shot | Lowest error of all 14 models at every horizon, in both windows. In the second window its lead over Chronos-2 is confirmed by both significance tests at 1, 2 and 7 days ahead |
| **Fallback, and daily campus totals** | **Chronos-2**, zero-shot | A close second. After calibration, its daily campus error is within 1 KL/day of Timestral's in both windows |

Both are *zero-shot*: pretrained foundation models used as they are, with no training on campus
data. Every forecast also goes through a per-tank volume correction and calibrated 80 % intervals.

The published study (first window) kept Chronos-2 as the production model because Timestral's
lead was not yet confirmed. The second window confirmed it, so the
[final analysis](results/refresh_2026-10-22/FINAL_ANALYSIS.md) recommends Timestral. The final
choice is the team's.

### The two evaluation windows

| | Window 1 — published study | Window 2 — refresh study |
|---|---|---|
| Data | Waltr export, 1 Jan 2025 – 22 Apr 2026 (270,849 tank-hours) | Re-downloaded, 21 Feb 2024 – 21 Oct 2026 (520,936 tank-hours) |
| Benchmark | 24 forecast start times, 24 Mar – 15 Apr 2026 | 24 forecast start times, 22 Sep – 14 Oct 2026 |
| 45-day daily view | 9 Mar – 22 Apr 2026 | 7 Sep – 21 Oct 2026 |
| Results | [`results/chronos2/`](results/chronos2/) | [`results/refresh_2026-10-22/`](results/refresh_2026-10-22/) |

### Leaderboard

Mean rank across the six horizons (1 = best of 14), and **MASE** one day ahead. MASE is the
error relative to a "same hour yesterday" forecast: **below 1.0 is better than that, and lower
is better**. MASE levels are not comparable *between* windows, because each window is scaled by
its own history; ranks are.

| Model | Kind | Rank, window 1 | Rank, window 2 | MASE 1 d, window 1 | MASE 1 d, window 2 |
|---|---|---|---|---|---|
| **Timestral 209M** | foundation, zero-shot | **1.0** | **1.0** | 0.653 | 0.680 |
| **Chronos-2** | foundation, zero-shot | 4.2 | 2.7 | 0.655 | 0.694 |
| Chronos-2 + calendar (XL) † | foundation + covariates | 2.0 | 4.5 | 0.655 | 0.699 |
| TimesFM 3.0 ‡ | foundation, zero-shot | 7.3 | 5.7 | 0.734 | 0.707 |
| TimesFM 2.5 | foundation, zero-shot | 6.2 | 7.0 | 0.690 | 0.727 |
| NPTS (current incumbent) | statistical | 7.5 | 8.3 | 0.710 | 0.831 |
| PatchTST, tuned † | deep learning, trained here | 9.0 | 13.2 | 0.802 | 1.003 |
| SeasonalNaive-24 (reference) | "same hour yesterday" | 14.0 | 12.8 | 1.032 | 0.995 |

- † Reads the academic calendar, which ends in May 2026, so its window-2 result is qualified.
- ‡ Benchmark only: TimesFM 3.0's licence forbids production use.
- Window-1 ranks come from the published study re-analysed with TimesFM 3.0 added; every
  published number is unchanged. All 14 models:
  [`window_comparison.md`](results/refresh_2026-10-22/window_comparison.md).

### Is the difference real?

Each comparison uses two tests over the 24 forecast start times: a paired bootstrap (10,000
resamples) and Diebold–Mariano. A result only one test supports is the weaker claim.

| Comparison | Window 1 | Window 2 |
|---|---|---|
| **Timestral vs Chronos-2** | Chronos-2's MASE 0.3–0.9 % higher; bootstrap only, at 3 d and 7 d | Chronos-2's MASE 1.9–2.5 % higher; **both tests at 1 d, 2 d and 7 d** |
| **Chronos-2 vs NPTS** (incumbent) | 5.7–12.5 % lower MASE; both tests, all six horizons | 11.9–27.0 % lower; both tests, all six horizons |
| **Chronos-2 vs PatchTST tuned** (trained here) | 16.5–18.6 % lower; both tests, all six horizons | 28.1–47.0 % lower; both tests, all six horizons |
| **Chronos-2 vs TimesFM 2.5** | 2.7–10.9 % lower; Diebold–Mariano confirms 1 of 6 horizons | 2.7–7.5 % lower; both tests, all six horizons |

### Daily campus volume — the operational view

One forecast every night for the next 24 hours, over 45 consecutive days. The corrections are
fitted on a separate, earlier 60 days. Daily campus error in KL/day, raw → after calibration:

| Model | Window 1 (actual ≈ 270 KL/day) | Window 2 (actual ≈ 228 KL/day) |
|---|---|---|
| Chronos-2 | 36.7 → **28.8** | 33.8 → **24.4** |
| Timestral 209M | 35.8 → **28.0** | 37.7 → **24.9** |
| TimesFM 2.5 | 41.7 → 32.6 | 38.8 → 25.0 |
| TimesFM 3.0 ‡ | — | 36.6 → 24.4 |
| NPTS (incumbent) | 41.1 → 36.3 | 47.9 → 40.6 |

Raw, the foundation models under-forecast total campus volume by 9.6–13.0 %; after the
correction every model's bias is within ±2.6 %. The day-by-day forecasts can be explored in
[`reports/holdout45_explorer.html`](reports/holdout45_explorer.html) (window 2; open it in a browser).

### What we learned

1. **Pretrained beats trained-here.** Timestral, Chronos-2 and TimesFM 2.5, used zero-shot, beat
   every model fitted on campus data at every horizon in both windows, including a PatchTST tuned to
   the paper's hourly settings. TimesFM 3.0 does so in window 2, but at only 3 of 6 horizons in
   window 1.
2. **Retraining did not help.** The March-trained PatchTST beats the one retrained on September
   data at all six horizons; the retrained tuned network collapsed to near-zero forecasts at
   four of them ([`OLD_VS_NEW.md`](results/refresh_2026-10-22/old_vs_new/OLD_VS_NEW.md)). The
   zero-shot models need no retraining. What fresher data improves is their calibration.
3. **Calibration is essential.** Raw intervals and volumes are biased. Chronos-2's 80 % interval
   covered 74 % of hours in window 1, 79 % after calibration (77 % → 83 % in window 2).
4. **It is cheap.** On the same laptop CPU, the full 144-forecast backtest took 72 s for Chronos-2,
   144 s for Timestral and TimesFM 3.0, and 167 s for TimesFM 2.5. Training the tuned PatchTST took
   62 minutes.
5. **The data needs care.** Waltr revised its own history in 2026: over the published study's
   period, campus outflow fell 4.8 % and 56,534 of 270,849 tank-hours changed. Of the 24 sensors, 15 are
   healthy, 6 degraded and 3 dead.

**Full reports:** [final analysis](results/refresh_2026-10-22/FINAL_ANALYSIS.md) (both windows) ·
[`docs/phase3_results.md`](docs/phase3_results.md) and
[`docs/review_summary.md`](docs/review_summary.md) (published study) ·
[`benchmark_table.md`](results/refresh_2026-10-22/benchmark_table.md) (every model, horizon and
metric).

---

## Models compared

| Model | Kind | Trained on campus data? | Deployable? |
|---|---|---|---|
| Chronos-2 (Amazon) — zero-shot, plus three variants that also read calendar features | foundation | No | Yes |
| Timestral 209M (Mistral) | foundation | No | Yes — modified MIT licence (no rights above USD 20M monthly revenue) |
| TimesFM 2.5 (Google) | foundation | No | Yes — Apache-2.0 |
| TimesFM 3.0 (Google) | foundation | No | **No** — non-commercial licence; benchmark only |
| PatchTST, default and tuned | deep learning | Yes | Yes |
| NPTS (the incumbent), ETS, Theta, DynamicOptimizedTheta | statistical | Refitted at every forecast | Yes |
| SeasonalNaive-24 | reference | — | — |

A plain-PyTorch MLP and linear baseline are compared separately in
[`docs/mlp_baseline.md`](docs/mlp_baseline.md).

## How the models are tested

- **Same rows for every model.** All 24 tanks, the same 24 forecast start times and six horizons
  (6 h, 12 h, 1 d, 2 d, 3 d, 7 d). The scorer fails the run if any two models differ by a row.
- **Every hour of the day.** Start times are 23 hours apart, so each hour of the day is a start
  time exactly once.
- **No look-ahead.** Models see only data up to each start time. Trained models are fitted before
  the first one, and every forecast window is checked.
- **Gaps stay gaps.** Missing sensor hours are kept as missing and scored for no model; they are
  never dropped or filled.
- **Significance, not just averages.** Every headline difference is tested with the paired
  bootstrap and Diebold–Mariano.

---

## Repository layout

| Path | Contents |
|---|---|
| `src/data/` | Data loading and curation (`curate.py`), the academic calendar, and the Waltr download tool (`waltr_fetch.py`) |
| `src/models/` | One runner per model, the shared backtest and metrics, scoring, significance tests, calibration, analysis and figures |
| `src/config.py` | Switches the pipeline between the two studies (environment variables) |
| `dataset/`, `dataset_2026-10-22/` | Raw per-tank daily JSON from Waltr — window 1 and window 2 |
| `results/chronos2/` | Window 1 (published study): tables, figures, calibration |
| `results/refresh_2026-10-22/` | Window 2 (refresh study), including `FINAL_ANALYSIS.md` |
| `eda/` | Sensor-integrity checks and trust tiers |
| `reports/` | Builders for the results page, the review deck and the 45-day explorer |
| `extension/` | The Waltr forecast dock (`waltr-dock/`), API and dashboard |
| `docs/` | Results reports, the reproduction guide and the real-time system design |
| `tests/` | 59 tests, no pytest needed |

## Quick start

```bash
git clone https://github.com/harsha135135/intelligent-water-management-system.git
cd intelligent-water-management-system
python3 -m venv venv && source venv/bin/activate      # Python 3.10+
pip install -r requirements.txt

python -m tests.test_metrics                          # must pass before trusting any number
python -m src.models.score_benchmark --strict         # window 1, from the prediction files
```

Prediction files and model weights are large and not committed; they regenerate in about two
hours. Full step-by-step reproduction with measured runtimes:
[`docs/reproducing.md`](docs/reproducing.md). The window-2 commands are in
[`reports/README.md`](reports/README.md).

## Documentation

| Document | Contents |
|---|---|
| [`results/refresh_2026-10-22/FINAL_ANALYSIS.md`](results/refresh_2026-10-22/FINAL_ANALYSIS.md) | **Final analysis** — both windows, significance, the daily view, the deployment recommendation |
| [`docs/phase3_results.md`](docs/phase3_results.md) | Phase III deliverable — the published study's results and inferences |
| [`docs/review_summary.md`](docs/review_summary.md) | Technical report — data, protocol, per-tank analysis, calibration, TimesFM (§17), Timestral (§18) |
| [`docs/reproducing.md`](docs/reproducing.md) | Reproduce every number, step by step |
| [`docs/architecture.md`](docs/architecture.md) | The forecasting engine and the measurement behind each design choice |
| [`docs/results_audit.md`](docs/results_audit.md) | Independent re-check of the committed results |
| [`docs/realtime_architecture.md`](docs/realtime_architecture.md) and [`docs/implementation_plan.md`](docs/implementation_plan.md) | The planned real-time system (with [`api_design`](docs/api_design.md), [`data_model`](docs/data_model.md), [`demo_plan`](docs/demo_plan.md), [`safety_and_controls`](docs/safety_and_controls.md)) |

## Status

| Component | Status |
|---|---|
| Data curation, backtest, metrics, significance tests | Done |
| 14-model benchmark in two windows, calibration, 45-day daily view | Done |
| Waltr forecast dock (offline, precomputed calibrated Timestral forecasts; Chronos-2 bundle beside it) | Done |
| Waltr data download (`waltr_fetch.py`) | Done — needs a browser session token, which lasts 15 minutes |
| Live Waltr data feed | Blocked — needs an issued service token |
| Real-time ingestion, alerts, decision engine, dashboard | Designed, not built — see the docs above |
| Motor control | Blocked — there is no motor API, and the data has no motor telemetry |

## Planned real-time system

The benchmark is an offline study. The next phase turns it into a running system: ingest tank
readings continuously, forecast every tank on a rolling basis, score each forecast once the actual
arrives, raise alerts, and propose motor actions through a separate safety gate. A replay mode will
run the historical data through the same pipeline, clearly labelled as replay and never as live.
**None of it is built yet**; the design documents are listed above.
