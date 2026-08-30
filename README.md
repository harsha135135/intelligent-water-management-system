# Intelligent Water Management System — Predictive Analytics for a Sustainable Campus

**Project ID:** PW26_PK_06
**Institution:** PES University, Ring Road Campus
**Guides:** Ms. Preet Kanwal · Mr. Prasad B Honnavalli
**Team:** Abhay Patil · Amogh E M · Harshavardhan M · Viraj Ved Shankar

Hourly water-demand forecasting for **24 distribution tanks** on the PES University RR campus,
built on **Chronos-2** (`amazon/chronos-2`), a pretrained time-series foundation model used
zero-shot — no training, no fine-tuning.

---

## Headline result

Measured on a single rolling-origin grid where all nine models score the **same 188,664 rows**:

| | Chronos-2 zero-shot vs the deployed incumbent (NPTS) |
|---|---|
| **MASE improvement** | **5.7 % – 12.5 %**, at every one of six horizons |
| **Statistical significance** | Significant at **all six horizons**, by paired bootstrap (10,000 resamples over 24 origins) **and** Diebold–Mariano — all *p* < 0.01 |
| **Tank-horizon cells won** | **101 / 144** (70 %) |
| **Healthy tanks significantly worse** | **0 / 15** — every significant loss is on a degraded or dead sensor |
| **Skill vs seasonal-naive** | Positive on **all 24 tanks** |
| **Cost** | **89 seconds** for the entire 24-origin × 6-horizon backtest |

Two caveats, both measured and both stated wherever the results appear:

- **Prediction intervals were overconfident** — p10–p90 covered **0.714–0.743** against a nominal
  0.80. Cause identified here: **~24 % of hourly readings are exactly zero**, and a
  continuous-density model puts its p10 above zero on 79 % of rows, so two-thirds of its
  lower-tail misses are zero-demand hours. **Corrected** by asymmetric conformal calibration:
  coverage **0.741 → 0.785**, measured on a disjoint later window.
- **24-hour volume was under-forecast by 12.15 %.** **Corrected** by a per-tank multiplicative
  factor: **−10.5 % → −1.8 %**, cutting daily campus error from 36.7 to 28.8 KL. A refill should
  still be sized on the calibrated interval, not the raw mean.

Full analysis: [`docs/phase3_results.md`](docs/phase3_results.md) ·
[`docs/review_summary.md`](docs/review_summary.md) ·
[`results/chronos2/benchmark_table.md`](results/chronos2/benchmark_table.md)

---

## Repository layout

| Path | Contents |
|---|---|
| `src/data/` | `curate.py` — 26 directories → 24 tanks, gapless hourly reindex, raises on drift. `calendar_pesu.py` — academic-calendar covariates |
| `src/models/` | `chronos2_forecasting.py` (the production model), `backtest.py` (shared evaluation grid), `metrics.py` (MAE/RMSE/MASE/RMSSE), `baselines_autogluon.py`, `score_benchmark.py`, `review_package.py`, `review_plots.py`, `phase3_analysis.py` |
| `dataset/` | Per-tank daily JSON — hourly inflow, outflow, opening and closing level. 2025-01-01 → 2026-04-22 |
| `eda/` | `eda_hourly.py` — mass-balance sensor integrity, trust tiers, covariate mutual-information study. Outputs `tank_trust.json` |
| `results/chronos2/` | Benchmark metrics, per-tank tables, and **23 figures** (A–N, O–W). Model binaries and prediction parquets are regenerable and not committed |
| `docs/` | Results reports and the full design specification for the real-time system |
| `tests/` | `test_metrics.py` — 6 tests, no pytest needed. Run these before trusting any number |
| `extension/` | FastAPI service, Next.js dashboard, and the Waltr MV3 forecast dock |
| `water_forecast_dash/` | Standalone Flask dashboard prototype |
| `notebooks/` | Exploratory analysis |

---

## Quick start

```bash
git clone https://github.com/harsha135135/<repo>.git && cd <repo>
python3 -m venv venv && source venv/bin/activate     # Python 3.10+ (3.13 used for the published run)
pip install -r requirements.txt
```

Then, in order:

```bash
python -m src.data.curate          # 26 dirs -> 24 tanks, 270,849 rows.  ~4 s
python -m tests.test_metrics       # 6/6 must pass. If they don't, no number below is valid
python -m src.models.score_benchmark --strict   # regenerates every metric.  ~7 s
```

`score_benchmark` needs the prediction parquets, which are **not committed** (they are large and
fully regenerable). To produce them:

```bash
python -m src.models.chronos2_forecasting --variants Chronos2-ZS   # ~1.5 min
python -m src.models.baselines_autogluon                           # ~26 min
```

Full step-by-step reproduction, with measured runtimes for every stage, is in
[Reproducing the final benchmark](#reproducing-the-final-benchmark) below.

---

## Documentation

### Results

| Document | Contents |
|---|---|
| [`docs/phase3_results.md`](docs/phase3_results.md) | **Phase III deliverable** — final results, significance testing, the calibration diagnosis, 23 figures, and the inference for each review category |
| [`docs/review_summary.md`](docs/review_summary.md) | 16-section technical report: dataset, evaluation protocol, per-tank analysis, covariate study, calibration, limitations |
| [`docs/architecture.md`](docs/architecture.md) | The forecasting engine, and the measurement behind each design choice |
| [`results/chronos2/benchmark_table.md`](results/chronos2/benchmark_table.md) | Complete benchmark: 9 models × 6 horizons × 6 metrics |

### Real-time system — designed, not yet built

The forecasting study is complete. Turning it into a running system is specified across six
documents; **none of the pipeline they describe is implemented yet.**

| Document | Contents |
|---|---|
| [`docs/realtime_architecture.md`](docs/realtime_architecture.md) | Master architecture — adapter seam, component catalogue, 10 Mermaid diagrams, quality rules, feedback loop, degradation matrix |
| [`docs/implementation_plan.md`](docs/implementation_plan.md) | Phases 0–11 with files, dependencies, tests, acceptance criteria, risks |
| [`docs/api_design.md`](docs/api_design.md) | REST surface, SSE event types, transport comparison, auth and roles |
| [`docs/data_model.md`](docs/data_model.md) | 20 entities with fields, indexes, retention |
| [`docs/demo_plan.md`](docs/demo_plan.md) | Historical replay, the actual-vs-predicted reveal, scenarios mined from the real record |
| [`docs/safety_and_controls.md`](docs/safety_and_controls.md) | Motor-control safety, per-tank limits derived from measurement, fail-safe matrix |

---

## Status

| Component | Status |
|---|---|
| Data curation, calendar features, backtest harness, metrics | **Implemented** |
| Chronos-2 inference, benchmark, review package, 23 figures | **Implemented** |
| Sensor trust tiering, mass-balance integrity (`eda/`) | **Implemented** |
| Significance testing, calibration diagnosis (`phase3_analysis.py`) | **Implemented** |
| Waltr forecast dock (offline, precomputed bundle) | **Implemented** |
| WALTR HTTP client | **Implemented but unusable** — requires a JWT; none is available |
| Real-time ingestion, state store, alerts, decision engine, Flutter dashboard | **Not built** — specified in Phases 1–8 |
| Conformal interval calibration, volume bias correction | **Not built** — Phase 9 |
| Live WALTR data feed | **Blocked** — requires an issued service token |
| Real motor control | **Blocked** — no motor API exists, and the dataset contains **no motor telemetry** |

---

## Notes on the data

- **24 physical tanks, not 26.** `dataset/` contains 26 directories; two are longer re-scrapes of
  existing tanks. `src/data/curate.py` keeps the superset, drops the stale copy, and **raises** if
  the count is not exactly 24.
- **Missing hours are held open as NaN, never dropped.** Collapsing a gap would shift every later
  timestamp and destroy the 24-hour seasonality the metrics scale against.
- **Sensor quality varies materially.** 15 tanks are healthy, 6 degraded, 3 dead (73–96 % zero
  hours). Tiers are in `eda/tank_trust.json`, derived from a mass-balance identity test.
- **Model binaries and prediction parquets are not committed.** They total ~380 MB and regenerate
  in about an hour. What is committed is the evidence — metrics, tables and figures.

---
## Reproducing the final benchmark

The Chronos-2 vs NPTS evaluation reported in [`docs/review_summary.md`](docs/review_summary.md)
and [`results/chronos2/benchmark_table.md`](results/chronos2/benchmark_table.md) is reproduced by
the commands below, in order. All are run from the repository root with `venv/` activated.
Runtimes are the **measured wall clock from the actual run** (Apple MPS laptop GPU, Python
3.13.3); they are recorded in `logs/` and in `results/chronos2/run_manifest.json`.

```bash
source venv/bin/activate
```

### 1. Data preparation

```bash
python -m src.data.curate
```

Loads `dataset/`, drops the two duplicated tank directories, relabels the surviving copies,
reindexes every tank onto a gapless hourly range, and prints a per-tank curation receipt.
Expect: `26 directories -> 24 tanks`, `rows=270,849`, range `2025-01-01 00:00 .. 2026-04-22 23:00`.
The loader **raises** if the tank count is not exactly 24. Runtime: ~4 s. Writes no files —
every downstream script calls `load_curated_hourly()` itself.

### 2. Metric tests — run these before trusting any number

```bash
python -m tests.test_metrics    # prints PASS/FAIL per test, exits non-zero on failure
```

6 tests, no pytest required. The load-bearing one is `test_seasonal_naive_mase_is_about_one`:
MASE is defined so a seasonal-naive forecast scores 1.0, and if that identity fails every number
in the benchmark is wrong. Runtime: ~3 s.

### 3. EDA — trust tiers and the covariate feature study

```bash
python eda/eda_hourly.py
```

Writes `eda/tank_trust.json` (healthy / degraded / dead per tank),
`eda/tank_mass_balance.csv`, `eda/tank_forecastability.csv`,
`eda/feature_mi_by_horizon.csv` and `eda/eda_hourly_report.txt`. The benchmark's per-tank tables
join against `tank_trust.json` and `tank_mass_balance.csv`, so run this before step 6.
Runtime: ~25 s; the outputs are deterministic and reproduce byte-identically.

### 4. Model execution — Chronos-2

```bash
# the proposed production model, on its own (89 s)
python -m src.models.chronos2_forecasting --variants Chronos2-ZS

# all four variants, as used for the covariate study (33 min total)
python -m src.models.chronos2_forecasting     --variants Chronos2-ZS Chronos2-COV Chronos2-COV-LEAN Chronos2-COV-XL
```

Zero-shot — nothing is fitted. Writes `results/chronos2/predictions_<variant>.parquet`
(190,080 rows each) and `results/chronos2/run_manifest.json` (origins, tanks, horizons,
per-variant `wall_clock_s`). Measured: ZS **1.5 min**, COV-LEAN 5.9 min, COV-XL 10.5 min,
COV 15.5 min. Useful flags: `--device cpu`, `--context 2048`, `--max-tanks N` for a smoke test.

### 5. Model execution — baselines (incumbent + reference)

```bash
python -m src.models.baselines_autogluon
```

Fits one AutoGluon predictor per horizon on data **at or before the first origin**
(2026-03-24 22:00) and rolls it forward over all 24 origins — `prediction_length` is fixed at fit
time, so each horizon needs its own predictor. Produces NPTS (the incumbent), SeasonalNaive
(the reference), ETS, Theta and DynamicOptimizedTheta in one pass. Writes
`results/chronos2/predictions_autogluon_baselines.parquet` (950,400 rows) and
`baselines_manifest.json`. Measured per horizon: 3.8 / 5.0 / 5.0 / 3.7 / 1.9 / 6.8 min —
**~26 min total**. Add `--include-neural` for PatchTST/TiDE (**not** part of this review; PatchTST
is out of scope).

### 6. Evaluation and scoring

```bash
python -m src.models.score_benchmark --strict
```

Reads every `results/chronos2/predictions_*.parquet`, attaches the seasonal-naive scales from
pre-origin history only, asserts no row has `timestamp <= origin`, and — with `--strict` — **fails
the run** if any two models were scored on different row counts. Writes:

| File | Contents |
|---|---|
| `results/chronos2/benchmark_table.md` | the complete benchmark document: evaluation grid, headline three-model comparison per metric, all 9 models × 6 horizons × 6 metrics, covariate table, calibration table, per-tank tables |
| `results/chronos2/metrics_by_horizon.csv` | one row per (model, horizon) — macro + volume-weighted metrics, p10–p90 coverage and width |
| `results/chronos2/metrics_per_tank.csv` | one row per (model, horizon, tank) |
| `results/chronos2/per_tank_comparison.csv` | 144 rows — Chronos-2 vs NPTS vs SeasonalNaive per tank and horizon, with improvement %, winner and sensor trust tier |

Runtime: ~7 s.

### 7. Review package (CSVs)

```bash
python -m src.models.review_package
```

Writes 10 CSVs under `results/chronos2/review/`: `headline_comparison.csv`,
`macro_metrics_all_models.csv`, `per_tank_metrics_all_models.csv`,
`per_tank_chronos2_vs_npts.csv`, `per_tank_practical_accuracy_h24.csv`,
`per_tank_practical_accuracy_h168.csv`, `per_tank_daily_volume_accuracy.csv`
(24 h volume error per tank — the "water requirement" view), `volume_bias.csv`
(signed under/over-forecast per model and horizon), `final_holdout_macro.csv`,
`final_holdout_per_tank.csv`, plus `review_manifest.json`. Re-scores nothing.
Runtime: ~6 s.

### 8. Plots

```bash
python -m src.models.review_plots
```

Writes 14 figures as **both PNG and SVG** to `results/chronos2/review/plots/`:

| ID | File | Shows |
|---|---|---|
| A–D | `A_mase_vs_horizon`, `B_rmse_vs_horizon`, `C_mae_vs_horizon`, `D_rmsse_vs_horizon` | each metric by horizon, Chronos-2 / NPTS / SeasonalNaive |
| E | `E_per_tank_mase_24h` | per-tank MASE at 1 d, all 24 tanks |
| F | `F_per_tank_improvement_24h` | signed per-tank improvement %, positive and negative |
| G | `G_per_tank_mase_heatmap` | Chronos-2 MASE, tank × horizon |
| H | `H_error_distribution_24h` | error distribution and absolute-error percentiles |
| I | `I_actual_vs_predicted_24h` | actual vs prediction with p10–p90 band |
| J | `J_final_holdout_7day` | 7-day final-holdout forecast with p10–p90 band |
| K | `K_interval_calibration` | coverage vs the nominal 0.80 |
| L | `L_variant_selection` | accuracy per minute of compute |
| M | `M_tanks_won` | number and % of tanks won, per horizon |
| N | `N_covariate_vs_zeroshot` | covariate gain against its cost and the incumbent gap |

Representative tanks in I and J are chosen **by measured role** (highest / median / lowest demand
among live tanks, plus the highest-MASE tank) and recorded in
`results/chronos2/review/representative_tanks.json`, so the panel cannot be cherry-picked.
Runtime: ~10 s.

### Expected files after a full run

```
results/chronos2/
├── predictions_Chronos2-ZS.parquet            190,080 rows
├── predictions_Chronos2-COV.parquet           190,080 rows
├── predictions_Chronos2-COV-LEAN.parquet      190,080 rows
├── predictions_Chronos2-COV-XL.parquet        190,080 rows
├── predictions_autogluon_baselines.parquet    950,400 rows  (5 models)
├── run_manifest.json / baselines_manifest.json
├── benchmark_table.md
├── metrics_by_horizon.csv                     54 rows  (9 models × 6 horizons)
├── metrics_per_tank.csv                    1,296 rows  (9 × 6 × 24)
├── per_tank_comparison.csv                   144 rows  (24 tanks × 6 horizons)
└── review/
    ├── 10 CSVs + review_manifest.json + representative_tanks.json
    └── plots/  14 figures × {png, svg}
docs/review_summary.md
```

Total end-to-end: **~60 minutes**, of which ~59 min is the two forecasting steps (33 min
Chronos-2, 26 min baselines). Re-running only steps 6–8 from existing predictions takes
**~23 seconds**.

### Methodology details that matter

These are properties of the evaluation, not options — changing any of them invalidates
comparison with the published numbers.

* **24 physical tanks.** `dataset/` has 26 directories; `GJBC_BLOCK_1_A1_BOY_S` and
  `GJBC_BLOCK_1_A2_GIRL_S` are longer re-scrapes of two existing tanks. `curate.py` keeps the
  superset, drops the stale copy, and raises if the count is not 24.
* **24 origins at a 23-hour stride, not 24.** A 24-hour stride puts every origin at the same
  clock hour, so the 6 h horizon would only ever be scored on the quiet 00:00–05:00 window.
  23 is co-prime with 24, so the 24 origins visit **every hour-of-day exactly once**.
* **Six horizons:** 6 h, 12 h, 1 d, 2 d, 3 d, 7 d.
* **Identical evaluation rows for every model** — 3,405 / 6,861 / 13,530 / 27,323 / 41,140 /
  96,405 per horizon, enforced by `backtest.assert_comparable()` and fatal under `--strict`.
* **No leakage.** Baselines are fitted only on data at or before the first origin; Chronos-2 is
  zero-shot with context truncated at each origin; every scored row satisfies
  `timestamp > origin`; MASE/RMSSE denominators use pre-origin history only.
* **Gapless hourly reindex.** Missing hours become NaN, never dropped — collapsing a gap would
  shift later timestamps and corrupt the 24-hour seasonality the metrics scale against.
* **Sanity check to look for first:** SeasonalNaive-24 must score MASE ≈ 1 (measured
  0.967–1.069). If it does not, the scaling denominator is wrong and every number is wrong.
* **PatchTST is out of scope** for this comparison and appears in none of the tables above.

---

## Planned real-time system

The benchmark above is an **offline study**. The next phase turns it into a running real-time
forecasting, monitoring, alerting and motor decision-support system. That system is **designed but
not yet built** — the documents below are the blueprint, and no part of the pipeline they describe
exists in this repository yet.

### What it will do

Ingest tank readings continuously → validate and quality-flag them → hold current state per tank →
generate rolling Chronos-2 forecasts at six horizons → record every prediction before the answer
is known → score it against the actual when that arrives → monitor rolling accuracy and drift →
raise deduplicated, severity-ranked alerts → propose motor actions through a decision engine and a
separate safety gate → and show all of it in a Flutter Web dashboard where **every number carries
its provenance**.

### Demo vs live — the distinction is structural

The demo replays the historical dataset through the production pipeline as if it were arriving
now. It is **not** live data and never claims to be:

* `mode` (`replay` | `live`) is a column on every fact table, a field in every API response and
  SSE event, and a non-dismissible banner in the UI.
* Forecasts during a replay are **real Chronos-2 inference** computed from the context available
  at that simulated moment — not precomputed and not replayed.
* Deleting a replay session deletes every row it created.

Switching to production is a configuration change (`SOURCE=waltr MOTOR=waltr CLOCK=real
MODE=live`), not a rewrite: validation, state, forecasting, accuracy, decisions, alerts, the API
and the UI are all source-agnostic by construction.

See [Documentation](#documentation) above for the six design documents, and [Status](#status) for what is and is not built.

Nothing in the planned system is running today.
