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

Measured on a single rolling-origin grid where all eleven models score the **same 188,664 rows**:

| | Chronos-2 zero-shot vs the deployed incumbent (NPTS) |
|---|---|
| **MASE improvement** | **5.7 % – 12.5 %**, at every one of six horizons |
| **Statistical significance** | Significant at **all six horizons**, by paired bootstrap (10,000 resamples over 24 origins) **and** Diebold–Mariano — all *p* < 0.01 |
| **Beaten significantly at every horizon** | **7 of 10** opponents — every model that is not a Chronos-2 covariate variant |
| **Tank-horizon cells won** | **101 / 144** vs the incumbent; **144 / 144** vs every classical method |
| **Healthy tanks significantly worse** | **0 / 15** — every significant loss is on a degraded or dead sensor |
| **Skill vs seasonal-naive** | Positive on **all 24 tanks** |
| **vs a PatchTST trained here** | **16.5 % – 18.6 % lower MASE** at every horizon, significant at all six, 120/144 cells |
| **Cost** | **89 seconds** for the entire backtest — the trained PatchTST needed 55 min, a **37×** difference |

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
| `src/models/` | **Forecasting** — `chronos2_forecasting.py` (the production model), `baselines_autogluon.py` (NPTS + classical), `patchtst_benchmark.py` (the trained deep control). **Evaluation** — `backtest.py` (the shared grid), `metrics.py`, `score_benchmark.py` (row parity, fatal under `--strict`), `significance.py` (paired bootstrap + Diebold-Mariano). **Reporting** — `unified_analysis.py` + `unified_figures.py` (every table and figure, all models), `calibration.py` + `calibrated_holdout.py`, `style.py` (one palette for every figure) |
| `dataset/` | Per-tank daily JSON — hourly inflow, outflow, opening and closing level. 2025-01-01 → 2026-04-22 |
| `eda/` | `eda_hourly.py` — mass-balance sensor integrity, trust tiers, covariate mutual-information study. Outputs `tank_trust.json` |
| `results/chronos2/` | `unified/` — the evidence: 10 CSVs, `summary.json` and **12 figures** covering all 11 models. `calibrated/` — conformal calibration and the 45-day operational view. `benchmark_table.md` and the metric CSVs. Model binaries, fitting scratch and prediction parquets are regenerable and not committed |
| `docs/` | Results reports and the full design specification for the real-time system |
| `tests/` | `test_metrics.py` — 6 tests, no pytest needed. Run these before trusting any number |
| `reports/` | `build_results_page.py` → the full HTML results page; `build_review_deck.py` → the PPTX review deck; `review_briefing.html` — the defence briefing. Both generated deliverables read the same CSVs, so they cannot disagree |
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
| [`docs/phase3_results.md`](docs/phase3_results.md) | **Phase III deliverable** — final results, significance testing, the calibration diagnosis, and the inference for each review category |
| [`docs/review_summary.md`](docs/review_summary.md) | 16-section technical report: dataset, evaluation protocol, per-tank analysis, covariate study, calibration, limitations |
| [`docs/architecture.md`](docs/architecture.md) | The forecasting engine, and the measurement behind each design choice |
| [`results/chronos2/benchmark_table.md`](results/chronos2/benchmark_table.md) | Complete benchmark: 11 models × 6 horizons × 6 metrics |
| [`results/chronos2/unified/`](results/chronos2/unified/) | The machine-readable evidence behind every claim: leaderboard, significance against every opponent, win matrix, per-tank, skill, calibration, cost |

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
| Chronos-2 inference (4 variants), benchmark, unified analysis, 12 figures | **Implemented** |
| PatchTST control (two configurations) | **Implemented** |
| Sensor trust tiering, mass-balance integrity (`eda/`) | **Implemented** |
| Significance against every opponent, calibration diagnosis | **Implemented** |
| Waltr forecast dock (offline, precomputed bundle) | **Implemented** |
| WALTR HTTP client | **Implemented but unusable** — requires a JWT; none is available |
| Real-time ingestion, state store, alerts, decision engine, Flutter dashboard | **Not built** — specified in Phases 1–8 |
| Conformal interval calibration, volume bias correction | **Implemented** — `calibration.py`, measured out of sample |
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
**~26 min total**. `--include-neural` would add PatchTST/TiDE to this same pass, but PatchTST is
run from its own module instead — see step 5b.

### 5b. Model execution — the PatchTST control

```bash
python -m src.models.patchtst_benchmark --preset default
python -m src.models.patchtst_benchmark --preset tuned \
    --out-name predictions_PatchTST-Tuned.parquet \
    --manifest-name patchtst_tuned_manifest.json \
    --model-dir results/chronos2/_patchtst_tuned_models
```

PatchTST is the **trained deep control**: the same patched-transformer family Chronos-2 belongs to,
but fitted on this campus rather than pretrained. One predictor per horizon, fitted strictly on
data at or before the first origin and rolled forward — the same protocol the statistical baselines
use, so nothing leaks and the row counts stay identical.

Two configurations are run because one would have been unfair. Chronos-2 conditions on 2,048 hours
of context; AutoGluon's default PatchTST sees 96. The `tuned` preset
uses the settings the PatchTST paper uses for hourly data — context 512 h,
100 epochs, 200 batches per epoch. Measured:
default **2.2 min**, tuned
**55 min**. Writes `predictions_PatchTST.parquet` and
`predictions_PatchTST-Tuned.parquet` (190,080 rows each), which `score_benchmark` picks up by glob.

Both are scored by the same `score_benchmark` pass as every other model, and compared in
step 7 alongside the rest of the field — there is no separate PatchTST study to run.

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

### 7. Unified analysis — every table on the results page

```bash
python -m src.models.unified_analysis      # ~4 min
```

One pass over **all 11 models** on the shared grid. Replaces the per-opponent studies that
preceded it (Chronos-2 vs NPTS, the covariate study, the PatchTST comparison), each of which
produced its own tables in its own shape. Writes to `results/chronos2/unified/`:

| File | Contents |
|---|---|
| `leaderboard.csv` | 66 rows — every model x horizon, macro and volume-weighted metrics, interval coverage and width, signed volume bias, rank |
| `significance_vs_all.csv` | 120 rows — Chronos-2 against **every** opponent, both metrics, all six horizons: paired bootstrap CI, bootstrap *p*, Diebold-Mariano statistic and *p*, origins won |
| `win_matrix_all_horizons.csv`, `win_matrix_h24.csv` | every ordered model pair, tank-horizon cells won |
| `per_tank.csv` | 1,584 rows — model x tank x horizon, with the sensor trust tier |
| `skill_h24.csv` | skill against the naive reference, per tank, for every model |
| `error_by_leadtime.csv`, `diurnal_h24.csv` | where the error lives, every model |
| `zero_inflation_h24.csv` | the interval mechanism: tail miss rates, how often p10 goes negative, what a clamp would do |
| `cost.csv` | measured wall clock against accuracy, read from the run manifests |
| `summary.json` | the headline numbers, so nothing downstream has to recompute them |

**A cross-check worth knowing about:** the unified pass computes the four macro metrics
independently of `score_benchmark`, and the two agree to **1e-16** across all 66 model x horizon
rows. Two implementations reaching the same number from the same parquets is a stronger statement
than either one alone.

### 8. Figures

```bash
python -m src.models.unified_figures       # ~25 s
python -m src.models.calibrated_holdout    # ~3 min, the operational view
```

Twelve figures to `results/chronos2/unified/plots/`, as **both PNG and SVG**. Every panel carries
every model; colour is assigned by family in `style.py` and never by rank, so a model keeps its
identity across the whole set.

| ID | Shows |
|---|---|
| U1 | Macro MASE vs horizon, all 11 models |
| U2 | The leaderboard as a heatmap, with rank in every cell |
| U3 | Chronos-2 against each opponent — 1 d with CIs, and across all six horizons |
| U4 | Win matrix: every ordered model pair, % of 144 tank-horizon cells |
| U5 | Per tank x per model MASE at 1 d, ordered by demand, labelled by sensor tier |
| U6 | Skill against the naive reference vs tank demand |
| U7 | Interval coverage against the width that bought it |
| U8 | Accuracy against measured compute |
| U9 | Error against lead time, out to 168 h |
| U10 | Error and bias by hour of day, against the demand profile |
| U11 | Why the intervals miss — tail asymmetry, negative p10, the clamp |
| U12 | Signed volume bias by model and horizon |

`calibrated_holdout` adds the three 45-day operational panels (one per headline model) and the
calibration measurements, fitted on 8 Jan - 8 Mar 2026 and reported on the disjoint
9 Mar - 22 Apr window.

### 9. The deliverables

```bash
python reports/build_results_page.py    # ~10 s -> reports/phase3_results_page.html
python reports/build_review_deck.py     # ~15 s -> reports/PW26_PK_06_phase3_review.pptx
```

Both read the same CSVs under `results/chronos2/unified/`, so the page and the deck cannot drift
from the data or from each other. Neither output is committed: each embeds figures that are
already tracked as PNG and SVG.

### Expected files after a full run

```
results/chronos2/
├── predictions_*.parquet                  190,080 rows each (7 files, not committed)
├── run_manifest.json / baselines_manifest.json / patchtst*_manifest.json
├── benchmark_table.md
├── metrics_by_horizon.csv                 66 rows  (11 models x 6 horizons)
├── metrics_per_tank.csv                   1,584 rows
├── unified/
│   ├── 10 CSVs + summary.json
│   └── plots/  12 figures x {png, svg}
└── calibrated/
    ├── calibration parameters and daily tables
    └── plots/  3 operational panels x {png, svg}
```

Total end-to-end: **~2 hours**, almost all of it the two forecasting steps. Re-running only the
analysis and reporting from existing predictions takes **~8 minutes**.

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
* **PatchTST is in scope and on the grid.** It is run from `src/models/patchtst_benchmark.py` at
  two configurations and scored on the identical rows, then compared in the same unified pass as
  every other model. The older `results/patchtst/` directory is a *different*, non-comparable
  evaluation and is not used anywhere.

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
