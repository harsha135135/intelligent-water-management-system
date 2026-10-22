# Reproducing the benchmark

Step-by-step reproduction of the published study (evaluation window 24 Mar – 22 Apr 2026), with
the measured runtime of every stage. It was moved here from the top-level README unchanged.

The second study, on the October 2026 Waltr snapshot, runs the same pipeline with three
environment variables set (`PESU_DATASET_DIR`, `PESU_RESULTS_DIR`, `PESU_EDA_DIR`) and writes to
`results/refresh_2026-10-22/`. Its commands, including the old-vs-new comparison and TimesFM 3.0,
are in [`reports/README.md`](../reports/README.md).

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

## Reproducing the final benchmark

The Chronos-2 vs NPTS evaluation reported in [`docs/review_summary.md`](review_summary.md)
and [`results/chronos2/benchmark_table.md`](../results/chronos2/benchmark_table.md) is reproduced by
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
`predictions_PatchTST-Tuned.parquet` (190,080 rows each), which `score_benchmark` picks up by glob —
for every model on the benchmark roster (`unified_analysis.ORDER`); a predictions file holding any
other model is work in progress and is skipped with a warning, never silently scored.

Both are scored by the same `score_benchmark` pass as every other model, and compared in
step 7 alongside the rest of the field — there is no separate PatchTST study to run.

### 5c. Model execution — the PyTorch MLP baseline

```bash
python -m tests.test_mlp                 # 8/8
python -m src.models.mlp_forecaster      # ~50 s on CPU
python -m src.models.mlp_comparison      # needs the step 4-5b parquets
```

Per-tank MLP (168 → 64 → 32 → 24) and linear models in plain PyTorch, fitted on data before the
first origin with a chronological validation split. Predictions go to `results/chronos2/mlp/`,
outside the glob `score_benchmark` and `unified_analysis` read, so the benchmark roster's results are unchanged.
Results and interpretation: [`docs/mlp_baseline.md`](mlp_baseline.md).

### 5d. Model execution — TimesFM 2.5, the second foundation model

```bash
python -m src.models.timesfm_forecasting      # ~3 min on CPU; writes predictions_TimesFM2.5-ZS.parquet
python -m tests.test_timesfm                  # 16 tests, stubbed model, no download
```

Google's `google/timesfm-2.5-200m-pytorch`, zero-shot, through the official `timesfm` package
(pinned in `requirements.txt`; the checkpoint revision is pinned in the module, and the weights
download once to the standard Hugging Face cache, never into this repository). The protocol is
Chronos-2 zero-shot's: the same grid, the target series only, a 2,048-hour context truncated at
each origin, and the median as the point forecast. The runner checks
`context_end <= origin < forecast_start` for every (origin, horizon, tank) window, writes
`timesfm_manifest.json` (checkpoint, config, device, measured timing, peak memory, the leakage and
missing-data audits), and refuses to write the canonical parquet if any series failed — the failures
go to `_timesfm_partial/` and the command exits non-zero. Forecasts are cached per (origin, horizon)
block under `results/chronos2/_timesfm_cache/`, keyed on the checkpoint, config and exact context
values, so a rerun with identical inputs recomputes nothing; `--no-cache` forces a fresh, timed run.

The official 2.5 code runs on CUDA or CPU, not Apple MPS, so it ran on CPU (measured **158 s**). For
a same-hardware comparison, Chronos-2 zero-shot was timed on the same CPU with
`python -m src.models.chronos2_forecasting --variants Chronos2-ZS --device cpu --out-dir results/chronos2/cpu_timing`
(**68 s**; timing only, not scored). TimesFM 3.0 is newer but its weights are licensed for
non-commercial, non-production use, so the deployable 2.5 checkpoint is the one evaluated.
Full comparison: [`docs/review_summary.md` §17](review_summary.md#17-timesfm-comparison).

### 5e. Model execution — Timestral, the third foundation model

```bash
python -m src.models.timestral_forecasting                 # ~3 min on CPU; predictions_Timestral-209M-ZS.parquet
python -m src.models.timestral_forecasting --size 5M       # also 28M, 123M: off the roster
python -m src.models.timestral_forecasting --sizes-report  # all four sizes, same rows -> timestral_sizes.csv
python -m tests.test_timestral                             # 15 tests, stubbed model, no download
```

Mistral's `mistralai/Timestral-209M`, zero-shot, through Mistral's `timestral` package (not on PyPI;
`requirements.txt` pins its GitHub commit, and the module pins each checkpoint revision). The
protocol and the plumbing are TimesFM's: the same grid, target only, 2,048-hour context, the median
as the point forecast, and `timesfm_forecasting`'s leakage check, failure isolation and block cache,
with Timestral's quantiles handed over by level value. Missing hours are passed as NaN, which
Timestral masks, as Chronos-2 does. Run on the same CPU as TimesFM (measured **145 s**). The code is
Apache-2.0; the weights are under a modified MIT licence that withholds all rights from a licensee
whose global monthly revenue exceeds USD 20M — production use is allowed below that. Only the 209M
size is on the roster; the 5M, 28M and 123M runs are scored only in `timestral_sizes.csv`.
Full comparison: [`docs/review_summary.md` §18](review_summary.md#18-timestral-comparison).

To rebuild the complete comparison — every table, figure and generated document — from the
existing prediction files (~2 min; the TimesFM and Timestral steps are served from their caches):

```bash
python -m src.models.timesfm_forecasting                 # cached: no model call if inputs unchanged
python -m src.models.timestral_forecasting               # cached likewise
python -m src.models.timestral_forecasting --sizes-report
python -m src.models.score_benchmark --strict            # metrics, benchmark_table.md §8-9, the CSVs
python -m src.models.unified_analysis                    # leaderboard, significance, win matrix, cost
python -m src.models.unified_figures                     # U1-U12, PNG + SVG
python -m src.models.calibrated_holdout --add timesfm timestral   # the 45-day view (others untouched)
python reports/build_timesfm_section.py                  # docs/review_summary.md §17
python reports/build_timestral_section.py                # docs/review_summary.md §18
python reports/build_results_page.py && python reports/build_review_deck.py
python reports/build_holdout45_page.py
```

### 6. Evaluation and scoring

```bash
python -m src.models.score_benchmark --strict
```

Reads every `results/chronos2/predictions_*.parquet`, attaches the seasonal-naive scales from
pre-origin history only, asserts no row has `timestamp <= origin`, and — with `--strict` — **fails
the run** if any two models were scored on different row counts or any row has an actual but no
prediction (a failed forecast is never averaged away). Only models on the roster are read. Writes:

| File | Contents |
|---|---|
| `results/chronos2/benchmark_table.md` | the complete benchmark document: evaluation grid, headline three-model comparison per metric, all 13 models × 6 horizons × 6 metrics, covariate table, calibration table, per-tank tables, §8 — TimesFM 2.5 against Chronos-2, NPTS and SeasonalNaive-24 — and §9, Timestral 209M against the same three and TimesFM |
| `results/chronos2/metrics_by_horizon.csv` | one row per (model, horizon) — macro + volume-weighted metrics, p10–p90 coverage and width |
| `results/chronos2/metrics_per_tank.csv` | one row per (model, horizon, tank) |
| `results/chronos2/per_tank_comparison.csv` | 144 rows — Chronos-2 vs NPTS vs SeasonalNaive per tank and horizon, with improvement %, winner and sensor trust tier |
| `results/chronos2/timesfm_comparison.csv` | 72 rows — TimesFM vs Chronos-2, NPTS and SeasonalNaive-24 per horizon and metric: both scores, the relative difference in words, tanks won / lost / tied, median and mean per-tank difference |
| `results/chronos2/timesfm_per_tank.csv` | 144 rows — TimesFM vs Chronos-2 per tank and horizon on all four metrics, winner on MASE, sensor trust tier |
| `results/chronos2/timestral_comparison.csv`, `timestral_per_tank.csv` | 96 / 144 rows — the same two tables for Timestral, which is also compared against TimesFM |

Runtime: ~7 s.

### 7. Unified analysis — every table on the results page

```bash
python -m src.models.unified_analysis      # ~45 s
```

One pass over **all 13 models** on the shared grid. Replaces the per-opponent studies that
preceded it (Chronos-2 vs NPTS, the covariate study, the PatchTST comparison), each of which
produced its own tables in its own shape. Writes to `results/chronos2/unified/`:

| File | Contents |
|---|---|
| `leaderboard.csv` | 78 rows — every model x horizon, macro and volume-weighted metrics, interval coverage and width, signed volume bias, rank |
| `significance_vs_all.csv` | 144 rows — Chronos-2 against **every** opponent, both metrics, all six horizons: paired bootstrap CI, bootstrap *p*, Diebold-Mariano statistic and *p*, origins won. Opponents are visited in the order they joined the study, so adding a model never re-draws the bootstrap intervals already published |
| `significance_timesfm.csv` | 84 rows — TimesFM against every non-foundation model, by the same two tests |
| `significance_timestral.csv` | 132 rows — Timestral against every model except Chronos-2 (already a row above). Drawn after every other table from the same seeded stream, so adding it re-drew no published interval |
| `win_matrix_all_horizons.csv`, `win_matrix_h24.csv` | every ordered model pair, tank-horizon cells won |
| `per_tank.csv` | 1,872 rows — model x tank x horizon, with the sensor trust tier |
| `skill_h24.csv` | skill against the naive reference, per tank, for every model |
| `error_by_leadtime.csv`, `diurnal_h24.csv` | where the error lives, every model |
| `zero_inflation_h24.csv` | the interval mechanism: tail miss rates, how often p10 goes negative, what a clamp would do |
| `cost.csv` | measured wall clock against accuracy, read from the run manifests, with the device each run recorded and the same-CPU timings for the three foundation models |
| `summary.json` | the headline numbers, so nothing downstream has to recompute them |

**A cross-check worth knowing about:** the unified pass computes the four macro metrics
independently of `score_benchmark`, and the two agree to **≤ 3.3e-16** across all 78 model x horizon
rows. Two implementations reaching the same number from the same parquets is a stronger statement
than either one alone.

### 8. Figures

```bash
python -m src.models.unified_figures       # ~5 s
python -m src.models.calibrated_holdout    # ~3 min, the operational view
python -m src.models.calibrated_holdout --add timesfm   # adds TimesFM to it; other rows untouched
python -m src.models.calibrated_holdout --add timestral # adds Timestral likewise
```

Twelve figures to `results/chronos2/unified/plots/`, as **both PNG and SVG**. Every panel carries
every model; colour is assigned by family in `style.py` and never by rank, so a model keeps its
identity across the whole set.

| ID | Shows |
|---|---|
| U1 | Macro MASE vs horizon, all 13 models |
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
9 Mar - 22 Apr window. `--add timesfm` adds TimesFM 2.5 to the same windows (panels
`T_holdout45_timesfm_calibrated` and `T_tanks45_timesfm`) by appending its rows to every table;
every table is per model, so the three published models' rows stay byte-identical. `--add timestral`
does the same for Timestral 209M (`TS_holdout45_timestral_calibrated`, `TS_tanks45_timestral`). The
campus panels share one y-axis, so TimesFM's 17 Mar overshoot widens it for all five. `--plots`
redraws the panels without recomputing anything.

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
├── predictions_*.parquet                  190,080 rows each (9 roster files, not committed)
├── run_manifest.json / baselines_manifest.json / patchtst*_manifest.json / timesfm_manifest.json
├── timestral_{5m,28m,123m,209m}_manifest.json
├── benchmark_table.md
├── metrics_by_horizon.csv                 78 rows  (13 models x 6 horizons)
├── metrics_per_tank.csv                   1,872 rows
├── timesfm_comparison.csv / timesfm_per_tank.csv   72 / 144 rows
├── timestral_comparison.csv / timestral_per_tank.csv   96 / 144 rows
├── timestral_sizes.csv                    24 rows  (4 sizes x 6 horizons)
├── cpu_timing/                           same-CPU Chronos-2 timing reference (manifest + README)
├── unified/
│   ├── 12 CSVs + summary.json
│   └── plots/  12 figures x {png, svg}
└── calibrated/
    ├── calibration parameters and daily tables
    └── plots/  5 campus + 5 per-tank operational panels x {png, svg}
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
* **No leakage.** Baselines are fitted only on data at or before the first origin; Chronos-2,
  TimesFM and Timestral are zero-shot with context truncated at each origin (the TimesFM and
  Timestral runners assert `context_end <= origin < forecast_start` on every window); every scored row satisfies
  `timestamp > origin`; MASE/RMSSE denominators use pre-origin history only.
* **Gapless hourly reindex.** Missing hours become NaN, never dropped — collapsing a gap would
  shift later timestamps and corrupt the 24-hour seasonality the metrics scale against.
* **Sanity check to look for first:** SeasonalNaive-24 must score MASE ≈ 1 (measured
  0.967–1.069). If it does not, the scaling denominator is wrong and every number is wrong.
* **PatchTST is in scope and on the grid.** It is run from `src/models/patchtst_benchmark.py` at
  two configurations and scored on the identical rows, then compared in the same unified pass as
  every other model. The older `results/patchtst/` directory is a *different*, non-comparable
  evaluation and is not used anywhere.
