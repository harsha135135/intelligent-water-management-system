# Audit of the existing results

Performed 2026-09-23 on branch `pytorch-mlp-baseline`, created from `unified-all-model-results`
(55f0fe3). The aim was to check that every committed number can be regenerated from the code,
and that the headline figures say what the README claims.

## 1. Reproduction: the committed tables regenerate byte-identically

The eleven prediction parquets are not committed (≈380 MB, see `.gitignore`), so the audit used
the parquets from the original run and re-executed only the scoring and analysis code:

```bash
python -m src.models.score_benchmark --strict   # row parity enforced
python -m src.models.unified_analysis
git status --short                               # -> nothing
```

Every regenerated file was identical to the committed version: `metrics_per_tank.csv`,
`metrics_by_horizon.csv`, `per_tank_comparison.csv`, `benchmark_table.md`, and all ten
`unified/*.csv` plus `summary.json` (the bootstrap is seeded, so the significance tables reproduce
exactly too). `tests/test_metrics.py`: 6/6 pass.

**Scope of that claim.** This re-verifies scoring, aggregation and significance testing. It
does not re-run Chronos-2, AutoGluon or PatchTST inference; the forecasts themselves are taken
as produced.

## 2. The headline improvement exists in two aggregations

| Source | How MASE is averaged | Chronos-2 vs NPTS |
|---|---|---|
| `metrics_by_horizon.csv` / `benchmark_table.md` | per tank over all rows, then mean over tanks | **5.73 – 12.43 %** |
| `unified/significance_vs_all.csv` | per (origin, tank), mean over tanks, then mean over origins | **5.73 – 12.50 %** |

Both are correct for their table. The README's "12.5 %" comes from the significance table; the
benchmark table rounds to 12.4 %. Quote the one whose table you cite.

## 3. SeasonalNaive MASE is near 1, not exactly 1

SeasonalNaive scores 0.97 – 1.07 MASE across horizons. The denominator is the seasonal-naive
error on the history *before* each origin, and the numerator is the error on the evaluation
window, so 1.0 is only expected, not guaranteed. `test_seasonal_naive_mase_is_about_one` tests
exactly that approximate identity.

## 4. Calibration figures verified

`results/chronos2/calibrated/`: Chronos-2 zero-shot, 24-hour horizon, p10–p90 coverage
**0.7406 → 0.7853** (nominal 0.80). The 60-day calibration window (2026-01-08 → 03-08) and the
45-day reported window (03-09 → 04-22) are disjoint (`windows_disjoint: true`).

## 5. Where each claim lives

| Claim | Public `main` | `unified-all-model-results` |
|---|---|---|
| Chronos-2 vs NPTS, 6 horizons, bootstrap + DM | ✅ (9 models, `phase3_analysis.py`) | ✅ (11 models, `significance.py`) |
| PatchTST on the shared grid, 16.5–18.6 % | ❌ | ✅ |
| PatchTST architecture search | ❌ (local-only, uncommitted) | ❌ |
| MLP / Linear-AR baselines | ❌ | ❌ — this branch |

## 6. Other defects found (not fixed on this branch)

- `extension/api`: 2 of 8 tests fail. `app/auth.py` raises `HTTPException` inside a
  `BaseHTTPMiddleware.dispatch`, where it is not converted to a 401, so an unsigned request
  errors instead of being rejected.
- `.github/workflows/ci.yml` uses `working-directory: api` / `web`; both live under `extension/`.
- The FastAPI service serves the AutoGluon / PatchTST / anomaly-ensemble predictors, not
  Chronos-2, and no `predictor.pkl` is committed.
- "NPTS is the deployed incumbent" rests on the comment in `score_benchmark.py` that the shipping
  AutoGluon WeightedEnsemble puts weight 1.0 on NPTS. The repository has no deployment record
  beyond that; "NPTS baseline" is the defensible wording.
