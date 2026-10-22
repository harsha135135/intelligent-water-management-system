# Reports

Two generated deliverables and one hand-written briefing. Both generated artefacts read the same
CSVs under `results/chronos2/unified/`, so **the page and the deck cannot disagree with each other
or with the data**.

| File | Committed? | What it is |
|---|---|---|
| `build_results_page.py` | ✅ yes | Generator for the results page. |
| `phase3_results_page.html` | ❌ regenerated | **The results page.** 13 sections, 17 figures, all 13 models including TimesFM 2.5 and Timestral 209M — the evidence itself. ~4.6 MB because the figures are embedded as base64; they are already tracked as PNG and SVG under `results/chronos2/`. |
| `build_review_deck.py` | ✅ yes | Generator for the deck. |
| `PW26_PK_06_phase3_review.pptx` | ❌ regenerated | **The review deck.** 40 slides across the six review categories — System Testing, Verification and Validation, Chronos-2, Deployment, Final Experiment Results, Performance Analysis — plus a one-slide summary and a reproduction appendix. |
| `review_briefing.html` | ✅ yes | **The briefing.** What was done, what each result means, the vocabulary that gets asked about, likely questions with answers, and the gaps to volunteer. Read this to *defend* the work. |

## Regenerating

```bash
python reports/build_results_page.py     # ~10 s
python reports/build_review_deck.py      # ~15 s   (needs python-pptx and pillow)
```

Every number is read from `results/chronos2/unified/` at build time — nothing is hard-coded, so
neither output can drift from the data. If the CSVs or figures are missing, regenerate them first:

```bash
python -m src.models.unified_analysis       # ~45 s  -> 12 tables, all models
python -m src.models.unified_figures        # ~5 s   -> figures U1-U12
python -m src.models.calibrated_holdout      # ~3 min -> calibration + the 45-day panels
python -m src.models.calibrated_holdout --add timesfm   # TimesFM's 45-day panel (O4)
python -m src.models.calibrated_holdout --add timestral # Timestral's 45-day panel (O5)
```

`reports/build_timesfm_section.py` and `reports/build_timestral_section.py` are the third and
fourth generators: they write §17 and §18 of `docs/review_summary.md` — the TimesFM and Timestral
comparisons — from the same result files, each between two marker comments, so those sections
cannot drift from the data either. `reports/build_holdout45_page.py` builds the interactive 45-day
explorer, `holdout45_explorer.html`, which now carries Timestral as its fifth series.

**`holdout45_explorer.html` currently shows the refresh study's window, 7 Sep – 21 Oct 2026**, on
the re-downloaded Waltr data (`results/refresh_2026-10-22/`), with PatchTST marked † as
calendar-incomplete. It also carries the published study's March-fitted PatchTST as a dashed line
(`--with-old`), run unchanged on the same origins by `python -m src.models.old_vs_new`, so the
retrained and the original network can be read day by day. Rebuilding with no environment set
regenerates the published 9 Mar – 22 Apr page instead. To rebuild the current one:

```bash
export PESU_DATASET_DIR=dataset_2026-10-22 PESU_RESULTS_DIR=results/refresh_2026-10-22
python -m src.models.old_vs_new          # ~8 min: March-fitted models on the new dates
python reports/build_holdout45_page.py --with-old --flag patchtst --flag-note "..."
cp results/refresh_2026-10-22/reports/holdout45_explorer.html reports/
```

`results/refresh_2026-10-22/old_vs_new/OLD_VS_NEW.md` is the written comparison.

### TimesFM 3.0 — benchmark only

TimesFM 3.0's weights are licensed for internal benchmarking and research only, not for production
systems or end users. The refresh study scores it as a reference point, marked ‡. It is never a
deployment candidate, and `build_dock_bundle` refuses it. The published study, its docs, deck and
page do not include it. Its published-window run is analysed in a separate folder that links the
published predictions read-only. Re-running the unified analysis there reproduces every published
row unchanged, and TimesFM 3.0's rows are appended. To rebuild it all:

```bash
R=results/refresh_2026-10-22; W=$R/window1_timesfm3
# window 1 (published data, defaults) — written into the refresh folder, never results/chronos2
python -m src.models.timesfm3_forecasting --out-dir $W --cache-dir $W/_timesfm3_cache
for p in results/chronos2/predictions_*.parquet; do ln -sf "$PWD/$p" "$W/$(basename $p)"; done
for f in run_manifest.json timesfm_manifest.json timestral_209m_manifest.json patchtst_manifest.json \
         patchtst_tuned_manifest.json baselines_manifest.json; do cp results/chronos2/$f $W/; done
mkdir -p $W/cpu_timing && cp results/chronos2/cpu_timing/run_manifest.json $W/cpu_timing/
PESU_RESULTS_DIR=$W python -m src.models.unified_analysis

# window 2 (the refresh study)
export PESU_DATASET_DIR=dataset_2026-10-22 PESU_RESULTS_DIR=$R PESU_EDA_DIR=$R/eda
python -m src.models.timesfm3_forecasting              # ~2.5 min on CPU
python -m src.models.score_benchmark --strict && python -m src.models.unified_analysis
python -m src.models.unified_figures
python -m src.models.calibrated_holdout --add timesfm3
python -m src.models.window_comparison --a $W --b $R --labels "Mar–Apr 2026" "Sep–Oct 2026" \
  --flag Chronos2-COV Chronos2-COV-LEAN Chronos2-COV-XL PatchTST PatchTST-Tuned \
  --benchmark-only TimesFM3-ZS --note "..."
python reports/build_timesfm_section.py && python reports/build_timestral_section.py
python -m src.models.old_vs_new --report && python reports/build_final_analysis.py
```

## Hosted copies

Both HTML reports were also published as Claude artifacts during review preparation. Those links
are **private by default** and need explicit sharing — the committed files here are the durable
copies.

- Results page — `https://claude.ai/code/artifact/75ed30fc-f1ea-47b5-b64f-eb7dc2cbf2d8`
- Briefing — `https://claude.ai/code/artifact/5b26e938-bdec-4720-922e-b1a0a8150dfc`
