# Reports

Two standalone HTML reports built for the Phase III review. Both open directly in a browser —
no server, no build step, no network.

| File | Committed? | What it is |
|---|---|---|
| `review_briefing.html` | ✅ yes | **The briefing.** What was done, what each result means, definitions for the terms that get asked about, eight likely review questions with answers, and the gaps to volunteer. Read this to *defend* the work. |
| `phase3_results_page.html` | ❌ regenerated | **The results page.** All 26 figures with every table. The evidence itself. ~5.5 MB because the figures are embedded as base64 — they are already tracked as PNG and SVG under `results/chronos2/`, so committing both would duplicate them. |
| `build_results_page.py` | ✅ yes | Generator for the results page. |

## Regenerating the results page

```bash
python reports/build_results_page.py     # ~5 s -> reports/phase3_results_page.html
```

It reads every number from the CSVs under `results/chronos2/` and embeds the figures from
`results/chronos2/review/plots/`, `.../phase3/plots/` and `.../calibrated/plots/`. Nothing is
hard-coded, so **the page cannot drift from the data** — if a metric changes, rerun the analysis
and then this, and the page follows.

If the figures or CSVs are missing, regenerate them first:

```bash
python -m src.models.phase3_analysis        # ~90 s  -> figures O-W + 9 tables
python -m src.models.calibrated_holdout     # ~3 min -> calibration + figures X-Z
python reports/build_results_page.py
```

## Hosted copies

Both were also published as Claude artifacts during the review preparation. Those links are
**private by default** and need explicit sharing to be opened by anyone else — the committed
files here are the durable copies.

- Results page — `https://claude.ai/code/artifact/75ed30fc-f1ea-47b5-b64f-eb7dc2cbf2d8`
- Briefing — `https://claude.ai/code/artifact/5b26e938-bdec-4720-922e-b1a0a8150dfc`
