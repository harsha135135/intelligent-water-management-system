# Reports

Two generated deliverables and one hand-written briefing. Both generated artefacts read the same
CSVs under `results/chronos2/unified/`, so **the page and the deck cannot disagree with each other
or with the data**.

| File | Committed? | What it is |
|---|---|---|
| `build_results_page.py` | ✅ yes | Generator for the results page. |
| `phase3_results_page.html` | ❌ regenerated | **The results page.** 13 sections, 15 figures, all 11 models — the evidence itself. ~3.9 MB because the figures are embedded as base64; they are already tracked as PNG and SVG under `results/chronos2/`. |
| `build_review_deck.py` | ✅ yes | Generator for the deck. |
| `PW26_PK_06_phase3_review.pptx` | ❌ regenerated | **The review deck.** 38 slides across the six review categories — System Testing, Verification and Validation, Chronos-2, Deployment, Final Experiment Results, Performance Analysis — plus a one-slide summary and a reproduction appendix. |
| `review_briefing.html` | ✅ yes | **The briefing.** What was done, what each result means, the vocabulary that gets asked about, likely questions with answers, and the gaps to volunteer. Read this to *defend* the work. |

## Regenerating

```bash
python reports/build_results_page.py     # ~10 s
python reports/build_review_deck.py      # ~15 s   (needs python-pptx and pillow)
```

Every number is read from `results/chronos2/unified/` at build time — nothing is hard-coded, so
neither output can drift from the data. If the CSVs or figures are missing, regenerate them first:

```bash
python -m src.models.unified_analysis       # ~4 min -> 10 tables, all models
python -m src.models.unified_figures        # ~25 s  -> figures U1-U12
python -m src.models.calibrated_holdout      # ~3 min -> calibration + the 45-day panels
```

## Hosted copies

Both HTML reports were also published as Claude artifacts during review preparation. Those links
are **private by default** and need explicit sharing — the committed files here are the durable
copies.

- Results page — `https://claude.ai/code/artifact/75ed30fc-f1ea-47b5-b64f-eb7dc2cbf2d8`
- Briefing — `https://claude.ai/code/artifact/5b26e938-bdec-4720-922e-b1a0a8150dfc`
