# Two-window comparison — Mar–Apr 2026 vs Sep–Oct 2026

Ranks and paired tests are compared across windows; raw MASE levels are not, because each window's MASE is scaled by its own pre-origin history.

The Mar–Apr window is the published study re-analysed with TimesFM 3.0 added (`window1_timesfm3/`): every published row is reproduced unchanged; only ranks below TimesFM 3.0 shift, because there are now 14 models.

Spearman rank correlation of the mean-rank leaderboards: **0.851** (p = 0.000116, 14 models). Per horizon: 6 h 0.820, 12 h 0.859, 1 d 0.842, 2 d 0.846, 3 d 0.908, 7 d 0.903.

| Model | mean rank, Mar–Apr 2026 | mean rank, Sep–Oct 2026 | MASE 1 d, Mar–Apr 2026 | MASE 1 d, Sep–Oct 2026 |
|---|---|---|---|---|
| Timestral-209M-ZS | 1.00 | 1.00 | 0.6533 | 0.6801 |
| Chronos2-ZS | 4.17 | 2.67 | 0.6552 | 0.6939 |
| Chronos2-COV-LEAN † | 4.50 | 3.00 | 0.6562 | 0.6963 |
| Chronos2-COV † | 3.33 | 4.17 | 0.6558 | 0.6981 |
| Chronos2-COV-XL † | 2.00 | 4.50 | 0.6550 | 0.6992 |
| TimesFM3-ZS ‡ | 7.33 | 5.67 | 0.7343 | 0.7069 |
| TimesFM2.5-ZS | 6.17 | 7.00 | 0.6898 | 0.7268 |
| NPTS | 7.50 | 8.33 | 0.7098 | 0.8306 |
| DynamicOptimizedTheta | 12.17 | 9.67 | 0.9575 | 0.9603 |
| Theta | 11.83 | 10.00 | 0.9561 | 0.9603 |
| ETS | 12.00 | 11.00 | 0.9611 | 0.9753 |
| PatchTST † | 10.00 | 12.00 | 0.8533 | 0.9990 |
| SeasonalNaive | 14.00 | 12.83 | 1.0320 | 0.9954 |
| PatchTST-Tuned † | 9.00 | 13.17 | 0.8019 | 1.0032 |

† qualified in the Sep–Oct 2026: Chronos2-COV, Chronos2-COV-LEAN, Chronos2-COV-XL, PatchTST, PatchTST-Tuned.

‡ benchmark only, not deployable: TimesFM3-ZS.

## Head-to-head on MASE, both windows

`improvement` is the base's MASE relative to the opponent's, from the base's side: positive = the base has the lower error.

| Base | Opponent | Horizon | Mar–Apr 2026: improvement · tests | Sep–Oct 2026: improvement · tests |
|---|---|---|---|---|
| Chronos2-ZS | Timestral-209M-ZS | 6 h | -0.74 % · neither test | -2.46 % · neither test |
| Chronos2-ZS | Timestral-209M-ZS | 12 h | -0.76 % · neither test | -2.27 % · bootstrap only, for Timestral-209M-ZS |
| Chronos2-ZS | Timestral-209M-ZS | 1 d | -0.27 % · neither test | -2.06 % · both tests, for Timestral-209M-ZS |
| Chronos2-ZS | Timestral-209M-ZS | 2 d | -0.42 % · neither test | -1.90 % · both tests, for Timestral-209M-ZS |
| Chronos2-ZS | Timestral-209M-ZS | 3 d | -0.57 % · bootstrap only, for Timestral-209M-ZS | -1.97 % · bootstrap only, for Timestral-209M-ZS |
| Chronos2-ZS | Timestral-209M-ZS | 7 d | -0.93 % · bootstrap only, for Timestral-209M-ZS | -1.99 % · both tests, for Timestral-209M-ZS |
| Chronos2-ZS | TimesFM2.5-ZS | 6 h | +10.86 % · bootstrap only, for Chronos2-ZS | +7.49 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | TimesFM2.5-ZS | 12 h | +7.51 % · bootstrap only, for Chronos2-ZS | +6.07 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | TimesFM2.5-ZS | 1 d | +5.23 % · bootstrap only, for Chronos2-ZS | +5.24 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | TimesFM2.5-ZS | 2 d | +3.61 % · bootstrap only, for Chronos2-ZS | +4.34 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | TimesFM2.5-ZS | 3 d | +3.14 % · bootstrap only, for Chronos2-ZS | +3.92 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | TimesFM2.5-ZS | 7 d | +2.65 % · both tests, for Chronos2-ZS | +2.73 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | NPTS | 6 h | +12.50 % · both tests, for Chronos2-ZS | +27.05 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | NPTS | 12 h | +9.11 % · both tests, for Chronos2-ZS | +21.21 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | NPTS | 1 d | +7.68 % · both tests, for Chronos2-ZS | +16.63 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | NPTS | 2 d | +6.42 % · both tests, for Chronos2-ZS | +14.61 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | NPTS | 3 d | +5.99 % · both tests, for Chronos2-ZS | +13.77 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | NPTS | 7 d | +5.73 % · both tests, for Chronos2-ZS | +11.90 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | Chronos2-COV-XL | 6 h | -0.24 % · neither test | +2.10 % · neither test |
| Chronos2-ZS | Chronos2-COV-XL | 12 h | -0.19 % · neither test | +1.49 % · neither test |
| Chronos2-ZS | Chronos2-COV-XL | 1 d | +0.03 % · neither test | +0.77 % · neither test |
| Chronos2-ZS | Chronos2-COV-XL | 2 d | -0.32 % · neither test | +0.34 % · neither test |
| Chronos2-ZS | Chronos2-COV-XL | 3 d | -0.32 % · neither test | +0.61 % · neither test |
| Chronos2-ZS | Chronos2-COV-XL | 7 d | -0.36 % · neither test | +0.04 % · neither test |
| Chronos2-ZS | PatchTST-Tuned | 6 h | +16.51 % · both tests, for Chronos2-ZS | +46.97 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | PatchTST-Tuned | 12 h | +17.40 % · both tests, for Chronos2-ZS | +42.60 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | PatchTST-Tuned | 1 d | +18.58 % · both tests, for Chronos2-ZS | +31.12 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | PatchTST-Tuned | 2 d | +17.25 % · both tests, for Chronos2-ZS | +38.71 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | PatchTST-Tuned | 3 d | +17.69 % · both tests, for Chronos2-ZS | +28.08 % · both tests, for Chronos2-ZS |
| Chronos2-ZS | PatchTST-Tuned | 7 d | +18.39 % · both tests, for Chronos2-ZS | +34.81 % · both tests, for Chronos2-ZS |
| Timestral-209M-ZS | TimesFM2.5-ZS | 6 h | +11.51 % · bootstrap only, for Timestral-209M-ZS | +9.71 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | TimesFM2.5-ZS | 12 h | +8.21 % · bootstrap only, for Timestral-209M-ZS | +8.15 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | TimesFM2.5-ZS | 1 d | +5.48 % · bootstrap only, for Timestral-209M-ZS | +7.15 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | TimesFM2.5-ZS | 2 d | +4.02 % · bootstrap only, for Timestral-209M-ZS | +6.12 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | TimesFM2.5-ZS | 3 d | +3.69 % · both tests, for Timestral-209M-ZS | +5.77 % · bootstrap only, for Timestral-209M-ZS |
| Timestral-209M-ZS | TimesFM2.5-ZS | 7 d | +3.55 % · both tests, for Timestral-209M-ZS | +4.63 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | NPTS | 6 h | +13.14 % · both tests, for Timestral-209M-ZS | +28.80 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | NPTS | 12 h | +9.80 % · both tests, for Timestral-209M-ZS | +22.96 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | NPTS | 1 d | +7.93 % · both tests, for Timestral-209M-ZS | +18.31 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | NPTS | 2 d | +6.81 % · both tests, for Timestral-209M-ZS | +16.20 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | NPTS | 3 d | +6.53 % · both tests, for Timestral-209M-ZS | +15.43 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | NPTS | 7 d | +6.60 % · both tests, for Timestral-209M-ZS | +13.62 % · both tests, for Timestral-209M-ZS |
| Chronos2-ZS | TimesFM3-ZS | 6 h | +17.67 % · neither test | +2.08 % · bootstrap only, for Chronos2-ZS |
| Chronos2-ZS | TimesFM3-ZS | 12 h | +15.81 % · neither test | +4.71 % · bootstrap only, for Chronos2-ZS |
| Chronos2-ZS | TimesFM3-ZS | 1 d | +11.41 % · neither test | +3.98 % · neither test |
| Chronos2-ZS | TimesFM3-ZS | 2 d | +6.75 % · neither test | +2.96 % · neither test |
| Chronos2-ZS | TimesFM3-ZS | 3 d | +4.62 % · neither test | +2.42 % · neither test |
| Chronos2-ZS | TimesFM3-ZS | 7 d | +2.03 % · neither test | +0.17 % · neither test |
| Timestral-209M-ZS | TimesFM3-ZS | 6 h | +18.27 % · neither test | +4.44 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | TimesFM3-ZS | 12 h | +16.45 % · neither test | +6.82 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | TimesFM3-ZS | 1 d | +11.64 % · bootstrap only, for Timestral-209M-ZS | +5.92 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | TimesFM3-ZS | 2 d | +7.14 % · bootstrap only, for Timestral-209M-ZS | +4.77 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | TimesFM3-ZS | 3 d | +5.16 % · bootstrap only, for Timestral-209M-ZS | +4.30 % · both tests, for Timestral-209M-ZS |
| Timestral-209M-ZS | TimesFM3-ZS | 7 d | +2.93 % · bootstrap only, for Timestral-209M-ZS | +2.12 % · bootstrap only, for Timestral-209M-ZS |
| TimesFM3-ZS | TimesFM2.5-ZS | 6 h | -8.27 % · neither test | +5.52 % · both tests, for TimesFM3-ZS |
| TimesFM3-ZS | TimesFM2.5-ZS | 12 h | -9.87 % · neither test | +1.42 % · neither test |
| TimesFM3-ZS | TimesFM2.5-ZS | 1 d | -6.97 % · neither test | +1.31 % · neither test |
| TimesFM3-ZS | TimesFM2.5-ZS | 2 d | -3.37 % · neither test | +1.42 % · neither test |
| TimesFM3-ZS | TimesFM2.5-ZS | 3 d | -1.56 % · neither test | +1.53 % · neither test |
| TimesFM3-ZS | TimesFM2.5-ZS | 7 d | +0.63 % · neither test | +2.57 % · both tests, for TimesFM3-ZS |
