# Same-CPU timing reference

A **timing measurement only**. The predictions written here are not scored and not part of the
benchmark; the published Chronos-2 predictions are the MPS run in `../predictions_Chronos2-ZS.parquet`.

## Why it exists

The official TimesFM 2.5 PyTorch code runs on CUDA or CPU only, so TimesFM's benchmark run was on
CPU, while Chronos-2's published 88.9 s (`../run_manifest.json`) was on Apple MPS. Those two wall
clocks are not a speed comparison. This directory holds Chronos-2 zero-shot re-run on the **same
CPU**, over the same 24 origins × 6 horizons × 24 tanks (144 forecast calls), so the foundation
models can be compared on the same hardware. (Timestral, added later, also ran on CPU.)

```bash
python -m src.models.chronos2_forecasting --variants Chronos2-ZS --device cpu \
    --out-dir results/chronos2/cpu_timing
```

## Measured

| | Value | Source |
|---|---|---|
| Chronos-2 zero-shot, CPU | 67.9 s for the 144-call backtest | `run_manifest.json` (`wall_clock_s`) |
| TimesFM 2.5 zero-shot, CPU | 158.4 s for the same 144 calls | `../timesfm_manifest.json` (`timing.wall_clock_s`) |
| Timestral 209M zero-shot, CPU | 144.7 s for the same 144 calls | `../timestral_209m_manifest.json` (`timing.wall_clock_s`) |
| Machine | Apple M4 Pro, 12 logical CPUs, 24 GB, macOS 26.5.2, torch 2.9.1, 8 torch threads | `../timesfm_manifest.json` (`hardware`) |

The Chronos-2 and TimesFM figures are single runs on the same machine in the same session.
Timestral's is a single run on the same machine, OS and torch thread count in a later session
(2026-10-21; `timing.measured_at` in its manifest). All exclude model load, and
`unified_analysis.cost()` carries them into `unified/cost.csv` as `wall_clock_s_same_cpu`.

**Sanity check.** The CPU forecasts agree with the published MPS forecasts on all 190,080 rows to
a maximum absolute difference of 3.1e-6 KL/h (mean 6.7e-8): the same model doing the same work,
so the difference in wall clock is the device, not the computation.
