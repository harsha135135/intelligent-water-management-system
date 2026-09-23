# A plain-PyTorch MLP baseline

**Question.** Does a small neural network, trained here on each tank's own history, improve on
repeating yesterday's demand — and where does it land against the models already in the study?

**Answer, in one line.** It beats the naive and linear baselines, ranks 8th of 13 at 24 hours,
and loses clearly to Chronos-2 zero-shot, NPTS and the tuned PatchTST across all tanks. On
healthy sensors alone it edges out the tuned PatchTST at 24 hours (+6.7 %, p < 0.05), but most
of its advantage over the linear model disappears.

Code: `src/models/mlp_forecaster.py` (training), `src/models/mlp_comparison.py` (scoring).
Tests: `tests/test_mlp.py` (8, no pytest needed). Outputs: `results/chronos2/mlp/`.

```bash
python -m tests.test_mlp                 # 8/8 must pass
python -m src.models.mlp_forecaster      # ~50 s on CPU, writes predictions + manifest
python -m src.models.mlp_comparison      # scores all 13 models, writes tables + figures
```

---

## Setup

| | |
|---|---|
| Input | previous 168 hours of outflow (one week) |
| Output | next 24 hourly values, in one forward pass |
| MLP | 168 → 64 → 32 → 24, ReLU between hidden layers |
| Linear-AR | 168 → 24, one linear layer — same inputs, loss and loop |
| Loss / optimiser | MSE on per-tank z-scored demand, Adam (lr 1e-3, batch 128) |
| Models | one per tank per architecture: 48 in total |
| Selection | early stopping on validation loss (patience 25, max 300 epochs) |
| Scored at | 6, 12, 24 h — the horizons a 24-step output covers |

### Protocol, and the leakage each rule prevents

1. **Fitted once on data at or before the first backtest origin** (2026-03-24 22:00), then
   rolled over all 24 origins without refitting — the same protocol as `patchtst_benchmark.py`.
2. **Validation is the last 28 days before that cutoff, split by target date.** Training targets
   all end before validation begins. A random split of overlapping windows would put nearly
   identical samples on both sides.
3. **Normalisation mean and std come from the training period only.**
4. **Training windows with any missing hour are skipped.** At test time an origin cannot be
   skipped (every model must score the same rows), so gaps in the input are forward-filled and
   counted: 199 of 576 test inputs had a gap, almost all from one campus-wide telemetry outage
   (84 hours per tank).
5. **Row parity is enforced**: the MLP's scored rows must equal the benchmark's exact
   (tank, origin, horizon, timestamp) keys, or `mlp_comparison` stops.
6. **Deterministic**: seeds fixed before model construction (a bug the leakage test caught — see
   below); two full runs give bit-identical predictions.

---

## Results

### Leaderboard — macro MASE, identical rows (lower is better)

| model | 6 h | 12 h | 24 h |
|---|---|---|---|
| Chronos2-ZS (zero-shot) | 0.602 | 0.629 | 0.655 |
| NPTS | 0.687 | 0.692 | 0.710 |
| PatchTST-Tuned | 0.719 | 0.761 | 0.802 |
| PatchTST | 0.774 | 0.825 | 0.853 |
| **MLP** | **0.826** | **0.860** | **0.870** |
| **Linear-AR** | **0.891** | **0.903** | **0.930** |
| Theta | 0.897 | 0.945 | 0.956 |
| ETS | 0.910 | 0.948 | 0.961 |
| SeasonalNaive | 0.981 | 0.967 | 1.032 |

(The three Chronos-2 covariate variants sit within 0.002 of Chronos2-ZS; full table in
`results/chronos2/mlp/report.md`.)

### Paired significance, MLP vs each opponent (24 origins; bootstrap + Diebold–Mariano-HLN)

Positive = the MLP has lower MASE. ✓ = both tests p < 0.05.

| opponent | all tanks 6 / 12 / 24 h | healthy tanks only 6 / 12 / 24 h |
|---|---|---|
| Linear-AR | +7.3 ✓ / +4.8 ✓ / +6.6 ✓ | +3.3 ✓ / +1.7 · / +2.3 · |
| SeasonalNaive | +15.3 · / +10.8 ✓ / +15.8 ✓ | +18.1 ✓ / +16.5 ✓ / +17.6 ✓ |
| NPTS | −20.6 ✓ / −24.4 ✓ / −22.8 ✓ | −4.5 · / −9.4 ✓ / −10.1 ✓ |
| PatchTST-Tuned | −15.1 ✓ / −13.1 ✓ / −8.3 ✓ | +1.6 · / +3.2 · / **+6.7 ✓** |
| Chronos2-ZS | −37.8 ✓ / −36.9 ✓ / −33.0 ✓ | −24.8 ✓ / −23.6 ✓ / −21.4 ✓ |

### 24-hour MASE by sensor trust tier

| tier (tanks) | MLP | Linear-AR | SeasonalNaive | Chronos2-ZS |
|---|---|---|---|---|
| healthy (15) | 0.807 | 0.825 | 0.982 | 0.667 |
| degraded (6) | 0.696 | 0.721 | 0.756 | 0.479 |
| dead (3) | 1.534 | 1.874 | 1.835 | 0.948 |

Per tank at 24 h: the MLP beats SeasonalNaive on 19/24, Linear-AR on 14/24, Chronos-2 on 0/24.

---

## What the results say

1. **Learning from a week of history clearly beats repeating yesterday.** 15–18 % lower MASE
   than SeasonalNaive, significant at every horizon on healthy tanks.
2. **The hidden layers add little on healthy sensors.** Across all tanks the MLP beats Linear-AR
   significantly, but the per-tank table shows why: most of that gap comes from dead and
   degraded tanks (e.g. INFORMATION_CENTRE 3.26 vs 4.23). On the 15 healthy tanks the edge is
   +1.7–3.3 % and significant only at 6 h. Most of what a one-week window can tell you here is
   linear.
3. **The MLP overfits within a few epochs.** Median best epoch is 5, with 11 of 24 tanks at epoch
   3 or earlier (`plots/M2_loss_curves`): training loss keeps falling while validation loss rises.
   Early stopping is doing real work. Weight decay, dropout or a cross-tank model are the obvious
   next experiments, and none was tried here, to keep this a baseline rather than a tuning study.
4. **Against PatchTST the tank mix decides the winner.** PatchTST-Tuned wins across all tanks,
   but on healthy tanks the MLP is significantly better at 24 h. The macro average is sensitive
   to a few broken sensors, which is itself worth stating whenever these models are compared.
5. **Chronos-2 zero-shot remains best everywhere**: 21–25 % lower MASE than the MLP even on
   healthy tanks, and better on all 24 tanks individually.

## Limitations

- One architecture and one set of hyperparameters; no search. The MLP is a baseline, not a
  tuned competitor.
- Point forecasts only: no quantiles, so no coverage or calibration comparison.
- Per-tank models do not share information across tanks; PatchTST and Chronos-2 do.
- Horizons beyond 24 h are not scored.
- 24 origins over about three weeks — the same test period as the rest of the study.

## The bug the tests caught

`test_training_never_sees_data_after_the_cutoff` corrupts every value after the cutoff and
asserts that the trained weights do not change. It failed on the first run. The cause was not
leakage: `torch.manual_seed` was called inside the training loop, *after* `nn.Linear` had
already drawn its initial weights, so two identical runs started from different points. Seeding
before construction fixed it, and the test now also guards reproducibility.

## What to be able to explain

- The tensor shapes: a batch is `(128, 168)` in and `(128, 24)` out; the first layer holds a
  `(64, 168)` weight matrix plus 64 biases.
- One training step: forward pass → MSE → `zero_grad()` (gradients accumulate by default) →
  `backward()` (fills each parameter's `.grad`) → `optimizer.step()`.
- Why `model.eval()` and `torch.no_grad()` are used for validation and forecasting.
- Why the split is chronological by target date, and what would leak with a random split.
- How the loss curves show overfitting, and what early stopping kept.
- Why the MLP-vs-linear conclusion changes when dead sensors are excluded.
