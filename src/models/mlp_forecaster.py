"""A plain-PyTorch MLP and a linear autoregressive model on the shared backtest grid.

The question this answers: **does a small neural network trained here improve on repeating
yesterday's demand, and how far is it from the models already benchmarked?** Every other trained
model in the study (PatchTST, the AutoGluon baselines) is fitted through a framework. This one is
written directly against ``torch.nn`` so every step of training is visible:

    Dataset -> DataLoader -> nn.Module.forward -> loss -> loss.backward() -> optimizer.step()

Design, and the reason for each choice:

* **One model per tank**, 168 hours in -> 24 hours out. Tanks differ in scale by three orders of
  magnitude, so a per-tank model with per-tank normalisation is the simplest thing that is fair.
* **Same protocol as ``patchtst_benchmark.py``**: fitted once on data at or before the first
  backtest origin, then rolled forward over all 24 origins without refitting. No test row is
  ever seen during training or model selection.
* **Chronological validation, split by target date.** Training windows overlap, so a random split
  would put nearly identical windows on both sides and make validation loss meaningless. The last
  ``VAL_DAYS`` before the fit cutoff are validation; training targets all end before it starts.
* **Normalisation statistics come from the training period only.**
* **Windows with any missing hour are skipped** for training. A filled-in value is not an
  observation, and training on it would teach the network the imputation rule.
* **Early stopping on validation loss** picks the epoch; the test grid is never consulted.
* **The linear model uses the exact same inputs, loss and training loop.** If the MLP only
  matches it, the hidden layers bought nothing.

Only horizons 6, 12 and 24 are scored: the network emits 24 steps, and scoring it at 48-168 h would
need a longer output head or recursive unrolling, which is a different experiment.

    python -m src.models.mlp_forecaster            # ~1-2 min on CPU, all 24 tanks
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .backtest import BacktestSpec, actuals_after, make_spec

logger = logging.getLogger(__name__)

LOOKBACK = 168          # one week of hourly history per input window
OUT_STEPS = 24          # one day of hourly forecasts per output
SCORED_HORIZONS = [6, 12, 24]
VAL_DAYS = 28
SEED = 20260923

OUT_DIR = Path("results/chronos2/mlp")


# ────────────────────────────────────────────────────────────── windows

def window_origins(y: np.ndarray, lookback: int = LOOKBACK, out: int = OUT_STEPS) -> np.ndarray:
    """Indices ``t`` where ``y[t-lookback+1 : t+1]`` (input) and ``y[t+1 : t+1+out]`` (target)
    are all observed. ``t`` is the forecast origin: the last hour the model is allowed to see."""
    n = len(y)
    if n < lookback + out:
        return np.array([], dtype=int)
    finite = np.isfinite(y).astype(np.int64)
    # Prefix sums turn "is every value in this span finite?" into one subtraction per window.
    csum = np.concatenate([[0], np.cumsum(finite)])
    t = np.arange(lookback - 1, n - out)
    span = lookback + out
    ok = (csum[t + out + 1] - csum[t - lookback + 1]) == span
    return t[ok]


def to_arrays(y: np.ndarray, origins: np.ndarray, lookback: int = LOOKBACK,
              out: int = OUT_STEPS) -> tuple[np.ndarray, np.ndarray]:
    """Stack windows into ``X: (N, lookback)`` and ``Y: (N, out)``."""
    if len(origins) == 0:
        return np.empty((0, lookback), np.float32), np.empty((0, out), np.float32)
    x_idx = origins[:, None] + np.arange(-lookback + 1, 1)[None, :]
    y_idx = origins[:, None] + np.arange(1, out + 1)[None, :]
    return y[x_idx].astype(np.float32), y[y_idx].astype(np.float32)


@dataclass
class Split:
    train: np.ndarray       # window origins (integer positions into the series)
    val: np.ndarray
    val_start: pd.Timestamp
    fit_cutoff: pd.Timestamp


def chronological_split(timestamps: pd.DatetimeIndex, origins: np.ndarray,
                        fit_cutoff: pd.Timestamp, val_days: int = VAL_DAYS,
                        out: int = OUT_STEPS) -> Split:
    """Train / validation split by *target* date, both strictly before ``fit_cutoff``.

    * validation: origin at or after ``val_start``, last target at or before ``fit_cutoff``
    * training:   last target at or before ``val_start``

    So no training target falls inside the validation period, and nothing at all falls after the
    first backtest origin.
    """
    val_start = fit_cutoff - pd.Timedelta(days=val_days)
    origin_ts = timestamps[origins]
    target_end = timestamps[origins + out]
    val = origins[(origin_ts >= val_start) & (target_end <= fit_cutoff)]
    train = origins[target_end <= val_start]
    return Split(train=train, val=val, val_start=val_start, fit_cutoff=fit_cutoff)


# ────────────────────────────────────────────────────────────── models

class MLP(nn.Module):
    """168 -> 64 -> 32 -> 24, ReLU between hidden layers, linear output (demand is unbounded
    above and negative values are clipped after de-normalisation, not by the network)."""

    def __init__(self, lookback: int = LOOKBACK, out: int = OUT_STEPS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lookback, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (batch, 168) -> (batch, 24)
        return self.net(x)


class LinearAR(nn.Module):
    """The same inputs mapped straight to the outputs: a direct linear autoregression."""

    def __init__(self, lookback: int = LOOKBACK, out: int = OUT_STEPS) -> None:
        super().__init__()
        self.net = nn.Linear(lookback, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


MODELS = {"MLP": MLP, "Linear-AR": LinearAR}


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 300
    patience: int = 25
    weight_decay: float = 0.0


@dataclass
class TrainResult:
    model: nn.Module
    best_epoch: int
    best_val_loss: float
    history: list[dict] = field(default_factory=list)


def train_model(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray,
                x_val: np.ndarray, y_val: np.ndarray, cfg: TrainConfig,
                seed: int = SEED) -> TrainResult:
    """The whole training loop, written out.

    Each step: forward pass -> MSE loss -> ``zero_grad`` (gradients accumulate by default, so the
    previous batch's must be cleared) -> ``backward`` (fills ``.grad`` on every parameter) ->
    ``step`` (Adam moves each parameter against its gradient). After every epoch the model is
    switched to ``eval()`` and scored on validation under ``no_grad``; the weights from the best
    validation epoch are kept, and training stops once ``patience`` epochs pass without a new best.
    """
    torch.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=cfg.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    xv, yv = torch.from_numpy(x_val), torch.from_numpy(y_val)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_state, best_loss, best_epoch, stale = None, float("inf"), -1, 0
    history: list[dict] = []
    for epoch in range(cfg.max_epochs):
        model.train()
        total, count = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            count += len(xb)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        history.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss})

        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model=model, best_epoch=best_epoch, best_val_loss=best_loss,
                       history=history)


# ────────────────────────────────────────────────────────────── inference

def input_window(y: np.ndarray, t: int, fill_value: float,
                 lookback: int = LOOKBACK) -> tuple[np.ndarray, int]:
    """The last ``lookback`` hours up to and including position ``t``, for a test origin.

    Unlike training, a test origin cannot be skipped — every model must score the same rows — so
    gaps are filled: forward-fill within the window, then ``fill_value`` (the training mean) for
    anything still missing. Returns the window and how many hours were filled, which is reported.
    """
    w = pd.Series(y[t - lookback + 1: t + 1], dtype=float)
    n_missing = int(w.isna().sum())
    w = w.ffill().fillna(fill_value)
    return w.to_numpy(dtype=np.float32), n_missing


@dataclass
class TankFit:
    item_id: str
    mean: float
    std: float
    n_train: int
    n_val: int
    fits: dict[str, TrainResult]


def fit_tank(item_id: str, series: pd.Series, fit_cutoff: pd.Timestamp,
             cfg: TrainConfig) -> TankFit:
    """Build windows, normalise from the training period, train both models."""
    ts = pd.DatetimeIndex(series.index)
    y = series.to_numpy(dtype=float)

    origins = window_origins(y)
    split = chronological_split(ts, origins, fit_cutoff)

    train_period = y[ts <= split.val_start]
    mean = float(np.nanmean(train_period))
    std = float(np.nanstd(train_period))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0          # a constant tank: centring alone, rather than dividing by ~0

    z = (y - mean) / std
    x_tr, y_tr = to_arrays(z, split.train)
    x_va, y_va = to_arrays(z, split.val)
    if len(x_tr) == 0 or len(x_va) == 0:
        raise ValueError(f"{item_id}: {len(x_tr)} train / {len(x_va)} val windows — cannot fit")

    fits = {}
    for name, cls in MODELS.items():
        # Seed *before* construction: nn.Linear draws its initial weights when it is created, so
        # seeding only inside train_model left the starting point — and the result — unrepeatable.
        torch.manual_seed(SEED)
        fits[name] = train_model(cls(), x_tr, y_tr, x_va, y_va, cfg)
    return TankFit(item_id=item_id, mean=mean, std=std, n_train=len(x_tr), n_val=len(x_va),
                   fits=fits)


def forecast_grid(panel: pd.DataFrame, spec: BacktestSpec,
                  fits: dict[str, TankFit]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roll every fitted model over every origin; emit rows in the benchmark's schema."""
    target = spec.target_col
    series = {i: g.set_index("timestamp")[target].sort_index()
              for i, g in panel.groupby("item_id", sort=False)}

    raw: dict[tuple[str, str, pd.Timestamp], np.ndarray] = {}
    filled = []
    for item_id, fit in fits.items():
        s = series[item_id]
        y = s.to_numpy(dtype=float)
        pos = {t: k for k, t in enumerate(s.index)}
        for origin in spec.origins:
            t = pos[origin]
            window, n_missing = input_window(y, t, fill_value=fit.mean)
            filled.append({"item_id": item_id, "origin": origin, "hours_filled": n_missing})
            x = torch.from_numpy(((window - fit.mean) / fit.std)[None, :].astype(np.float32))
            for name, res in fit.fits.items():
                with torch.no_grad():
                    z = res.model(x).numpy()[0]
                raw[(name, item_id, origin)] = np.clip(z * fit.std + fit.mean, 0.0, None)

    blocks = []
    for origin in spec.origins:
        for horizon in SCORED_HORIZONS:
            fut = actuals_after(panel, origin, horizon)
            for name in MODELS:
                block = fut[["item_id", "timestamp", "step"]].copy()
                block["pred"] = [raw[(name, i, origin)][s - 1]
                                 for i, s in zip(fut["item_id"], fut["step"], strict=True)]
                block["actual"] = fut[target].to_numpy()
                block["model"] = name
                block["origin"] = origin
                block["horizon"] = horizon
                blocks.append(block)
    return pd.concat(blocks, ignore_index=True), pd.DataFrame(filled)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    ap.add_argument("--patience", type=int, default=TrainConfig.patience)
    ap.add_argument("--max-tanks", type=int, default=None)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from ..data.curate import load_curated_hourly

    torch.set_num_threads(1)          # tiny models: threading adds overhead and nondeterminism
    panel = load_curated_hourly(with_features=False)
    spec = make_spec(panel)
    fit_cutoff = spec.origins[0]
    cfg = TrainConfig(max_epochs=args.max_epochs, patience=args.patience)
    tanks = sorted(panel["item_id"].unique())[: args.max_tanks]
    panel = panel[panel["item_id"].isin(tanks)]
    logger.info("Backtest: %s | fit cutoff %s | %d tanks", spec.describe(), fit_cutoff, len(tanks))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    fits: dict[str, TankFit] = {}
    histories = []
    for item_id in tanks:
        s = (panel[panel["item_id"] == item_id].set_index("timestamp")[spec.target_col]
             .sort_index())
        fit = fit_tank(item_id, s, fit_cutoff, cfg)
        fits[item_id] = fit
        for name, res in fit.fits.items():
            for h in res.history:
                histories.append({"item_id": item_id, "model": name, **h})
        logger.info("%-26s train=%5d val=%4d | %s", item_id, fit.n_train, fit.n_val,
                    "  ".join(f"{n}: epoch {r.best_epoch:3d} val {r.best_val_loss:.4f}"
                              for n, r in fit.fits.items()))
    fit_s = time.time() - t0

    preds, filled = forecast_grid(panel, spec, fits)
    leak = int((preds["timestamp"] <= preds["origin"]).sum())
    if leak:
        raise AssertionError(f"{leak} prediction rows at or before their origin")
    for name, g in preds.groupby("model"):
        g.to_parquet(out_dir / f"predictions_{name}.parquet", index=False)
    pd.DataFrame(histories).to_csv(out_dir / "training_history.csv", index=False)
    filled.to_csv(out_dir / "test_inputs_filled.csv", index=False)

    manifest = {
        "protocol": "per-tank models, fitted once on data <= first origin, rolled over 24 origins",
        "lookback_h": LOOKBACK, "out_steps": OUT_STEPS, "scored_horizons": SCORED_HORIZONS,
        "fit_cutoff": str(fit_cutoff), "val_days": VAL_DAYS,
        "val_start": str(fit_cutoff - pd.Timedelta(days=VAL_DAYS)),
        "models": {"MLP": "168-64-32-24, ReLU", "Linear-AR": "168-24 linear"},
        "train": {"loss": "MSE on per-tank z-scored demand", "optimizer": "Adam",
                  **cfg.__dict__, "seed": SEED, "device": "cpu",
                  "torch": torch.__version__},
        "wall_clock_fit_s": round(fit_s, 1),
        "tanks": {
            i: {"mean_kl_h": f.mean, "std_kl_h": f.std, "n_train_windows": f.n_train,
                "n_val_windows": f.n_val,
                **{f"{n}_best_epoch": r.best_epoch for n, r in f.fits.items()},
                **{f"{n}_best_val_mse": round(r.best_val_loss, 5) for n, r in f.fits.items()}}
            for i, f in fits.items()},
        "test_origins_with_filled_input": int((filled["hours_filled"] > 0).sum()),
        "test_input_hours_filled": int(filled["hours_filled"].sum()),
    }
    (out_dir / "mlp_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Fitted %d tanks x %d models in %.1fs -> %s", len(fits), len(MODELS), fit_s, out_dir)


if __name__ == "__main__":
    main()
