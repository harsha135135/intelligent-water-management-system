"""Correctness tests for the PyTorch MLP baseline.

The load-bearing ones are the leakage tests: a forecaster that sees a single test hour during
training or model selection produces a number that means nothing, and nothing downstream would
reveal it. Run with ``python -m tests.test_mlp`` — no pytest needed, same as ``test_metrics``.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.mlp_forecaster import (  # noqa: E402
    LOOKBACK, MLP, OUT_STEPS, LinearAR, TrainConfig, chronological_split, fit_tank,
    input_window, to_arrays, train_model, window_origins,
)


def _daily_series(n_days=90, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=24 * n_days, freq="h")
    daily = 2.0 + 1.5 * np.sin(2 * np.pi * np.arange(24) / 24)
    y = np.tile(daily, n_days) + rng.normal(0, 0.1, len(ts))
    return pd.Series(np.clip(y, 0, None), index=ts)


def test_windows_skip_any_missing_hour():
    y = np.arange(400, dtype=float)
    y[250] = np.nan
    origins = window_origins(y)
    x, t = to_arrays(y, origins)
    assert np.isfinite(x).all() and np.isfinite(t).all()
    # every window whose input or target span touches index 250 must be gone
    touching = [o for o in origins if o - LOOKBACK + 1 <= 250 <= o + OUT_STEPS]
    assert not touching


def test_window_contents_are_aligned():
    y = np.arange(300, dtype=float)
    origins = window_origins(y)
    x, t = to_arrays(y, origins[:1])
    o = origins[0]
    assert x[0, -1] == o, "last input value must be the origin hour itself"
    assert t[0, 0] == o + 1, "first target must be the hour after the origin"
    assert x.shape == (1, LOOKBACK) and t.shape == (1, OUT_STEPS)


def test_split_has_no_target_overlap_and_nothing_after_cutoff():
    s = _daily_series()
    ts = pd.DatetimeIndex(s.index)
    origins = window_origins(s.to_numpy())
    cutoff = ts[-24 * 10]
    split = chronological_split(ts, origins, cutoff)
    train_end = ts[split.train + OUT_STEPS]
    val_origin, val_end = ts[split.val], ts[split.val + OUT_STEPS]
    assert len(split.train) and len(split.val)
    assert train_end.max() <= split.val_start <= val_origin.min()
    assert val_end.max() <= cutoff


def test_training_never_sees_data_after_the_cutoff():
    """Corrupt everything after the cutoff; the fitted normalisation and weights must not move."""
    s = _daily_series()
    cutoff = s.index[-24 * 10]
    cfg = TrainConfig(max_epochs=5, patience=5)
    a = fit_tank("T", s, cutoff, cfg)
    corrupted = s.copy()
    corrupted[corrupted.index > cutoff] = 1e6
    b = fit_tank("T", corrupted, cutoff, cfg)
    assert a.mean == b.mean and a.std == b.std
    for name in a.fits:
        for pa, pb in zip(a.fits[name].model.parameters(), b.fits[name].model.parameters(),
                          strict=True):
            assert np.array_equal(pa.detach().numpy(), pb.detach().numpy())


def test_test_input_uses_only_the_past():
    y = np.arange(500, dtype=float)
    w, filled = input_window(y, 300, fill_value=0.0)
    assert w[-1] == 300 and w[0] == 300 - LOOKBACK + 1 and filled == 0


def test_test_input_gaps_are_filled_and_counted():
    y = np.arange(500, dtype=float)
    y[295:298] = np.nan
    w, filled = input_window(y, 300, fill_value=-1.0)
    assert filled == 3 and np.isfinite(w).all()
    assert w[-6] == 294 and w[-5] == 294     # forward-filled, not the fill value


def test_output_shapes():
    import torch
    x = torch.zeros(7, LOOKBACK)
    assert MLP()(x).shape == (7, OUT_STEPS)
    assert LinearAR()(x).shape == (7, OUT_STEPS)


def test_training_reduces_validation_loss_on_a_learnable_series():
    s = _daily_series()
    y = ((s - s.mean()) / s.std()).to_numpy()
    origins = window_origins(y)
    x, t = to_arrays(y, origins)
    cut = int(0.8 * len(x))
    res = train_model(MLP(), x[:cut], t[:cut], x[cut:], t[cut:],
                      TrainConfig(max_epochs=40, patience=40))
    first, best = res.history[0]["val_loss"], res.best_val_loss
    assert best < 0.5 * first, (first, best)
    assert best < 0.1, "a clean daily sine should be almost perfectly predictable"


def _run_all() -> int:
    import traceback

    tests = sorted(k for k, v in globals().items() if k.startswith("test_") and callable(v))
    failed = []
    for name in tests:
        try:
            globals()[name]()
            print(f"PASS  {name}")
        except Exception:
            failed.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
