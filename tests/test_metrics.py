"""Correctness tests for the scaled metrics.

The load-bearing test is `test_seasonal_naive_mase_is_about_one`: MASE is *defined* so that a
seasonal-naive forecast scores 1.0. If that identity fails, the scaling denominator is wrong and
every number in the benchmark table is wrong.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.metrics import (  # noqa: E402
    aggregate_metrics, attach_scales, per_tank_metrics, seasonal_scales,
)

M = 24
TARGET = "Outflow in KL"


def _panel(n_tanks=3, n_hours=24 * 60, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    ts = pd.date_range("2025-01-01", periods=n_hours, freq="h")
    for k in range(n_tanks):
        daily = 2.0 + 1.5 * np.sin(2 * np.pi * np.arange(24) / 24)
        base = np.tile(daily, n_hours // 24)[:n_hours] * (k + 1)
        y = np.clip(base + rng.normal(0, 0.3, n_hours), 0, None)
        frames.append(pd.DataFrame({"item_id": f"T{k}", "timestamp": ts, TARGET: y}))
    return pd.concat(frames, ignore_index=True)


def test_seasonal_scales_hand_worked():
    # y_t - y_{t-2} is +2 everywhere for m=2 on this ramp.
    hist = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    s_mae, s_mse = seasonal_scales(hist, m=2)
    assert np.isclose(s_mae, 2.0)
    assert np.isclose(s_mse, 4.0)


def test_seasonal_scales_ignores_nan():
    hist = np.array([1.0, 2, np.nan, 4, 5, 6])
    s_mae, _ = seasonal_scales(hist, m=2)
    assert np.isclose(s_mae, 2.0)  # NaN pair dropped, remaining diffs all +2


def test_constant_series_scale_is_zero():
    s_mae, s_mse = seasonal_scales(np.zeros(100), m=M)
    assert s_mae == 0.0 and s_mse == 0.0


def test_perfect_forecast_scores_zero():
    panel = _panel()
    origin = panel["timestamp"].iloc[-M - 1]
    fut = panel[panel["timestamp"] > origin].copy()
    preds = fut.rename(columns={TARGET: "actual"}).assign(
        model="oracle", origin=origin, horizon=M,
    )
    preds["step"] = preds.groupby("item_id").cumcount() + 1
    preds["pred"] = preds["actual"]
    scored = attach_scales(preds, panel, target_col=TARGET)
    agg = aggregate_metrics(per_tank_metrics(scored))
    assert np.allclose(agg[["macro_mae", "macro_rmse", "macro_mase", "macro_rmsse"]], 0.0)


def test_seasonal_naive_mase_is_about_one():
    """The defining property of MASE. A seasonal-naive forecast must score ~1.0."""
    panel = _panel(n_tanks=4, n_hours=24 * 120, seed=7)
    rows = []
    # Several origins, each forecasting 24h by repeating the previous day.
    for k in range(20):
        origin = panel["timestamp"].iloc[-(M * (k + 2)) - 1]
        for item_id, g in panel.groupby("item_id"):
            g = g.sort_values("timestamp")
            hist = g[g["timestamp"] <= origin]
            fut = g[g["timestamp"] > origin].head(M)
            rows.append(pd.DataFrame({
                "model": "SeasonalNaive-24", "item_id": item_id, "origin": origin,
                "horizon": M, "step": range(1, len(fut) + 1),
                "timestamp": fut["timestamp"].to_numpy(),
                "actual": fut[TARGET].to_numpy(),
                "pred": hist[TARGET].to_numpy()[-M:][: len(fut)],
            }))
    scored = attach_scales(pd.concat(rows, ignore_index=True), panel, target_col=TARGET)
    agg = aggregate_metrics(per_tank_metrics(scored))
    mase = float(agg["macro_mase"].iloc[0])
    rmsse = float(agg["macro_rmsse"].iloc[0])
    assert 0.85 < mase < 1.15, f"seasonal-naive MASE should be ~1.0, got {mase}"
    assert 0.85 < rmsse < 1.15, f"seasonal-naive RMSSE should be ~1.0, got {rmsse}"


def test_degenerate_series_excluded_from_scaled_metrics():
    panel = _panel(n_tanks=2, n_hours=24 * 40)
    dead = panel[panel["item_id"] == "T0"].copy()
    dead["item_id"] = "DEAD"
    dead[TARGET] = 0.0
    panel = pd.concat([panel, dead], ignore_index=True)

    origin = panel["timestamp"].max() - pd.Timedelta(hours=M)
    rows = []
    for item_id, g in panel.groupby("item_id"):
        fut = g[g["timestamp"] > origin].sort_values("timestamp")
        rows.append(pd.DataFrame({
            "model": "m", "item_id": item_id, "origin": origin, "horizon": M,
            "step": range(1, len(fut) + 1), "timestamp": fut["timestamp"].to_numpy(),
            "actual": fut[TARGET].to_numpy(), "pred": 0.0,
        }))
    scored = attach_scales(pd.concat(rows, ignore_index=True), panel, target_col=TARGET)
    per_tank = per_tank_metrics(scored)
    agg = aggregate_metrics(per_tank)

    assert not per_tank.loc[per_tank["item_id"] == "DEAD", "scaled_ok"].iloc[0]
    assert int(agg["n_tanks"].iloc[0]) == 3
    assert int(agg["n_tanks_scaled"].iloc[0]) == 2  # DEAD excluded, and visibly so
    assert np.isfinite(agg["macro_mase"].iloc[0])


def _run_all() -> int:
    """Run every test in this module without needing pytest installed.

    `python -m tests.test_metrics` is the command the README documents, so it has to work in a
    bare `venv/` where pytest is not present.
    """
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
