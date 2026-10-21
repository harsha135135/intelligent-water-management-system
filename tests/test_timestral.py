"""Tests for the Timestral integration.

Every test uses a stub in place of the real model, so nothing is downloaded and the suite runs in
seconds. The stub has timestral's own ``forecast(context, prediction_length=..., ...)`` signature
and output - ``.quantiles`` of shape ``(B, 1, horizon, Q)`` and ``.quantile_levels`` - so the
adapter code under test is exactly the code the benchmark runs.

    python -m tests.test_timestral
"""
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import significance  # noqa: E402
from src.models import unified_analysis as ua  # noqa: E402
from src.models.metrics import PRED_SCHEMA  # noqa: E402
from src.models.backtest import make_spec  # noqa: E402
from src.models.score_benchmark import load_all_predictions  # noqa: E402
from src.models.timesfm_forecasting import (  # noqa: E402
    MODEL_NAME as TIMESFM, RunStats, forecast_one,
)
from src.models.timestral_forecasting import (  # noqa: E402
    FORECAST_FLAGS, MODEL_NAME, SIZES, TimestralForecaster, model_name, run,
)

TARGET = "Outflow in KL"
SENTINEL = 1.0e6
LEVELS = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
          0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)      # the released checkpoints'


@dataclass
class _Forecast:
    quantiles: object
    quantile_levels: tuple


class StubModel:
    """Stands in for ``timestral.TimestralForecaster``: repeats each context's last day as the
    median, puts quantile level ``p`` at ``median + 2 (p - 0.5)``, and records every input."""

    def __init__(self, levels=LEVELS, fail_on_value=None, as_torch=False, extra_steps=0):
        self.levels, self.fail_on_value = tuple(levels), fail_on_value
        self.as_torch, self.extra_steps = as_torch, extra_steps
        self.calls, self.kwargs = [], []

    def forecast(self, context, prediction_length, **kwargs):
        x = np.array(context, dtype=float, copy=True)
        self.calls.append(x)
        self.kwargs.append(kwargs)
        if self.fail_on_value is not None and (x == self.fail_on_value).any():
            raise ValueError("forecast produced NaN values; the input is degenerate")
        h = prediction_length + self.extra_steps
        day = np.nan_to_num(x[:, -24:])
        median = np.tile(day, (1, int(np.ceil(h / 24))))[:, :h]
        q = median[..., None] + 2 * (np.asarray(self.levels) - 0.5)
        q = q[:, None].astype(np.float32)                              # (B, 1, h, Q)
        return _Forecast(torch.from_numpy(q) if self.as_torch else q, self.levels)


def _fc(model=None, **extra):
    fp = {"model_id": "stub", "revision": "0", "timestral": "test", "torch": "test",
          "device": "cpu", "forecast_config": dict(FORECAST_FLAGS)}
    fp.update(extra)
    return TimestralForecaster(model=model or StubModel(), fingerprint=fp)


def _panel(n_tanks=3, n_hours=24 * 30, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n_hours, freq="h")
    frames = []
    for k in range(n_tanks):
        base = np.tile(2 + 1.5 * np.sin(2 * np.pi * np.arange(24) / 24), n_hours // 24)
        y = np.clip(base * (k + 1) + rng.normal(0, 0.3, n_hours), 0, None)
        frames.append(pd.DataFrame({"item_id": f"T{k}", "timestamp": ts, TARGET: y}))
    return pd.concat(frames, ignore_index=True)


# ────────────────────────────────────────────────────────────── adapter

def test_quantiles_are_picked_by_level_not_position():
    ctx = [np.arange(96, dtype=float), np.arange(96, dtype=float) * 2]
    p_a, q_a = _fc(StubModel()).forecast(24, ctx)
    shifted = (0.005, 0.01, 0.025, 0.05, 0.075) + LEVELS[2:-2] + (0.925, 0.95, 0.975, 0.99)
    p_b, q_b = _fc(StubModel(levels=shifted)).forecast(24, ctx)
    assert q_a.shape == (2, 24, 10) and p_a.shape == (2, 24)
    np.testing.assert_allclose(q_a, q_b)
    np.testing.assert_allclose(p_a, p_b)
    # slot 0 the median, slots 1-9 the 0.1 ... 0.9 quantiles
    np.testing.assert_allclose(q_a[..., 0], p_a)
    for slot, level in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], start=1):
        np.testing.assert_allclose(q_a[..., slot], p_a + 2 * (level - 0.5), atol=1e-5)


def test_torch_output_is_accepted():
    ctx = [np.arange(96, dtype=float)]
    p_np, q_np = _fc(StubModel()).forecast(12, ctx)
    p_t, q_t = _fc(StubModel(as_torch=True)).forecast(12, ctx)
    np.testing.assert_array_equal(q_np, q_t)
    assert isinstance(q_t, np.ndarray) and isinstance(p_t, np.ndarray)


def test_a_missing_quantile_level_raises():
    no_q20 = tuple(lv for lv in LEVELS if lv != 0.2)
    try:
        _fc(StubModel(levels=no_q20)).forecast(24, [np.arange(96, dtype=float)])
    except ValueError as exc:
        assert "0.2" in str(exc)
    else:
        raise AssertionError("a missing 0.2 quantile must raise")


def test_a_wrong_output_shape_raises():
    try:
        _fc(StubModel(extra_steps=1)).forecast(24, [np.arange(96, dtype=float)])
    except ValueError as exc:
        assert "expected" in str(exc)
    else:
        raise AssertionError("a forecast of the wrong length must raise")


def test_ragged_contexts_are_left_padded_with_nan():
    stub = StubModel()
    a, b = np.arange(96, dtype=float), np.arange(90, dtype=float) + 1
    _fc(stub).forecast(24, [a, b])
    sent = stub.calls[0]
    assert sent.shape == (2, 96)
    np.testing.assert_array_equal(sent[0], a)
    assert np.isnan(sent[1, :6]).all()
    np.testing.assert_array_equal(sent[1, 6:], b)


def test_package_defaults_are_passed_explicitly():
    stub = StubModel()
    _fc(stub).forecast(24, [np.arange(96, dtype=float)])
    assert stub.kwargs == [{"symmetric_averaging": True, "clip_nonnegative": True}]


# ────────────────────────────────────────────────────────────── through the shared plumbing

def test_point_is_the_median_and_quantiles_are_mapped():
    panel = _panel()
    origin = panel["timestamp"].iloc[-49]
    out = forecast_one(_fc(), panel, origin, 24, target_col=TARGET, context=96)
    # Stub: q10 = median - 0.8, q90 = median + 0.8, clipped at zero.
    assert np.allclose(out["0.5"], out["pred"])
    assert np.allclose(out["0.9"], np.clip(out["pred"] + 0.8, 0, None), atol=1e-5)
    assert np.allclose(out["0.1"], np.clip(out["pred"] - 0.8, 0, None), atol=1e-5)


def test_no_future_observation_reaches_the_model():
    panel = _panel()
    origin = panel["timestamp"].iloc[-60]
    poisoned = panel.copy()
    poisoned.loc[poisoned["timestamp"] > origin, TARGET] = SENTINEL
    stub, stats = StubModel(), RunStats()
    forecast_one(_fc(stub), poisoned, origin, 48, target_col=TARGET, context=96, stats=stats)
    sent = stub.calls[0]
    assert sent.shape == (3, 96) and np.nanmax(sent) < SENTINEL, "a post-origin value leaked"
    for k, item in enumerate(["T0", "T1", "T2"]):
        at_origin = panel[(panel.item_id == item) & (panel.timestamp == origin)][TARGET].item()
        assert np.isclose(sent[k, -1], at_origin), "context must end exactly at the origin"
    assert stats.windows_checked == 3 and stats.leakage_violations == 0


def test_missing_hours_are_passed_through_not_filled_by_the_adapter():
    panel = _panel()
    origin = panel["timestamp"].iloc[-49]
    gap = (panel["timestamp"] > origin - pd.Timedelta(hours=5)) & (panel["timestamp"] <= origin)
    panel.loc[gap & (panel.item_id == "T1"), TARGET] = np.nan
    stub = StubModel()
    forecast_one(_fc(stub), panel, origin, 24, target_col=TARGET, context=96)
    t1 = stub.calls[0][1]
    assert np.isnan(t1[-5:]).all() and not np.isnan(t1[:-5]).any()


def test_a_failing_series_is_recorded_and_left_unscored():
    panel = _panel()
    origin = panel["timestamp"].iloc[-49]
    panel.loc[(panel.item_id == "T2") & (panel.timestamp == origin - pd.Timedelta(hours=3)),
              TARGET] = 777.0                              # the stub fails on this value
    stats = RunStats()
    out = forecast_one(_fc(StubModel(fail_on_value=777.0)), panel, origin, 24,
                       target_col=TARGET, context=96, stats=stats)
    assert [f["item_id"] for f in stats.failures] == ["T2"]
    assert stats.failures[0]["error"] == "ValueError"
    assert out.loc[out.item_id == "T2", "pred"].isna().all()       # kept, not dropped
    assert out.loc[out.item_id != "T2", "pred"].notna().all()


def test_output_schema_matches_the_benchmark():
    panel = _panel()
    spec = make_spec(panel, n_origins=2, horizons=[6, 24])
    preds, stats = run(_fc(), panel, spec, context=96, name=MODEL_NAME)
    assert set(PRED_SCHEMA) <= set(preds.columns)
    assert {"0.1", "0.5", "0.9"} <= set(preds.columns)
    assert (preds["model"] == MODEL_NAME).all()
    assert len(preds) == 3 * 2 * (6 + 24)
    assert not preds.duplicated(["item_id", "origin", "horizon", "timestamp"]).any()
    assert (preds["timestamp"] > preds["origin"]).all()
    assert stats.calls == 4 and not stats.failures


# ────────────────────────────────────────────────────────────── roster

def test_only_the_209m_size_is_on_the_roster():
    assert MODEL_NAME == "Timestral-209M-ZS" == ua.TIMESTRAL
    assert MODEL_NAME in ua.ORDER
    assert ua.ADDED_LATER == [TIMESFM, MODEL_NAME]
    assert ua.FAMILY[MODEL_NAME] == "foundation"
    assert MODEL_NAME in ua.LABEL and MODEL_NAME in ua.SHORT
    off = [model_name(s) for s in SIZES if s != "209M"]
    assert not set(off) & set(ua.ORDER)
    with tempfile.TemporaryDirectory() as d:
        base = {"item_id": ["T0"], "timestamp": [pd.Timestamp("2025-01-02")], "step": [1],
                "actual": [1.0], "pred": [1.0], "origin": [pd.Timestamp("2025-01-01 23:00")],
                "horizon": [6]}
        for m in [MODEL_NAME, *off]:
            pd.DataFrame({**base, "model": [m]}).to_parquet(Path(d) / f"predictions_{m}.parquet")
        assert set(load_all_predictions(Path(d)).model) == {MODEL_NAME}


def _scored(models):
    rows = []
    rng = np.random.default_rng(11)
    origins = pd.date_range("2025-03-01", periods=24, freq="23h")
    for m, level in models.items():
        for o in origins:
            for t in range(3):
                for h in ua.HORIZONS:
                    e = abs(rng.normal(level, 0.1))
                    rows.append({"model": m, "origin": o, "item_id": f"T{t}", "horizon": h,
                                 "abs_err": e / 3, "scaled_abs": e})
    return pd.DataFrame(rows)


def test_adding_timestral_redraws_neither_published_significance_table():
    """The significance tables draw from one seeded stream, TimesFM's straight after
    Chronos-2's. Adding Timestral must leave every published interval in both byte-identical,
    append Chronos-2-vs-Timestral rows, and draw Timestral's own table after all of them."""
    scored = _scored({"Chronos2-ZS": 0.6, "NPTS": 0.7, "SeasonalNaive": 1.0,
                      TIMESFM: 0.65, MODEL_NAME: 0.62})
    without = scored[scored.model != MODEL_NAME]

    def draw(s, fn=ua.draw_significance):
        significance.RNG = np.random.default_rng(significance.SEED)
        return fn(s)

    try:
        sig_a, tfm_a, tsl_a = draw(without)
        sig_b, tfm_b, tsl_b = draw(scored)
        # The same drawn naively - every opponent in one Chronos-2 pass, TimesFM's after - shows
        # the guard is load-bearing: TimesFM's intervals would move.
        naive_tfm = draw(scored, lambda s: (ua.significance_vs_all(s), ua.significance_vs_all(
            s, base=TIMESFM, opponents=["NPTS", "SeasonalNaive"]))[1])
    finally:
        significance.RNG = np.random.default_rng(significance.SEED)      # leave it as found

    pd.testing.assert_frame_equal(tfm_a, tfm_b)
    assert MODEL_NAME not in set(tfm_b.opponent)
    old = sig_b[sig_b.opponent != MODEL_NAME].reset_index(drop=True)
    pd.testing.assert_frame_equal(sig_a.reset_index(drop=True), old)
    assert (sig_b.opponent == MODEL_NAME).sum() == len(ua.HORIZONS) * 2
    assert list(sig_b.opponent.unique())[-1] == MODEL_NAME
    assert not naive_tfm.equals(tfm_b)
    # Timestral's own table is drawn last, against everything but Chronos-2 (already a row above).
    assert tsl_a.empty
    assert set(tsl_b.opponent) == {"NPTS", "SeasonalNaive", TIMESFM}
    assert (tsl_b.base == MODEL_NAME).all()


def _synthetic_tables():
    from src.models.metrics import aggregate_metrics
    rows = []
    rng = np.random.default_rng(3)
    models = {MODEL_NAME: 0.64, TIMESFM: 0.70, "Chronos2-ZS": 0.65, "NPTS": 0.72,
              "SeasonalNaive": 1.0}
    for m, level in models.items():
        for h in ua.HORIZONS:
            for t in range(6):
                mase = level + rng.normal(0, 0.05)
                rows.append({"model": m, "horizon": h, "item_id": f"T{t}", "n": 24,
                             "mae": mase / 3, "rmse": mase / 2, "mase": mase,
                             "rmsse": mase * 0.8, "mean_actual": 1.0 + t,
                             "mean_pred": 1.0, "scaled_ok": True})
    pt = pd.DataFrame(rows)
    return pt, aggregate_metrics(pt)


def test_benchmark_document_gains_section_9_and_leaves_section_8_alone():
    from src.models import score_benchmark as sb
    pt, agg = _synthetic_tables()
    quality = pd.DataFrame({"item_id": [f"T{t}" for t in range(6)],
                            "sensor_trust_tier": ["healthy"] * 6})
    grid = {"tanks": 6, "origins": 1, "stride": 23, "hours_covered": 1, "n_models": 5,
            "first_origin": "a", "last_origin": "b", "rows_per_model": 1}
    h2h = sb.head_to_head(pt, agg, TIMESFM, sb.TIMESFM_OPPONENTS)
    tpt = sb.timesfm_per_tank(pt, quality)
    ch = sb.CHALLENGERS[1]
    ts_h2h = sb.head_to_head(pt, agg, ch.model, ch.opponents)
    ts_tpt = sb.challenger_per_tank(pt, quality, ch)
    before = sb.to_markdown(agg, pd.DataFrame(), None, grid, h2h=h2h, tpt=tpt)
    after = sb.to_markdown(agg, pd.DataFrame(), None, grid, h2h=h2h, tpt=tpt,
                           challengers=[(ch, ts_h2h, ts_tpt)])
    assert after.startswith(before)                       # §8 and everything above unchanged
    added = after[len(before):]
    assert "## 9. Timestral 209M zero-shot against Chronos-2, TimesFM 2.5, NPTS and " \
           "SeasonalNaive-24" in added
    assert "| vs Chronos-2 zero-shot | vs TimesFM 2.5 | vs NPTS | vs SeasonalNaive-24 |" in added
    assert set(ts_tpt.winner.dropna()) <= {"Timestral", "Chronos-2", "tie"}
    assert "timestral_mase" in ts_tpt and "timestral_vs_chronos2_pct" in ts_tpt


def test_timesfm_summary_keeps_its_published_keys():
    scored = _scored({"Chronos2-ZS": 0.6, "NPTS": 0.7, "SeasonalNaive": 1.0,
                      TIMESFM: 0.65, MODEL_NAME: 0.62})
    try:
        significance.RNG = np.random.default_rng(significance.SEED)
        sig, sig_tfm, sig_tsl = ua.draw_significance(scored)
    finally:
        significance.RNG = np.random.default_rng(significance.SEED)
    lb = pd.DataFrame([{"model": m, "horizon": h, "rank_mase": i + 1, "coverage": 0.8}
                       for h in ua.HORIZONS
                       for i, m in enumerate(["Chronos2-ZS", MODEL_NAME, TIMESFM, "NPTS",
                                              "SeasonalNaive"])])
    wm = pd.DataFrame([{"model_a": a, "model_b": b, "a_wins": 1}
                       for a in (TIMESFM, MODEL_NAME) for b in ("NPTS", "SeasonalNaive")])
    tfm = ua._challenger_summary(TIMESFM, "timesfm", sig, sig_tfm, lb, wm)
    assert list(tfm) == [
        "model", "rank_by_horizon", "chronos2_improvement_over_timesfm_pct",
        "chronos2_better_significantly_at", "timesfm_better_significantly_at",
        "improvement_pct_h24", "opponents_beaten_significantly_all_horizons",
        "opponents_beaten_by_both_tests_all_horizons", "chronos2_vs_timesfm_dm_p_by_horizon",
        "tank_horizon_cells_won_vs", "coverage_by_horizon"]
    tsl = ua._challenger_summary(MODEL_NAME, "timestral", sig, sig_tsl, lb, wm)
    assert "chronos2_improvement_over_timestral_pct" in tsl and tsl["model"] == MODEL_NAME


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
