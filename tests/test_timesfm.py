"""Tests for the TimesFM integration.

Every test uses a stub in place of the real model, so nothing is downloaded and the suite runs in
seconds. The stub has TimesFM's own ``forecast(horizon=..., inputs=[...])`` signature and output
layout - ``(n, horizon)`` points and ``(n, horizon, 10)`` quantiles, mean first - so the adapter
code under test is exactly the code the benchmark runs.

    python -m tests.test_timesfm
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import significance  # noqa: E402
from src.models import unified_analysis as ua  # noqa: E402
from src.models.backtest import actuals_after, make_spec  # noqa: E402
from src.models.metrics import (  # noqa: E402
    PRED_SCHEMA, aggregate_metrics, attach_scales, per_tank_metrics,
)
from src.models.score_benchmark import (  # noqa: E402
    TIMESFM_OPPONENTS, head_to_head, load_all_predictions, relative_words, timesfm_per_tank,
    to_markdown,
)
from src.models.timesfm_forecasting import (  # noqa: E402
    MODEL_NAME, N_OUTPUTS, QUANTILE_INDEX, BlockCache, RunStats, TimesFMForecaster, cache_key,
    forecast_one, run,
)

TARGET = "Outflow in KL"
SENTINEL = 1.0e6


class StubModel:
    """Stands in for ``TimesFM_2p5_200M_torch``: repeats each context's last day, with a fixed
    spread for the quantiles, and records every input it is handed."""

    def __init__(self, fail_on_value=None):
        self.calls = []
        self.fail_on_value = fail_on_value

    def forecast(self, horizon, inputs):
        self.calls.append([np.array(x, copy=True) for x in inputs])
        if self.fail_on_value is not None and any(
                np.nanmax(x) == self.fail_on_value for x in inputs):
            raise RuntimeError("stub failure")
        pts = []
        for x in inputs:
            day = np.nan_to_num(np.asarray(x, dtype=float)[-24:])
            pts.append(np.tile(day, int(np.ceil(horizon / 24)))[:horizon])
        point = np.array(pts, dtype=np.float32)
        spread = np.linspace(-0.4, 0.4, 9, dtype=np.float32)
        q = point[..., None] + spread                             # q10..q90
        quantiles = np.concatenate([point[..., None], q], axis=-1)  # mean first, like TimesFM
        return point, quantiles


def _fc(model=None, **extra):
    fp = {"model_id": "stub", "revision": "0", "timesfm": "test", "torch": "test",
          "torch_compile": False, "device": "cpu", "forecast_config": {"max_context": 96}}
    fp.update(extra)
    return TimesFMForecaster(model=model or StubModel(), fingerprint=fp)


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

def test_adapter_accepts_repository_data():
    panel = _panel()
    origin = panel["timestamp"].iloc[-49]
    out = forecast_one(_fc(), panel, origin, 24, target_col=TARGET, context=96)
    assert len(out) == 3 * 24
    assert set(out["item_id"]) == {"T0", "T1", "T2"}
    assert out["pred"].notna().all()
    for _, g in out.groupby("item_id"):
        assert g["step"].tolist() == list(range(1, 25))


def test_forecast_length_matches_horizon_for_every_horizon():
    panel = _panel()
    origin = panel["timestamp"].iloc[-200]
    for h in (6, 12, 24, 48, 72, 168):
        out = forecast_one(_fc(), panel, origin, h, target_col=TARGET, context=96)
        assert out.groupby("item_id").size().eq(h).all(), h


def test_timestamps_align_with_the_scored_window():
    panel = _panel()
    origin = panel["timestamp"].iloc[-49]
    out = forecast_one(_fc(), panel, origin, 24, target_col=TARGET, context=96)
    fut = actuals_after(panel, origin, 24)
    assert (out["timestamp"] == origin + pd.to_timedelta(out["step"], unit="h")).all()
    m = out.merge(fut, on=["item_id", "timestamp"])
    assert len(m) == len(out) and np.allclose(m["actual"], m[TARGET])


def test_point_is_the_median_and_quantiles_are_mapped():
    panel = _panel()
    origin = panel["timestamp"].iloc[-49]
    out = forecast_one(_fc(), panel, origin, 24, target_col=TARGET, context=96)
    # Stub: q10 = point - 0.4, q50 = point, q90 = point + 0.4, clipped at zero.
    assert np.allclose(out["0.5"], out["pred"])
    assert np.allclose(out["0.9"], np.clip(out["pred"] + 0.4, 0, None), atol=1e-5)
    assert np.allclose(out["0.1"], np.clip(out["pred"] - 0.4, 0, None), atol=1e-5)
    assert QUANTILE_INDEX == {"0.1": 1, "0.5": 5, "0.9": 9}


def test_no_future_observation_reaches_the_model():
    panel = _panel()
    origin = panel["timestamp"].iloc[-60]
    poisoned = panel.copy()
    poisoned.loc[poisoned["timestamp"] > origin, TARGET] = SENTINEL
    stub, stats = StubModel(), RunStats()
    forecast_one(_fc(stub), poisoned, origin, 48, target_col=TARGET, context=96, stats=stats)
    sent = [x for call in stub.calls for x in call]
    assert sent and all(np.nanmax(x) < SENTINEL for x in sent), "a post-origin value leaked"
    assert all(len(x) == 96 for x in sent)
    for item, x in zip(["T0", "T1", "T2"], sent):
        at_origin = panel[(panel.item_id == item) & (panel.timestamp == origin)][TARGET].item()
        assert np.isclose(x[-1], at_origin), "context must end exactly at the origin"
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


# ────────────────────────────────────────────────────────────── schema and metrics

def test_output_schema_matches_the_benchmark():
    panel = _panel()
    spec = make_spec(panel, n_origins=2, horizons=[6, 24])
    preds, stats = run(_fc(), panel, spec, context=96)
    assert set(PRED_SCHEMA) <= set(preds.columns)
    assert {"0.1", "0.5", "0.9"} <= set(preds.columns)
    assert (preds["model"] == MODEL_NAME).all()
    assert len(preds) == 3 * 2 * (6 + 24)
    assert not preds.duplicated(["item_id", "origin", "horizon", "timestamp"]).any()
    assert (preds["timestamp"] > preds["origin"]).all()
    assert stats.calls == 4 and not stats.failures


def test_metrics_are_the_repository_implementation():
    """The TimesFM comparison is computed from ``metrics.per_tank_metrics`` output, never
    re-derived: its macro values are exactly ``metrics.aggregate_metrics``'."""
    panel = _panel(n_tanks=4, n_hours=24 * 40)
    spec = make_spec(panel, n_origins=3, horizons=[24])
    preds, _ = run(_fc(), panel, spec, context=96)
    other = preds.assign(model="Chronos2-ZS", pred=preds["pred"] * 1.05)
    scored = attach_scales(pd.concat([preds, other]), panel, target_col=TARGET)
    pt = per_tank_metrics(scored)
    agg = aggregate_metrics(pt)
    h2h = head_to_head(pt, agg, MODEL_NAME, ["Chronos2-ZS"])
    for metric in ("MAE", "RMSE", "MASE", "RMSSE"):
        r = h2h[h2h.metric == metric].iloc[0]
        a = agg[(agg.model == MODEL_NAME)][f"macro_{metric.lower()}"].item()
        b = agg[(agg.model == "Chronos2-ZS")][f"macro_{metric.lower()}"].item()
        assert r.model_value == a and r.opponent_value == b
        assert np.isclose(r.improvement_pct, 100 * (b - a) / b)
        assert r.model_wins + r.opponent_wins + r.ties == r.tanks_compared == 4


# ────────────────────────────────────────────────────────────── cache

def test_cache_key_is_deterministic_and_input_sensitive():
    fp = _fc().fingerprint
    x = np.arange(96, dtype=float)
    o = pd.Timestamp("2025-01-20 03:00")
    base = cache_key(fp, o, 24, 96, [("T0", x)])
    assert base == cache_key(fp, o, 24, 96, [("T0", x.copy())])
    y = x.copy()
    y[10] += 1e-9
    assert base != cache_key(fp, o, 24, 96, [("T0", y)])                       # one value
    assert base != cache_key(fp, o + pd.Timedelta(hours=1), 24, 96, [("T0", x)])  # origin
    assert base != cache_key(fp, o, 48, 96, [("T0", x)])                       # horizon
    assert base != cache_key(fp, o, 24, 96, [("T1", x)])                       # series
    assert base != cache_key(_fc(revision="1").fingerprint, o, 24, 96, [("T0", x)])  # checkpoint
    n1, n2 = x.copy(), x.copy()
    n1[5], n2[5] = np.nan, -np.float64("nan")
    assert cache_key(fp, o, 24, 96, [("T0", n1)]) == cache_key(fp, o, 24, 96, [("T0", n2)])


def test_cache_hit_reproduces_the_forecast_without_calling_the_model():
    panel = _panel()
    origin = panel["timestamp"].iloc[-49]
    with tempfile.TemporaryDirectory() as d:
        cache = BlockCache(Path(d))
        stub = StubModel()
        first = forecast_one(_fc(stub), panel, origin, 24, target_col=TARGET, context=96,
                             cache=cache)
        stats = RunStats()
        second = forecast_one(_fc(stub), panel, origin, 24, target_col=TARGET, context=96,
                              cache=cache, stats=stats)
        assert len(stub.calls) == 1 and stats.cache_hits == 1
        pd.testing.assert_frame_equal(first, second)


def test_incomplete_or_corrupt_cache_entries_are_not_trusted():
    with tempfile.TemporaryDirectory() as d:
        cache = BlockCache(Path(d))
        p = np.ones((2, 24), np.float32)
        q = np.ones((2, 24, N_OUTPUTS), np.float32)
        cache.put("k", p, q, 1.0)
        assert cache.get("k", 2, 24) is not None
        assert cache.get("k", 3, 24) is None                   # wrong series count
        assert cache.get("k", 2, 48) is None                   # wrong horizon
        p_bad = p.copy()
        p_bad[0, 0] = np.nan
        cache.put("nan", p_bad, q, 1.0)
        assert cache.get("nan", 2, 24) is None                 # non-finite
        (Path(d) / "trunc.npz").write_bytes(b"not an npz")
        assert cache.get("trunc", 2, 24) is None               # corrupt file
        assert not list(Path(d).glob("*.tmp"))                 # writes are atomic


# ────────────────────────────────────────────────────────────── failures

def test_a_failing_series_is_recorded_and_left_unscored():
    panel = _panel()
    origin = panel["timestamp"].iloc[-49]
    panel.loc[(panel.item_id == "T2") & (panel.timestamp == origin - pd.Timedelta(hours=3)),
              TARGET] = 777.0                              # the stub fails on this value
    with tempfile.TemporaryDirectory() as d:
        cache, stats = BlockCache(Path(d)), RunStats()
        out = forecast_one(_fc(StubModel(fail_on_value=777.0)), panel, origin, 24,
                           target_col=TARGET, context=96, cache=cache, stats=stats)
        assert [f["item_id"] for f in stats.failures] == ["T2"]
        f = stats.failures[0]
        assert f["horizon"] == 24 and f["origin"] == str(origin) and f["error"] == "RuntimeError"
        assert out.loc[out.item_id == "T2", "pred"].isna().all()   # kept, not dropped
        assert out.loc[out.item_id != "T2", "pred"].notna().all()
        assert not list(Path(d).glob("*.npz")), "a block with a failure must not be cached"


# ────────────────────────────────────────────────────────────── benchmark tables

def _synthetic_tables():
    rows = []
    rng = np.random.default_rng(3)
    models = {MODEL_NAME: 0.70, "Chronos2-ZS": 0.65, "NPTS": 0.72, "SeasonalNaive": 1.0}
    for m, level in models.items():
        for h in (6, 12, 24, 48, 72, 168):
            for t in range(6):
                mase = level + rng.normal(0, 0.05)
                rows.append({"model": m, "horizon": h, "item_id": f"T{t}", "n": 24,
                             "mae": mase / 3, "rmse": mase / 2, "mase": mase,
                             "rmsse": mase * 0.8, "mean_actual": 1.0 + t,
                             "mean_pred": 1.0, "scaled_ok": True})
    pt = pd.DataFrame(rows)
    return pt, aggregate_metrics(pt)


def test_benchmark_tables_include_timesfm():
    pt, agg = _synthetic_tables()
    h2h = head_to_head(pt, agg, MODEL_NAME, TIMESFM_OPPONENTS)
    assert len(h2h) == 3 * 6 * 4
    assert set(h2h.opponent) == {"Chronos2-ZS", "NPTS", "SeasonalNaive"}
    quality = pd.DataFrame({"item_id": [f"T{t}" for t in range(6)],
                            "sensor_trust_tier": ["healthy"] * 6})
    tpt = timesfm_per_tank(pt, quality)
    assert len(tpt) == 6 * 6
    # Regression: the winner must agree with the two MASE values on every row.
    expect = np.where(tpt.timesfm_mase < tpt.chronos2_mase, "TimesFM", "Chronos-2")
    assert (tpt.winner == expect).all()
    for h, g in tpt.groupby("horizon"):
        r = h2h[(h2h.opponent == "Chronos2-ZS") & (h2h.horizon == h) & (h2h.metric == "MASE")]
        assert (g.winner == "TimesFM").sum() == r.model_wins.item()
    md = to_markdown(agg, pd.DataFrame(), None, {
        "tanks": 6, "origins": 1, "stride": 23, "hours_covered": 1, "n_models": 4,
        "first_origin": "a", "last_origin": "b", "rows_per_model": 1}, h2h=h2h, tpt=tpt)
    assert "## 8. TimesFM 2.5 zero-shot" in md and "TimesFM 2.5 zero-shot" in md


def test_relative_difference_is_worded_by_direction():
    assert relative_words(5.0) == "5.00 % lower"      # TimesFM's error lower -> better
    assert relative_words(-5.0) == "5.00 % higher"    # TimesFM's error higher -> worse
    assert relative_words(0.001) == "the same"
    assert relative_words(float("nan")) == ""


def test_roster_skips_work_in_progress_models():
    with tempfile.TemporaryDirectory() as d:
        base = {"item_id": ["T0"], "timestamp": [pd.Timestamp("2025-01-02")], "step": [1],
                "actual": [1.0], "pred": [1.0], "origin": [pd.Timestamp("2025-01-01 23:00")],
                "horizon": [6]}
        pd.DataFrame({**base, "model": [MODEL_NAME]}).to_parquet(
            Path(d) / f"predictions_{MODEL_NAME}.parquet")
        pd.DataFrame({**base, "model": ["Some-WIP-Model"]}).to_parquet(
            Path(d) / "predictions_Some-WIP-Model.parquet")
        got = load_all_predictions(Path(d))
        assert set(got.model) == {MODEL_NAME}


def test_adding_a_model_does_not_redraw_published_intervals():
    """The significance bootstrap draws from one seeded stream. Adding TimesFM must leave every
    existing opponent's interval byte-identical, however it sorts in display order."""
    rows = []
    rng = np.random.default_rng(11)
    origins = pd.date_range("2025-03-01", periods=24, freq="23h")
    for m, level in {"Chronos2-ZS": 0.6, "NPTS": 0.7, "SeasonalNaive": 1.0,
                     MODEL_NAME: 0.65}.items():
        for o in origins:
            for t in range(3):
                for h in ua.HORIZONS:
                    e = abs(rng.normal(level, 0.1))
                    rows.append({"model": m, "origin": o, "item_id": f"T{t}", "horizon": h,
                                 "abs_err": e / 3, "scaled_abs": e})
    scored = pd.DataFrame(rows)
    without = scored[scored.model != MODEL_NAME]

    significance.RNG = np.random.default_rng(significance.SEED)
    a = ua.significance_vs_all(without)
    significance.RNG = np.random.default_rng(significance.SEED)
    b = ua.significance_vs_all(scored)
    significance.RNG = np.random.default_rng(significance.SEED)          # leave it as found
    old = b[b.opponent != MODEL_NAME].reset_index(drop=True)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), old)
    assert (b.opponent == MODEL_NAME).sum() == len(ua.HORIZONS) * 2


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
