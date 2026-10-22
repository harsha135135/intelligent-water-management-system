"""Tests for the TimesFM 3.0 integration.

Every test uses a stub in place of the real model, so nothing is downloaded and the suite runs in
seconds. The stub has ``timesfm3.TimesFM3Forecaster``'s own ``predict_batch(contexts=..., horizon=...,
...)`` signature and output - one item per series with ``.quantiles`` of shape ``(horizon, Q)`` -
and a ``config.quantiles`` list, so the adapter code under test is exactly the code the benchmark
runs.

    python -m tests.test_timesfm3
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import significance  # noqa: E402
from src.models import unified_analysis as ua  # noqa: E402
from src.models.backtest import make_spec  # noqa: E402
from src.models.metrics import PRED_SCHEMA  # noqa: E402
from src.models.timesfm3_forecasting import (  # noqa: E402
    DEPLOYABLE, FORECAST_FLAGS, MODEL_NAME, TimesFM3Adapter, run,
)
from src.models.timesfm_forecasting import MODEL_NAME as TIMESFM, RunStats, forecast_one  # noqa: E402
from src.models.timestral_forecasting import MODEL_NAME as TIMESTRAL  # noqa: E402

TARGET = "Outflow in KL"
SENTINEL = 1.0e6
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]          # the released checkpoint's


@dataclass
class _Out:
    quantiles: np.ndarray
    forecast: np.ndarray | None = None
    ts_id: str | None = None


@dataclass
class _Config:
    quantiles: list = field(default_factory=lambda: list(LEVELS))


class StubModel:
    """Stands in for ``TimesFM3Forecaster``: repeats each context's last day as the median, puts
    quantile level ``p`` at ``median + 2 (p - 0.5)``, and records every input."""

    def __init__(self, levels=LEVELS, fail_on_value=None, extra_steps=0):
        self.config = _Config(list(levels))
        self.fail_on_value, self.extra_steps = fail_on_value, extra_steps
        self.calls, self.kwargs = [], []

    def predict_batch(self, contexts, horizon, **kwargs):
        xs = [np.array(c, dtype=float, copy=True) for c in contexts]
        self.calls.append(xs)
        self.kwargs.append(kwargs)
        for x in xs:
            if self.fail_on_value is not None and (x == self.fail_on_value).any():
                raise ValueError("degenerate input")
        h = horizon + self.extra_steps
        for x in xs:
            day = np.nan_to_num(x[-24:])
            median = np.tile(day, int(np.ceil(h / 24)))[:h]
            q = median[:, None] + 2 * (np.asarray(self.config.quantiles) - 0.5)
            yield _Out(quantiles=q.astype(np.float32))


def _fc(model=None):
    fp = {"model_id": "stub", "revision": "0", "timesfm": "test", "torch": "test",
          "device": "cpu", "forecast_flags": dict(FORECAST_FLAGS)}
    return TimesFM3Adapter(model=model or StubModel(), fingerprint=fp)


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
    shuffled = [0.5, 0.9, 0.1, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6]
    p, q = _fc(StubModel(levels=shuffled)).forecast(24, [np.arange(96, dtype=float)])
    p0, q0 = _fc(StubModel()).forecast(24, [np.arange(96, dtype=float)])
    np.testing.assert_allclose(q, q0)
    np.testing.assert_allclose(p, q[..., 5])                # slot 0 is the median, as slot 5


def test_a_missing_quantile_level_raises():
    try:
        _fc(StubModel(levels=[lv for lv in LEVELS if lv != 0.2])).forecast(
            24, [np.arange(96, dtype=float)])
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


def test_the_evaluator_settings_are_passed_explicitly():
    stub = StubModel()
    _fc(stub).forecast(24, [np.arange(96, dtype=float)])
    assert stub.kwargs == [{"return_quantiles": True, "use_symmetric_averaging": True,
                            "make_positive": True, "sort_quantiles": True, "use_znorm": False,
                            "padding_mode": "none"}]


def test_a_block_split_across_forward_passes_is_refused():
    stub = StubModel()
    stub.config.per_core_batch_size = 4                       # 3 series need 6 queries
    try:
        _fc(stub).forecast(24, [np.arange(96, dtype=float)] * 3)
    except ValueError as exc:
        assert "one forward pass" in str(exc)
    else:
        raise AssertionError("a block that does not fit one forward pass must be refused")
    stub.config.per_core_batch_size = 6
    _fc(stub).forecast(24, [np.arange(96, dtype=float)] * 3)  # exactly fits


def test_each_tank_is_its_own_univariate_series():
    stub = StubModel()
    a, b = np.arange(96, dtype=float), np.arange(90, dtype=float) + 1
    _fc(stub).forecast(24, [a, b])
    sent = stub.calls[0]
    assert len(sent) == 2 and sent[0].ndim == sent[1].ndim == 1
    np.testing.assert_array_equal(sent[1], b)                 # not padded by the adapter


# ────────────────────────────────────────────────────────────── through the shared plumbing

def test_point_is_the_median_and_quantiles_are_mapped():
    panel = _panel()
    origin = panel["timestamp"].iloc[-49]
    out = forecast_one(_fc(), panel, origin, 24, target_col=TARGET, context=96)
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
    assert all(len(x) == 96 and np.nanmax(x) < SENTINEL for x in sent), "a post-origin value leaked"
    for k, item in enumerate(["T0", "T1", "T2"]):
        at_origin = panel[(panel.item_id == item) & (panel.timestamp == origin)][TARGET].item()
        assert np.isclose(sent[k][-1], at_origin), "context must end exactly at the origin"
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
              TARGET] = 777.0
    stats = RunStats()
    out = forecast_one(_fc(StubModel(fail_on_value=777.0)), panel, origin, 24,
                       target_col=TARGET, context=96, stats=stats)
    assert [f["item_id"] for f in stats.failures] == ["T2"]
    assert out.loc[out.item_id == "T2", "pred"].isna().all()
    assert out.loc[out.item_id != "T2", "pred"].notna().all()


def test_output_schema_matches_the_benchmark():
    panel = _panel()
    spec = make_spec(panel, n_origins=2, horizons=[6, 24])
    preds, stats = run(_fc(), panel, spec, context=96)
    assert set(PRED_SCHEMA) <= set(preds.columns)
    assert (preds["model"] == MODEL_NAME).all()
    assert not preds.duplicated(["item_id", "origin", "horizon", "timestamp"]).any()
    assert (preds["timestamp"] > preds["origin"]).all()
    assert stats.calls == 4 and not stats.failures


# ────────────────────────────────────────────────────────────── roster and licence

def test_on_the_roster_but_never_a_deployment_candidate():
    assert MODEL_NAME == "TimesFM3-ZS" == ua.TIMESFM3 and MODEL_NAME != TIMESFM
    assert MODEL_NAME in ua.ORDER and ua.FAMILY[MODEL_NAME] == "foundation"
    assert ua.ADDED_LATER == [TIMESFM, TIMESTRAL, MODEL_NAME]
    assert DEPLOYABLE is False and MODEL_NAME in ua.NOT_DEPLOYABLE


def test_the_dock_bundle_refuses_it():
    from src.models import build_dock_bundle as bdb
    try:
        bdb.check_deployable(MODEL_NAME)
    except SystemExit as exc:
        assert "licence" in str(exc).lower()
    else:
        raise AssertionError("the extension bundle must refuse a non-deployable model")
    bdb.check_deployable(ua.CHRONOS)                         # a deployable one passes


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


def test_adding_timesfm3_redraws_no_existing_interval():
    """Every row of the three tables that existed before TimesFM 3.0 must come out
    byte-identical with it present; its rows are appended, and its own table drawn last."""
    scored = _scored({"Chronos2-ZS": 0.6, "NPTS": 0.7, "SeasonalNaive": 1.0,
                      TIMESFM: 0.65, TIMESTRAL: 0.62, MODEL_NAME: 0.63})
    without = scored[scored.model != MODEL_NAME]

    def draw(s):
        significance.RNG = np.random.default_rng(significance.SEED)
        return ua.draw_significance(s)

    try:
        sig_a, tfm_a, tsl_a, new_a = draw(without)
        sig_b, tfm_b, tsl_b, new_b = draw(scored)
    finally:
        significance.RNG = np.random.default_rng(significance.SEED)

    pd.testing.assert_frame_equal(tfm_a, tfm_b)
    for before, after in ((sig_a, sig_b), (tsl_a, tsl_b)):
        old = after[after.opponent != MODEL_NAME].reset_index(drop=True)
        pd.testing.assert_frame_equal(before.reset_index(drop=True), old)
        assert list(after.opponent.unique())[-1] == MODEL_NAME          # appended, not inserted
        assert (after.opponent == MODEL_NAME).sum() == len(ua.HORIZONS) * 2
    assert new_a.empty and (new_b.base == MODEL_NAME).all()
    assert set(new_b.opponent) == {"NPTS", "SeasonalNaive", TIMESFM}   # no pair drawn twice


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:                               # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
