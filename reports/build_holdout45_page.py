"""Build the interactive 45-day holdout explorer — actual vs predicted, hour by hour.

The published figures in ``phase3_results_page.html`` are static PNGs: they show that Chronos-2
wins, but they cannot answer "what did each model say on 3 April?". This page can. It carries the
full continuous holdout — 2026-03-09 .. 2026-04-22, 45 consecutive daily origins at horizon 24 h,
24 tanks, 1 080 hours — as embedded JSON, and draws it client-side so any tank, any date range
and any subset of models can be read off directly.

Sources (all raw, uncalibrated point forecasts, so every model is directly comparable):
    results/chronos2/calibrated/predictions_calibrated.parquet      Chronos2-ZS, NPTS, TimesFM2.5-ZS,
                                                                     Timestral-209M-ZS
    results/chronos2/holdout45_patchtst/predictions_holdout45_patchtst.parquet   PatchTST

All were produced on the identical origin grid. The two fitted models share the fit cutoff
(2026-03-08 23:00); the three foundation models are zero-shot with their context cut at each
origin. So no model saw a single hour of the window it is scored on. TimesFM and Timestral join the
calibrated run through ``python -m src.models.calibrated_holdout --add timesfm`` / ``--add timestral``.

    python reports/build_holdout45_page.py     # -> reports/holdout45_explorer.html
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAL = ROOT / "results/chronos2/calibrated/predictions_calibrated.parquet"
PT = ROOT / "results/chronos2/holdout45_patchtst/predictions_holdout45_patchtst.parquet"
OUT = ROOT / "reports" / "holdout45_explorer.html"

# Order fixes the colour assignment: slot 1 blue, slot 2 orange, slot 3 aqua, then TimesFM's
# magenta (validated against the other three on both surfaces: light #b0177f, dark #d65bb1), then
# Timestral's violet (light #5000d0, dark #7850b8, one hue on both surfaces; all five pass every
# gate at all-pairs on each surface). The figures draw Timestral dark green, but no green clears
# the normal-vision floor against this page's dark-surface aqua, so the page has its own swatch —
# as it already does for PatchTST. Colour follows the model, never its rank, so toggling one off
# never repaints the others.
MODELS = [
    {"key": "c2", "label": "Chronos-2 ZS", "src": "Chronos2-ZS",
     "note": "foundation model, zero-shot"},
    {"key": "npts", "label": "NPTS", "src": "NPTS",
     "note": "incumbent, fitted on this campus"},
    {"key": "patchtst", "label": "PatchTST", "src": "PatchTST",
     "note": "transformer, trained on this campus"},
    {"key": "timesfm", "label": "TimesFM 2.5", "src": "TimesFM2.5-ZS",
     "note": "second foundation model, zero-shot"},
    {"key": "timestral", "label": "Timestral 209M", "src": "Timestral-209M-ZS",
     "note": "third foundation model, zero-shot"},
]
# The models read from the calibrated run; PatchTST comes from its own holdout file.
CAL_MODELS = {"Chronos2-ZS", "NPTS", "TimesFM2.5-ZS", "Timestral-209M-ZS"}
ADD_HINT = {"TimesFM2.5-ZS": "timesfm", "Timestral-209M-ZS": "timestral"}
NWORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(len(MODELS), str(len(MODELS)))

PRETTY = {"I&H_BLOCK": "I&H Block", "NBX": "NBX"}


def tank_label(item_id: str) -> str:
    if item_id in PRETTY:
        return PRETTY[item_id]
    return item_id.replace("_", " ").title().replace("Oht", "OHT").replace("Ro", "RO") \
        .replace("Gjbc", "GJBC").replace("Mm ", "MM ").replace("Mrd", "MRD") \
        .replace("Me ", "ME ").replace("It ", "IT ")


def r2(a):
    """Round to 2 dp and turn NaN into JSON null — the payload is ~10× smaller for it."""
    return [None if not np.isfinite(v) else round(float(v), 2) for v in a]


def load() -> pd.DataFrame:
    frames = []
    cal = pd.read_parquet(CAL, columns=["item_id", "timestamp", "model", "actual",
                                        "pred", "q10", "q90"])
    frames.append(cal[cal["model"].isin(CAL_MODELS)])
    if PT.exists():
        pt = pd.read_parquet(PT, columns=["item_id", "timestamp", "model", "actual",
                                          "pred", "q10", "q90"])
        frames.append(pt)
    else:
        raise SystemExit(f"missing {PT}\n  run: python -m src.models.holdout45_patchtst")
    df = pd.concat(frames, ignore_index=True)
    src = {m["src"] for m in MODELS}
    missing = src - set(df["model"].unique())
    if missing:
        hint = "".join(f"\n  run: python -m src.models.calibrated_holdout --add {ADD_HINT[m]}"
                       for m in sorted(missing) if m in ADD_HINT)
        raise SystemExit(f"models absent from the sources: {sorted(missing)}{hint}")
    return df[df["model"].isin(src)]


def build_payload(df: pd.DataFrame) -> dict:
    hours = pd.DatetimeIndex(sorted(df["timestamp"].unique()))
    tanks = sorted(df["item_id"].unique())
    n_h, n_t = len(hours), len(tanks)
    h_ix = {t: i for i, t in enumerate(hours)}
    t_ix = {t: i for i, t in enumerate(tanks)}

    # One (tank × hour) matrix per series. The actual is taken from Chronos-2's rows; every
    # model is scored on identical rows, so any of them would give the same actuals.
    def matrix(sub: pd.DataFrame, col: str) -> np.ndarray:
        m = np.full((n_t, n_h), np.nan)
        m[sub["item_id"].map(t_ix).to_numpy(), sub["timestamp"].map(h_ix).to_numpy()] = \
            sub[col].to_numpy()
        return m

    by_model = {m["key"]: df[df["model"] == m["src"]] for m in MODELS}
    actual = matrix(by_model["c2"], "actual")
    for key, sub in by_model.items():                      # the actuals must agree everywhere
        a = matrix(sub, "actual")
        both = np.isfinite(a) & np.isfinite(actual)
        assert np.allclose(a[both], actual[both]), f"{key}: actuals disagree with Chronos-2's"

    preds = {k: matrix(s, "pred") for k, s in by_model.items()}
    lo = {k: matrix(s, "q10") for k, s in by_model.items()}
    hi = {k: matrix(s, "q90") for k, s in by_model.items()}

    present = np.isfinite(actual)                          # the scored cells, shared by all models
    tank_rows = []
    for i, t in enumerate(tanks):
        row = {"id": t, "label": tank_label(t), "actual": r2(actual[i]),
               "cov": [int(v) for v in present[i]],
               "pred": {k: r2(np.where(present[i], preds[k][i], np.nan)) for k in preds},
               "lo": {k: r2(np.where(present[i], lo[k][i], np.nan)) for k in lo},
               "hi": {k: r2(np.where(present[i], hi[k][i], np.nan)) for k in hi}}
        tank_rows.append(row)

    # Campus total: matched-pair sums over the cells where the actual exists, so the actual and
    # every prediction are summed over exactly the same tanks in exactly the same hours. An hour
    # with no tank reporting is a hole, not a zero — the campus-wide outage on 2026-03-25 would
    # otherwise draw as eleven hours of zero demand that every model "over-forecast".
    dead = ~present.any(axis=0)

    def total(m):
        s = np.where(present, m, 0.0).sum(axis=0)
        return np.where(dead, np.nan, s)

    campus = {"id": "__campus__", "label": f"Campus total ({n_t} tanks)",
              "actual": r2(total(actual)), "cov": [int(v) for v in present.sum(axis=0)],
              "pred": {k: r2(total(v)) for k, v in preds.items()},
              "lo": {k: r2(total(v)) for k, v in lo.items()},
              "hi": {k: r2(total(v)) for k, v in hi.items()}}

    day0 = hours[0].normalize()
    days = [{"iso": str((day0 + pd.Timedelta(days=d)).date()),
             "label": (day0 + pd.Timedelta(days=d)).strftime("%a %-d %B"),
             "short": (day0 + pd.Timedelta(days=d)).strftime("%-d %b"),
             "dow": (day0 + pd.Timedelta(days=d)).strftime("%a")}
            for d in range(n_h // 24)]

    return {
        "start": hours[0].isoformat(),
        "n_hours": n_h,
        "n_days": n_h // 24,
        "n_tanks": n_t,
        "models": [{k: m[k] for k in ("key", "label", "note")} for m in MODELS],
        "days": days,
        "campus": campus,
        "tanks": tank_rows,
    }


CSS = """
:root{
  color-scheme: light;
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --hairline:rgba(11,11,11,.10); --wash:rgba(11,11,11,.04);
  --actual:#0b0b0b;
  --s-c2:#2a78d6; --s-npts:#eb6834; --s-patchtst:#1baf7a; --s-timesfm:#b0177f;
  --s-timestral:#5000d0;
  --good:#0ca30c; --warn:#fab219;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  color-scheme: dark;
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --hairline:rgba(255,255,255,.10); --wash:rgba(255,255,255,.05);
  --actual:#f2f1ec;
  --s-c2:#3987e5; --s-npts:#d95926; --s-patchtst:#199e70; --s-timesfm:#d65bb1;
  --s-timestral:#7850b8;
}}
:root[data-theme="dark"]{
  color-scheme: dark;
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --hairline:rgba(255,255,255,.10); --wash:rgba(255,255,255,.05);
  --actual:#f2f1ec;
  --s-c2:#3987e5; --s-npts:#d95926; --s-patchtst:#199e70; --s-timesfm:#d65bb1;
  --s-timestral:#7850b8;
}

*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--plane);color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 64px}

header h1{font-size:23px;line-height:1.25;margin:0 0 6px;letter-spacing:-.01em}
header p{margin:0;color:var(--ink-2);max-width:74ch}
header .sub{color:var(--muted);font-size:12.5px;margin-top:8px}

.card{background:var(--surface);border:1px solid var(--hairline);border-radius:12px}

/* ── filters: one row, above everything they scope ─────────────────────── */
.filters{position:sticky;top:0;z-index:30;margin:22px 0 0;padding:12px 14px;
  background:var(--surface);border:1px solid var(--hairline);border-radius:12px 12px 0 0;
  display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center}
.filters .grp{display:flex;align-items:center;gap:8px;min-height:32px}
.filters label.lab{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
select{font:inherit;font-size:13px;color:var(--ink);background:var(--surface);
  border:1px solid var(--axis);border-radius:7px;padding:5px 9px;max-width:260px}
.seg{display:inline-flex;border:1px solid var(--axis);border-radius:7px;overflow:hidden}
.seg button{font:inherit;font-size:12.5px;padding:5px 11px;border:0;background:transparent;
  color:var(--ink-2);cursor:pointer;border-right:1px solid var(--axis)}
.seg button:last-child{border-right:0}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--surface)}
.seg button:hover:not([aria-pressed="true"]){background:var(--wash)}
.step{font:inherit;font-size:13px;border:1px solid var(--axis);background:var(--surface);
  color:var(--ink);border-radius:7px;width:30px;height:30px;cursor:pointer;line-height:1}
.step:hover{background:var(--wash)} .step:disabled{opacity:.35;cursor:default}
input[type=range]{accent-color:var(--s-c2);width:190px}

/* ── series toggles: the chip carries the series colour ─────────────────── */
.chips{display:flex;flex-wrap:wrap;gap:8px;padding:0 14px 12px;background:var(--surface);
  border-left:1px solid var(--hairline);border-right:1px solid var(--hairline)}
.chip{display:inline-flex;align-items:center;gap:8px;font:inherit;font-size:13px;
  padding:5px 12px 5px 10px;border:1px solid var(--axis);border-radius:999px;cursor:pointer;
  background:var(--surface);color:var(--ink-2)}
.chip:hover{background:var(--wash)}
.chip .key{width:16px;height:2px;border-radius:2px;background:currentColor;flex:none}
.chip[aria-pressed="true"]{color:var(--ink);border-color:var(--hairline);box-shadow:inset 0 0 0 1px var(--hairline)}
.chip[aria-pressed="false"] .key{opacity:.35}
.chip[aria-pressed="false"]{opacity:.62;text-decoration:line-through;text-decoration-thickness:1px}
.chip.c2 .key{background:var(--s-c2)} .chip.npts .key{background:var(--s-npts)}
.chip.patchtst .key{background:var(--s-patchtst)} .chip.timesfm .key{background:var(--s-timesfm)}
.chip.timestral .key{background:var(--s-timestral)}
.chip.actual .key{background:var(--actual);height:3px}
.chip.band .key{background:linear-gradient(var(--muted),var(--muted));height:9px;opacity:.35;border-radius:2px}

/* ── stat tiles ─────────────────────────────────────────────────────────── */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:1px;
  background:var(--hairline);border:1px solid var(--hairline);border-top:0}
.tile{background:var(--surface);padding:11px 14px}
.tile .k{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.tile .v{font-size:20px;margin-top:3px;letter-spacing:-.01em}
.tile .v .u{font-size:12px;color:var(--ink-2);margin-left:3px;letter-spacing:0}
.tile .d{font-size:11.5px;color:var(--ink-2);margin-top:1px}
.tile .sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;
  vertical-align:baseline;transform:translateY(1px)}

/* ── chart ──────────────────────────────────────────────────────────────── */
.plot{background:var(--surface);border:1px solid var(--hairline);border-top:0;
  border-radius:0 0 12px 12px;padding:8px 4px 4px;position:relative}
svg{display:block;width:100%;height:auto;touch-action:none}
svg:focus{outline:none} svg:focus-visible{outline:2px solid var(--s-c2);outline-offset:-2px}
.gridline{stroke:var(--grid);stroke-width:1;shape-rendering:crispEdges}
.axisline{stroke:var(--axis);stroke-width:1;shape-rendering:crispEdges}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.axtitle{fill:var(--muted);font-size:11px;letter-spacing:.05em;text-transform:uppercase}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.line.actual{stroke:var(--actual);stroke-width:2.5}
.band{stroke:none;opacity:.13}
.gap{fill:var(--wash)}
.gaplabel{fill:var(--muted);font-size:10px}
.hair{stroke:var(--muted);stroke-width:1;shape-rendering:crispEdges;pointer-events:none}
.dot{stroke:var(--surface);stroke-width:2}
.endlab{font-size:11.5px;font-weight:600;paint-order:stroke;stroke:var(--surface);
  stroke-width:3.5px;stroke-linejoin:round}

.tip{position:absolute;pointer-events:none;z-index:20;min-width:196px;
  background:var(--surface);border:1px solid var(--hairline);border-radius:9px;
  box-shadow:0 6px 22px rgba(0,0,0,.14);padding:9px 11px;font-size:12.5px;opacity:0;
  transition:opacity .09s}
.tip.on{opacity:1}
.tip h4{margin:0 0 7px;font-size:12.5px;font-weight:600;letter-spacing:-.005em}
.tip .cov{font-size:11px;color:var(--warn);margin:-4px 0 6px}
.tip table{border-collapse:collapse;width:100%}
.tip td{padding:2px 0;vertical-align:baseline}
.tip td.k{width:15px} .tip td.k i{display:block;width:13px;height:2px;border-radius:2px}
.tip td.n{color:var(--ink-2);padding-right:10px;white-space:nowrap}
.tip td.v{text-align:right;font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap}
.tip td.e{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted);
  padding-left:9px;white-space:nowrap;font-size:11.5px}
.tip tr.act td.v{font-size:13.5px}
.hint{color:var(--muted);font-size:11.5px;margin:9px 2px 0}

/* ── table view (the relief for low-contrast series and for no-hover use) ── */
details.tbl{margin-top:20px}
details.tbl summary{cursor:pointer;color:var(--ink-2);font-size:13px;padding:6px 0}
details.tbl summary:hover{color:var(--ink)}
.tblwrap{overflow-x:auto;border:1px solid var(--hairline);border-radius:10px;
  background:var(--surface);max-height:430px;overflow-y:auto}
table.data{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
table.data th,table.data td{padding:5px 11px;text-align:right;white-space:nowrap;
  border-bottom:1px solid var(--grid)}
table.data th{position:sticky;top:0;background:var(--surface);color:var(--muted);font-weight:500;
  font-size:11px;letter-spacing:.05em;text-transform:uppercase;z-index:1;
  box-shadow:inset 0 -1px 0 var(--hairline)}
table.data th:first-child,table.data td:first-child{text-align:left}
table.data td.miss{color:var(--muted)}

footer{margin-top:26px;color:var(--muted);font-size:12px;line-height:1.65}
footer code{font-size:11.5px;background:var(--wash);padding:1px 5px;border-radius:4px}
@media (max-width:640px){ .filters{position:static} .wrap{padding:20px 12px 48px} }
"""


JS = r"""
const D = window.__HOLDOUT45__;
const MK = D.models.map(m => m.key);
const EXPECTED = {};              // cells expected per point: 24 tanks for campus, 1 for a tank
const ENT = {};
ENT[D.campus.id] = D.campus; EXPECTED[D.campus.id] = D.n_tanks;
D.tanks.forEach(t => { ENT[t.id] = t; EXPECTED[t.id] = 1; });

const state = {
  tank: D.campus.id, gran: "daily", win: 7, day0: 0,
  on: Object.assign({actual: true}, ...MK.map(k => ({[k]: true}))),
  band: false, hover: null,
};

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const colorOf = k => css(k === "actual" ? "--actual" : "--s-" + k);
const labelOf = k => k === "actual" ? "Actual" : D.models.find(m => m.key === k).label;

function fmt(v, unit) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const a = Math.abs(v);
  const d = a >= 100 ? 0 : a >= 10 ? 1 : 2;
  return v.toLocaleString("en-IN", {minimumFractionDigits: d, maximumFractionDigits: d})
       + (unit ? " " + unit : "");
}
const signed = (v, d) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);

/* ── the visible slice ─────────────────────────────────────────────────────
   Daily points are matched-pair sums: the actual and every prediction are summed over
   exactly the hours whose actual exists, so a partial day never flatters one model. */
function view() {
  const e = ENT[state.tank], exp = EXPECTED[state.tank];
  if (state.gran === "hourly") {
    const a = state.day0 * 24, b = Math.min(D.n_hours, a + state.win * 24);
    const idx = []; for (let i = a; i < b; i++) idx.push(i);
    const pick = arr => idx.map(i => arr[i]);
    return {
      n: idx.length, gran: "hourly", unit: "kL/h", base: a,
      x: idx.map(i => {
        const d = D.days[Math.floor(i / 24)], h = i % 24;
        return {major: h === 0, label: String(h).padStart(2, "0") + ":00",
                majorLabel: d.short, full: d.label + ", " + String(h).padStart(2, "0") + ":00",
                iso: d.iso + " " + String(h).padStart(2, "0") + ":00", day: Math.floor(i / 24)};
      }),
      actual: pick(e.actual), cov: idx.map(i => e.cov[i] / exp),
      pred: Object.fromEntries(MK.map(k => [k, pick(e.pred[k])])),
      lo: Object.fromEntries(MK.map(k => [k, pick(e.lo[k])])),
      hi: Object.fromEntries(MK.map(k => [k, pick(e.hi[k])])),
    };
  }
  const n = D.n_days, actual = [], cov = [], gap = 24 * exp;
  const pred = {}, lo = {}, hi = {};
  MK.forEach(k => { pred[k] = []; lo[k] = []; hi[k] = []; });
  for (let d = 0; d < n; d++) {
    let sa = 0, sc = 0, seen = 0;
    const sp = {}, sl = {}, sh = {};
    MK.forEach(k => { sp[k] = 0; sl[k] = 0; sh[k] = 0; });
    for (let h = d * 24; h < d * 24 + 24; h++) {
      sc += e.cov[h];
      if (e.actual[h] === null) continue;
      seen++; sa += e.actual[h];
      MK.forEach(k => { sp[k] += e.pred[k][h] || 0; sl[k] += e.lo[k][h] || 0; sh[k] += e.hi[k][h] || 0; });
    }
    actual.push(seen ? sa : null); cov.push(sc / gap);
    MK.forEach(k => { pred[k].push(seen ? sp[k] : null);
                      lo[k].push(seen ? sl[k] : null); hi[k].push(seen ? sh[k] : null); });
  }
  return {n, gran: "daily", unit: "kL/day", base: 0, actual, cov, pred, lo, hi,
          x: D.days.map((d, i) => ({major: d.dow === "Mon", label: d.short, majorLabel: d.short,
                                    full: d.label, iso: d.iso, day: i}))};
}

/* Metrics use only points with (near-)complete coverage — a day the campus meters were down
   for 11 hours is not a day any model got wrong. */
function metrics(v) {
  const out = {};
  const ok = i => v.actual[i] !== null && v.cov[i] >= 0.95;
  let sa = 0, nDays = 0;
  for (let i = 0; i < v.n; i++) if (ok(i)) { sa += v.actual[i]; nDays++; }
  out.__actual__ = {total: sa, mean: nDays ? sa / nDays : null, n: nDays, dropped: v.n - nDays};
  MK.forEach(k => {
    let ae = 0, ape = 0, sp = 0, m = 0;
    for (let i = 0; i < v.n; i++) {
      if (!ok(i) || v.pred[k][i] === null) continue;
      const err = v.pred[k][i] - v.actual[i];
      ae += Math.abs(err); sp += v.pred[k][i]; m++;
      if (v.actual[i] > 0) ape += Math.abs(err) / v.actual[i];
    }
    out[k] = m ? {mae: ae / m, mape: 100 * ape / m, bias: sa ? 100 * (sp - sa) / sa : null,
                  n: m, wins: 0}
               : null;
  });
  // Head-to-head over the models currently on screen: who landed closest, point by point.
  const live = MK.filter(k => state.on[k] && out[k]);
  if (live.length > 1) {
    out.__wins__ = 0;
    for (let i = 0; i < v.n; i++) {
      if (!ok(i)) continue;
      let best = null, bd = Infinity;
      live.forEach(k => {
        if (v.pred[k][i] === null) return;
        const d = Math.abs(v.pred[k][i] - v.actual[i]);
        if (d < bd) { bd = d; best = k; }
      });
      if (best) { out[best].wins++; out.__wins__++; }
    }
  }
  return out;
}

/* ── chart ─────────────────────────────────────────────────────────────── */
const NS = "http://www.w3.org/2000/svg";
const el = (t, a) => { const n = document.createElementNS(NS, t);
  for (const k in a) if (a[k] !== null && a[k] !== undefined) n.setAttribute(k, a[k]); return n; };
const txt = (t, a) => { const n = el("text", a); n.textContent = t; return n; };

const M = {t: 24, r: 96, b: 36, l: 58};
const VW = 1160, VH = 430;
let CUR = null;                                   // the rendered frame, reused by the hover layer

function niceTicks(max) {
  if (!(max > 0)) return {max: 1, ticks: [0, 0.5, 1]};
  const raw = max / 4, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * p).find(s => s >= raw) || 10 * p;
  const top = Math.ceil(max / step) * step, ticks = [];
  for (let t = 0; t <= top + 1e-9; t += step) ticks.push(t);
  return {max: top, ticks};
}

function draw() {
  const v = view(), svg = document.getElementById("chart");
  svg.textContent = "";
  const W = VW - M.l - M.r, H = VH - M.t - M.b;
  const active = MK.filter(k => state.on[k]);

  let ymax = 0;
  for (let i = 0; i < v.n; i++) {
    if (state.on.actual && v.actual[i] !== null) ymax = Math.max(ymax, v.actual[i]);
    active.forEach(k => {
      if (v.pred[k][i] !== null) ymax = Math.max(ymax, v.pred[k][i]);
      if (state.band && v.hi[k][i] !== null) ymax = Math.max(ymax, v.hi[k][i]);
    });
  }
  const {max: ytop, ticks} = niceTicks(ymax * 1.04);
  const step = ticks.length > 1 ? ticks[1] - ticks[0] : ytop;
  const td = step >= 10 ? 0 : step >= 1 ? 1 : 2;
  const tickFmt = t => t.toLocaleString("en-IN", {minimumFractionDigits: td,
                                                  maximumFractionDigits: td});
  const X = i => M.l + (v.n === 1 ? W / 2 : (i * W) / (v.n - 1));
  const Y = y => M.t + H - (y / ytop) * H;
  CUR = {v, X, Y, W, H, active, ytop};

  // no-data shading first, so every mark sits on top of it
  let run = null;
  for (let i = 0; i <= v.n; i++) {
    const dead = i < v.n && v.actual[i] === null;
    if (dead && run === null) run = i;
    if (!dead && run !== null) {
      const x0 = X(Math.max(0, run - .5)), x1 = X(Math.min(v.n - 1, i - .5));
      svg.appendChild(el("rect", {class: "gap", x: x0, y: M.t, width: Math.max(2, x1 - x0), height: H}));
      if (x1 - x0 > 46) svg.appendChild(txt("no data", {x: (x0 + x1) / 2, y: M.t + 13,
        class: "gaplabel", "text-anchor": "middle"}));
      run = null;
    }
  }

  ticks.forEach(t => {
    svg.appendChild(el("line", {class: "gridline", x1: M.l, x2: M.l + W, y1: Y(t), y2: Y(t)}));
    svg.appendChild(txt(tickFmt(t), {x: M.l - 9, y: Y(t) + 4, class: "tick", "text-anchor": "end"}));
  });
  svg.appendChild(txt(v.unit, {x: M.l - 9, y: M.t - 6, class: "axtitle", "text-anchor": "end"}));
  svg.appendChild(el("line", {class: "axisline", x1: M.l, x2: M.l + W, y1: Y(0), y2: Y(0)}));

  // x ticks: every day boundary in hourly view, a readable subset of days in daily view
  const every = v.gran === "hourly" ? 1 : Math.ceil(v.n / 12);
  v.x.forEach((p, i) => {
    const show = v.gran === "hourly" ? (p.major && (state.win <= 3 || i % (24 * Math.ceil(state.win / 8)) === 0))
                                     : i % every === 0;
    if (!show) return;
    svg.appendChild(el("line", {class: "axisline", x1: X(i), x2: X(i), y1: Y(0), y2: Y(0) + 4}));
    svg.appendChild(txt(v.gran === "hourly" ? p.majorLabel : p.label,
      {x: X(i), y: Y(0) + 17, class: "tick", "text-anchor": "middle"}));
  });
  if (v.gran === "hourly" && state.win <= 3) {
    v.x.forEach((p, i) => {
      if (i % 6 !== 0 || p.major) return;
      svg.appendChild(txt(p.label.slice(0, 2), {x: X(i), y: Y(0) + 17, class: "tick",
        "text-anchor": "middle", opacity: .75}));
    });
  }

  const seg = (arr, f) => {                       // split on nulls so gaps stay gaps
    const out = []; let cur = [];
    arr.forEach((y, i) => {
      if (y === null) { if (cur.length) out.push(cur); cur = []; } else cur.push([X(i), f(y)]);
    });
    if (cur.length) out.push(cur); return out;
  };
  const dpath = pts => pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");

  if (state.band) active.forEach(k => {
    const runs = [];
    let cur = [];
    for (let i = 0; i < v.n; i++) {
      if (v.lo[k][i] === null) { if (cur.length) runs.push(cur); cur = []; }
      else cur.push(i);
    }
    if (cur.length) runs.push(cur);
    runs.forEach(idx => {
      const up = idx.map(i => [X(i), Y(v.hi[k][i])]);
      const dn = idx.slice().reverse().map(i => [X(i), Y(v.lo[k][i])]);
      svg.appendChild(el("path", {class: "band", fill: colorOf(k),
        d: dpath(up) + " " + dn.map(p => "L" + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ") + " Z"}));
    });
  });

  active.forEach(k => seg(v.pred[k], Y).forEach(s =>
    svg.appendChild(el("path", {class: "line", stroke: colorOf(k), d: dpath(s)}))));
  if (state.on.actual) seg(v.actual, Y).forEach(s =>
    svg.appendChild(el("path", {class: "line actual", d: dpath(s)})));

  // direct end labels — identity without relying on colour alone
  const ends = [];
  (state.on.actual ? ["actual"] : []).concat(active).forEach(k => {
    const arr = k === "actual" ? v.actual : v.pred[k];
    for (let i = v.n - 1; i >= 0; i--) if (arr[i] !== null) { ends.push({k, y: Y(arr[i]), x: X(i)}); break; }
  });
  ends.sort((a, b) => a.y - b.y);
  ends.forEach(e => { e.ly = e.y + 4; });
  for (let i = 1; i < ends.length; i++)
    if (ends[i].ly - ends[i - 1].ly < 14) ends[i].ly = ends[i - 1].ly + 14;
  if (ends.length) {                              // slide the whole stack back inside the plot
    const over = ends[ends.length - 1].ly - (M.t + H);
    if (over > 0) ends.forEach(e => { e.ly -= over; });
    const under = (M.t + 10) - ends[0].ly;
    if (under > 0) ends.forEach(e => { e.ly += under; });
  }
  ends.forEach(e => svg.appendChild(txt(labelOf(e.k), {x: M.l + W + 8, y: e.ly,
    class: "endlab", fill: colorOf(e.k)})));

  svg.appendChild(el("g", {id: "hoverlayer"}));
  svg.appendChild(el("rect", {id: "hit", x: M.l, y: M.t, width: W, height: H, fill: "transparent"}));
  if (state.hover !== null && state.hover < v.n) paintHover(state.hover); else clearHover();
  renderTiles(v);
  renderTable(v);
}

function nearest(evt) {
  const svg = document.getElementById("chart"), r = svg.getBoundingClientRect();
  const px = ((evt.clientX - r.left) / r.width) * VW;
  const {v, W} = CUR;
  return Math.max(0, Math.min(v.n - 1, Math.round(((px - M.l) / W) * (v.n - 1))));
}

function clearHover() {
  const g = document.getElementById("hoverlayer"); if (g) g.textContent = "";
  document.getElementById("tip").classList.remove("on");
}

function paintHover(i) {
  const {v, X, Y, active} = CUR, g = document.getElementById("hoverlayer");
  g.textContent = "";
  g.appendChild(el("line", {class: "hair", x1: X(i), x2: X(i), y1: M.t, y2: M.t + CUR.H}));
  const rows = [];
  if (state.on.actual && v.actual[i] !== null) rows.push(["actual", v.actual[i], null]);
  active.forEach(k => { if (v.pred[k][i] !== null)
    rows.push([k, v.pred[k][i], v.actual[i] === null ? null : v.pred[k][i] - v.actual[i]]); });
  rows.forEach(([k, val]) => g.appendChild(el("circle", {class: "dot", cx: X(i), cy: Y(val),
    r: k === "actual" ? 4.5 : 4, fill: colorOf(k)})));

  const tip = document.getElementById("tip"), p = v.x[i];
  tip.textContent = "";
  const h = document.createElement("h4"); h.textContent = p.full; tip.appendChild(h);
  if (v.cov[i] < 0.999) {
    const c = document.createElement("div"); c.className = "cov";
    c.textContent = v.cov[i] === 0 ? "meters down — no reading"
      : Math.round(v.cov[i] * 100) + "% of readings present";
    tip.appendChild(c);
  }
  const t = document.createElement("table"), tb = document.createElement("tbody");
  rows.forEach(([k, val, err]) => {
    const tr = document.createElement("tr");
    if (k === "actual") tr.className = "act";
    const kd = document.createElement("td"); kd.className = "k";
    const key = document.createElement("i"); key.style.background = colorOf(k);
    if (k === "actual") key.style.height = "3px";
    kd.appendChild(key);
    const nd = document.createElement("td"); nd.className = "n"; nd.textContent = labelOf(k);
    const vd = document.createElement("td"); vd.className = "v"; vd.textContent = fmt(val, "");
    const ed = document.createElement("td"); ed.className = "e";
    ed.textContent = err === null ? "" : signed(err, Math.abs(err) >= 10 ? 1 : 2);
    tr.append(kd, nd, vd, ed); tb.appendChild(tr);
  });
  t.appendChild(tb); tip.appendChild(t);
  const u = document.createElement("div"); u.className = "cov";
  u.style.color = "var(--muted)"; u.style.margin = rows.length ? "6px 0 0" : "0";
  u.textContent = rows.length
    ? v.unit + (rows.length > 1 && v.actual[i] !== null ? " · right column = error" : "")
    : "nothing to compare here";
  tip.appendChild(u);

  const svg = document.getElementById("chart"), r = svg.getBoundingClientRect();
  const scale = r.width / VW, cx = X(i) * scale;
  tip.classList.add("on");
  const w = tip.offsetWidth;
  tip.style.left = Math.max(4, Math.min(r.width - w - 4, cx + (cx > r.width * 0.62 ? -w - 16 : 16))) + "px";
  tip.style.top = Math.max(6, Math.min(r.height - tip.offsetHeight - 6, M.t * scale + 6)) + "px";
}

/* ── tiles, table, controls ────────────────────────────────────────────── */
function renderTiles(v) {
  const m = metrics(v), box = document.getElementById("tiles");
  box.textContent = "";
  const tile = (k, vhtml, unit, d, sw) => {
    const n = document.createElement("div"); n.className = "tile";
    const a = document.createElement("div"); a.className = "k";
    if (sw) { const s = document.createElement("span"); s.className = "sw"; s.style.background = sw;
              a.appendChild(s); }
    a.appendChild(document.createTextNode(k));
    const b = document.createElement("div"); b.className = "v"; b.textContent = vhtml;
    if (unit) { const u = document.createElement("span"); u.className = "u"; u.textContent = unit;
                b.appendChild(u); }
    const c = document.createElement("div"); c.className = "d"; c.textContent = d;
    n.append(a, b, c); box.appendChild(n);
  };
  const A = m.__actual__;
  tile("Actual demand", fmt(A.mean, ""), v.unit,
       A.n + (v.gran === "daily" ? " full days" : " hours") +
       (A.dropped ? " · " + A.dropped + " excluded" : "") + " · " + fmt(A.total, "kL") + " total");
  MK.forEach(k => {
    if (!state.on[k]) return;
    const s = m[k];
    const d = s ? "MAPE " + s.mape.toFixed(1) + "% · bias " + signed(s.bias, 1) + "%"
                  + (m.__wins__ ? " · closest " + s.wins + "/" + m.__wins__ : "")
                : "no scored points";
    tile(labelOf(k), s ? fmt(s.mae, "") : "—", s ? "MAE " + v.unit : "", d, colorOf(k));
  });
}

function renderTable(v) {
  const det = document.getElementById("tv");
  document.getElementById("tblcount").textContent =
    v.n + (v.gran === "daily" ? " days" : " hours");
  if (!det.open) return;
  const head = document.getElementById("thead"), body = document.getElementById("tbody");
  const active = MK.filter(k => state.on[k]);
  head.textContent = ""; body.textContent = "";
  const hr = document.createElement("tr");
  [v.gran === "daily" ? "Day" : "Hour", "Actual"].concat(
    active.flatMap(k => [labelOf(k), "err"])).concat(["Coverage"]).forEach(t => {
      const th = document.createElement("th"); th.textContent = t; hr.appendChild(th); });
  head.appendChild(hr);
  const frag = document.createDocumentFragment();
  for (let i = 0; i < v.n; i++) {
    const tr = document.createElement("tr");
    const c0 = document.createElement("td"); c0.textContent = v.x[i].full; tr.appendChild(c0);
    const ca = document.createElement("td");
    ca.textContent = fmt(v.actual[i], ""); if (v.actual[i] === null) ca.className = "miss";
    tr.appendChild(ca);
    active.forEach(k => {
      const cv = document.createElement("td"); cv.textContent = fmt(v.pred[k][i], "");
      const ce = document.createElement("td"); ce.className = "miss";
      ce.textContent = (v.pred[k][i] === null || v.actual[i] === null) ? "—"
        : signed(v.pred[k][i] - v.actual[i], 2);
      tr.append(cv, ce);
    });
    const cc = document.createElement("td"); cc.className = v.cov[i] >= 0.95 ? "" : "miss";
    cc.textContent = Math.round(v.cov[i] * 100) + "%";
    tr.appendChild(cc); frag.appendChild(tr);
  }
  body.appendChild(frag);
}

function syncControls() {
  document.getElementById("tank").value = state.tank;
  document.querySelectorAll("[data-gran]").forEach(b =>
    b.setAttribute("aria-pressed", String(b.dataset.gran === state.gran)));
  document.querySelectorAll("[data-win]").forEach(b =>
    b.setAttribute("aria-pressed", String(+b.dataset.win === state.win)));
  document.querySelectorAll(".chip[data-series]").forEach(b =>
    b.setAttribute("aria-pressed", String(!!state.on[b.dataset.series])));
  document.getElementById("bandchip").setAttribute("aria-pressed", String(state.band));
  const hourly = state.gran === "hourly";
  document.getElementById("winbox").hidden = !hourly;
  document.getElementById("navbox").hidden = !hourly || state.win >= D.n_days;
  const maxDay = Math.max(0, D.n_days - state.win);
  state.day0 = Math.min(state.day0, maxDay);
  const sl = document.getElementById("scrub");
  sl.max = maxDay; sl.value = state.day0;
  document.getElementById("prev").disabled = state.day0 <= 0;
  document.getElementById("next").disabled = state.day0 >= maxDay;
  const a = D.days[state.day0], b = D.days[Math.min(D.n_days - 1, state.day0 + state.win - 1)];
  document.getElementById("range").textContent =
    state.win === 1 ? a.label : a.short + " – " + b.short;
}

const render = () => { syncControls(); draw(); };

function init() {
  const sel = document.getElementById("tank");
  [D.campus].concat(D.tanks).forEach(t => {
    const o = document.createElement("option"); o.value = t.id; o.textContent = t.label;
    sel.appendChild(o);
  });
  sel.value = state.tank;
  sel.addEventListener("change", () => { state.tank = sel.value; state.hover = null; render(); });

  document.querySelectorAll("[data-gran]").forEach(b => b.addEventListener("click", () => {
    state.gran = b.dataset.gran; state.hover = null; render(); }));
  document.querySelectorAll("[data-win]").forEach(b => b.addEventListener("click", () => {
    state.win = +b.dataset.win; state.hover = null; render(); }));
  document.querySelectorAll(".chip[data-series]").forEach(b => b.addEventListener("click", () => {
    const k = b.dataset.series;
    if (state.on[k] && Object.keys(state.on).filter(x => state.on[x]).length === 1) return;
    state.on[k] = !state.on[k]; render(); }));
  document.getElementById("bandchip").addEventListener("click", () => {
    state.band = !state.band; render(); });
  document.getElementById("prev").addEventListener("click", () => {
    state.day0 = Math.max(0, state.day0 - state.win); render(); });
  document.getElementById("next").addEventListener("click", () => {
    state.day0 = Math.min(D.n_days - state.win, state.day0 + state.win); render(); });
  document.getElementById("scrub").addEventListener("input", e => {
    state.day0 = +e.target.value; render(); });

  const svg = document.getElementById("chart");
  svg.addEventListener("pointermove", e => { state.hover = nearest(e); paintHover(state.hover); });
  svg.addEventListener("pointerleave", () => { state.hover = null; clearHover(); });
  svg.addEventListener("pointerdown", e => {          // click a day to open its 24 hours
    const i = nearest(e);
    if (state.gran === "daily") { state.gran = "hourly"; state.win = 1; state.day0 = i;
                                  state.hover = null; render(); }
  });
  svg.addEventListener("focus", () => { if (state.hover === null) { state.hover = 0; paintHover(0); } });
  svg.addEventListener("blur", () => { state.hover = null; clearHover(); });
  svg.addEventListener("keydown", e => {
    const n = CUR.v.n; let i = state.hover === null ? 0 : state.hover;
    if (e.key === "ArrowRight") i = Math.min(n - 1, i + 1);
    else if (e.key === "ArrowLeft") i = Math.max(0, i - 1);
    else if (e.key === "Home") i = 0;
    else if (e.key === "End") i = n - 1;
    else if (e.key === "Escape") { state.hover = null; clearHover(); return; }
    else return;
    e.preventDefault(); state.hover = i; paintHover(i);
  });
  document.getElementById("tv").addEventListener("toggle", e => {
    if (e.target.open) renderTable(view());
  });
  document.getElementById("theme").addEventListener("click", () => {
    const r = document.documentElement;
    const cur = r.getAttribute("data-theme")
      || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    r.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
    render();                                   // series colours are read from CSS at draw time
  });
  window.addEventListener("resize", () => { if (state.hover !== null) paintHover(state.hover); });
  render();
}
document.addEventListener("DOMContentLoaded", init);
"""


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>45-Day Holdout Explorer</title>
<style>__CSS__</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>45-day holdout explorer</h1>
  <p>Every hour of the reported holdout, actual against predicted. __NWORDCAP__ models forecast
     <strong>24&nbsp;hours ahead from a fresh origin each night</strong>, so the 45 days tile the
     window exactly once — no overlap, no gaps, and no model ever saw the day it is predicting.</p>
  <p class="sub">__WINDOW__ &middot; __NTANKS__ tanks &middot; __NHOURS__ hourly points per model
     &middot; point forecasts are raw (uncalibrated), so the __NWORD__ are directly comparable</p>
</header>

<div class="filters">
  <div class="grp">
    <label class="lab" for="tank">Tank</label>
    <select id="tank"></select>
  </div>
  <div class="grp">
    <span class="lab">Granularity</span>
    <span class="seg">
      <button type="button" data-gran="daily" aria-pressed="true">Daily total</button>
      <button type="button" data-gran="hourly" aria-pressed="false">Hourly</button>
    </span>
  </div>
  <div class="grp" id="winbox" hidden>
    <span class="lab">Window</span>
    <span class="seg">
      <button type="button" data-win="1">1 d</button>
      <button type="button" data-win="3">3 d</button>
      <button type="button" data-win="7">7 d</button>
      <button type="button" data-win="14">14 d</button>
      <button type="button" data-win="45">All</button>
    </span>
  </div>
  <div class="grp" id="navbox" hidden>
    <button type="button" class="step" id="prev" aria-label="Earlier">&#8592;</button>
    <input type="range" id="scrub" min="0" max="44" value="0" aria-label="First visible day">
    <button type="button" class="step" id="next" aria-label="Later">&#8594;</button>
    <span class="lab" id="range" style="text-transform:none;letter-spacing:0;font-size:12.5px"></span>
  </div>
  <button type="button" class="step" id="theme" style="margin-left:auto"
          aria-label="Switch between light and dark">&#9680;</button>
</div>

<div class="chips">
  <button type="button" class="chip actual" data-series="actual" aria-pressed="true">
    <span class="key"></span>Actual</button>
  __CHIPS__
  <button type="button" class="chip band" id="bandchip" aria-pressed="false">
    <span class="key"></span>80% interval</button>
</div>

<div class="tiles" id="tiles"></div>

<div class="plot">
  <svg id="chart" viewBox="0 0 1160 430" role="img" tabindex="0"
       aria-label="Actual against predicted demand over the 45-day holdout. Use the arrow keys to
                   step through points; values are also in the table below."></svg>
  <div class="tip" id="tip"></div>
</div>
<p class="hint">Hover or focus the chart and use &larr;/&rarr; to read any point.
   In the daily view, click a day to open its 24 hours.</p>

<details class="tbl" id="tv">
  <summary>Table view — every value on screen (<span id="tblcount"></span>)</summary>
  <div class="tblwrap">
    <table class="data"><thead id="thead"></thead><tbody id="tbody"></tbody></table>
  </div>
</details>

<footer>
  <p><strong>How to read it.</strong> Each point is a forecast made before the fact:
     the origin is 23:00 the night before, the horizon is the next 24 hours. Daily points are
     matched-pair sums — the actual and every prediction are summed over exactly the hours whose
     meter reading exists — so a partial day cannot flatter one model. Coverage below 100% is
     called out in the tooltip and in the table; the headline MAE, MAPE and bias skip points under
     95% coverage, which is why the campus outage on 25 March and the gap on 16 March do not count
     against anyone.</p>
  <p><strong>Provenance.</strong> Chronos-2&nbsp;ZS, NPTS, TimesFM&nbsp;2.5 and Timestral&nbsp;209M come from
     <code>results/chronos2/calibrated/predictions_calibrated.parquet</code>; PatchTST from
     <code>results/chronos2/holdout45_patchtst/predictions_holdout45_patchtst.parquet</code>.
     All __NWORD__ share the origin grid. The two fitted models (NPTS, PatchTST) share the fit
     cutoff (__CUTOFF__); the three foundation models (Chronos-2, TimesFM, Timestral) are
     zero-shot, with their context cut at each origin — so nothing leaks. TimesFM carries the last
     observed value across a sensor gap that reaches the origin, where Chronos-2 and Timestral mask
     it; that is why it overshoots the day after the 16 March outage.
     Rebuild with <code>python reports/build_holdout45_page.py</code>.</p>
</footer>

</div>
<script>window.__HOLDOUT45__ = __DATA__;</script>
<script>__JS__</script>
</body>
</html>
"""


def main() -> None:
    payload = build_payload(load())
    chips = "\n  ".join(
        f'<button type="button" class="chip {m["key"]}" data-series="{m["key"]}" '
        f'aria-pressed="true" title="{m["note"]}"><span class="key"></span>{m["label"]}</button>'
        for m in MODELS)
    first, last = payload["days"][0]["label"], payload["days"][-1]["label"]
    html = (HTML
            .replace("__CSS__", CSS)
            .replace("__JS__", JS)
            .replace("__CHIPS__", chips)
            .replace("__WINDOW__", f"{first} – {last} 2026 ({payload['n_days']} days)")
            .replace("__NTANKS__", str(payload["n_tanks"]))
            .replace("__NHOURS__", f"{payload['n_hours']:,}")
            .replace("__CUTOFF__", "2026-03-08 23:00")
            .replace("__NWORDCAP__", NWORD.capitalize())
            .replace("__NWORD__", NWORD)
            .replace("__DATA__", json.dumps(payload, separators=(",", ":"))))
    OUT.write_text(html)
    kb = OUT.stat().st_size / 1024
    print(f"{OUT}  ({kb:,.0f} KB)  {payload['n_days']} days x {payload['n_hours']} h "
          f"x {payload['n_tanks']} tanks x {len(MODELS)} models")


if __name__ == "__main__":
    main()
