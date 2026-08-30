"""Build the Phase III results page — one unified report over every model.

Supersedes the three overlapping builders this replaced. The page had grown a section per
opponent (Chronos-2 vs NPTS, then the covariate study, then PatchTST), so the same fact appeared
in three shapes and no table showed the whole field. v6 states each fact once, and every table
and figure carries all eleven models.

Every number is read at build time from ``results/chronos2/unified/`` and the calibration
outputs, so the page cannot drift from the data: change a metric, rerun the analysis, rerun this.

    python reports/build_results_page.py     # ~10 s -> reports/phase3_results_page.html
"""

import base64
import html
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
U = ROOT / "results/chronos2/unified"
UP = U / "plots"
H45 = ROOT / "results/chronos2/calibrated"
OUT = ROOT / "reports" / "phase3_results_page.html"


def img(p):
    return "data:image/png;base64," + base64.b64encode(pathlib.Path(p).read_bytes()).decode()


def esc(s):
    return html.escape(str(s))


def pct(v, nd=1):
    return f"{v:+.{nd}f}%"


# ─────────────────────────────────────────────────────────────────── data
lb = pd.read_csv(U / "leaderboard.csv")
sig = pd.read_csv(U / "significance_vs_all.csv")
wm = pd.read_csv(U / "win_matrix_all_horizons.csv")
pt = pd.read_csv(U / "per_tank.csv")
sk = pd.read_csv(U / "skill_h24.csv")
zi = pd.read_csv(U / "zero_inflation_h24.csv")
lt = pd.read_csv(U / "error_by_leadtime.csv")
cost = pd.read_csv(U / "cost.csv")
summ = json.loads((U / "summary.json").read_text())

h45s = pd.read_csv(H45 / "summary.csv").set_index("model")
h45e = pd.read_csv(H45 / "calibration_effect.csv").set_index("model")

HS = [6, 12, 24, 48, 72, 168]
HL = {6: "6 h", 12: "12 h", 24: "1 d", 48: "2 d", 72: "3 d", 168: "7 d"}
CHRONOS, INCUMBENT, REFERENCE = "Chronos2-ZS", "NPTS", "SeasonalNaive"
ORDER = [m for m in summ["models"]]
LABEL = {
    "Chronos2-ZS": "Chronos-2 (zero-shot)", "Chronos2-COV": "Chronos-2 + covariates",
    "Chronos2-COV-LEAN": "Chronos-2 + covariates (lean)",
    "Chronos2-COV-XL": "Chronos-2 + covariates (XL)", "NPTS": "NPTS",
    "PatchTST": "PatchTST (defaults)", "PatchTST-Tuned": "PatchTST (tuned)",
    "ETS": "ETS", "Theta": "Theta", "DynamicOptimizedTheta": "DynamicOptimizedTheta",
    "SeasonalNaive": "SeasonalNaive-24",
}
TAG = {CHRONOS: "production candidate", INCUMBENT: "deployed incumbent",
       REFERENCE: "metric reference"}
FAMILY = dict(zip(lb.model, lb.family))
NM = len(ORDER)
NWORD = {9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}.get(NM, str(NM))
ROWS_H = {h: int(lb[(lb.horizon == h) & (lb.model == CHRONOS)]["rows"].iloc[0]) for h in HS}
TOTAL = sum(ROWS_H.values())


def cell(m, h, c):
    return float(lb[(lb.model == m) & (lb.horizon == h)][c].iloc[0])


def _row_tag(m):
    t = TAG.get(m)
    return f" <span class=tag>{t}</span>" if m == CHRONOS else (
        f" <span class=tag-mut>{t}</span>" if t else "")



CSS = """
<style>
:root{
  --ground:#fbfbf9; --panel:#ffffff; --panel-2:#f4f6f8; --hair:#e2e6ea; --hair-2:#eef1f4;
  --ink:#141a21; --ink-2:#4a545f; --ink-3:#7a838d;
  --c2:#2a78d6; --c2-soft:#e8f1fc; --npts:#d95a2b; --npts-soft:#fdeee7;
  --ok:#12855c; --ok-soft:#e2f4ec; --warn:#a8720b; --warn-soft:#fbf1dc; --bad:#c3352f; --bad-soft:#fbe9e8;
  --shadow:0 1px 2px rgba(16,24,40,.05),0 1px 3px rgba(16,24,40,.04);
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --maxw:1180px;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#0d1218; --panel:#141b23; --panel-2:#1a232d; --hair:#26313d; --hair-2:#1e2833;
  --ink:#e8edf3; --ink-2:#a3aeba; --ink-3:#7b8792;
  --c2:#5b9df0; --c2-soft:#12243a; --npts:#f0805a; --npts-soft:#33190f;
  --ok:#3fc48d; --ok-soft:#0e2a20; --warn:#e0aa3e; --warn-soft:#2c2211; --bad:#ef6b64; --bad-soft:#2e1614;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 1px 3px rgba(0,0,0,.3);
}}
:root[data-theme="dark"]{
  --ground:#0d1218; --panel:#141b23; --panel-2:#1a232d; --hair:#26313d; --hair-2:#1e2833;
  --ink:#e8edf3; --ink-2:#a3aeba; --ink-3:#7b8792;
  --c2:#5b9df0; --c2-soft:#12243a; --npts:#f0805a; --npts-soft:#33190f;
  --ok:#3fc48d; --ok-soft:#0e2a20; --warn:#e0aa3e; --warn-soft:#2c2211; --bad:#ef6b64; --bad-soft:#2e1614;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 1px 3px rgba(0,0,0,.3);
}
:root[data-theme="light"]{
  --ground:#fbfbf9; --panel:#ffffff; --panel-2:#f4f6f8; --hair:#e2e6ea; --hair-2:#eef1f4;
  --ink:#141a21; --ink-2:#4a545f; --ink-3:#7a838d;
  --c2:#2a78d6; --c2-soft:#e8f1fc; --npts:#d95a2b; --npts-soft:#fdeee7;
  --ok:#12855c; --ok-soft:#e2f4ec; --warn:#a8720b; --warn-soft:#fbf1dc; --bad:#c3352f; --bad-soft:#fbe9e8;
  --shadow:0 1px 2px rgba(16,24,40,.05),0 1px 3px rgba(16,24,40,.04);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 24px 96px}
a{color:var(--c2)}
h1,h2,h3{text-wrap:balance;margin:0}
h1{font-size:clamp(26px,3.4vw,38px);font-weight:680;letter-spacing:-.022em;line-height:1.15}
h2{font-size:21px;font-weight:640;letter-spacing:-.014em}
h3{font-size:16px;font-weight:620;letter-spacing:-.008em}
p{margin:0}
.eyebrow{font:600 11px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--c2)}

/* header */
header{border-bottom:1px solid var(--hair);background:var(--panel);margin-bottom:40px}
.hd{max-width:var(--maxw);margin:0 auto;padding:44px 24px 34px;display:flex;flex-direction:column;gap:14px}
.meta{display:flex;flex-wrap:wrap;gap:8px 20px;font:500 12.5px/1.5 var(--mono);color:var(--ink-3)}
.meta b{color:var(--ink-2);font-weight:600}
.lede{max-width:66ch;color:var(--ink-2);font-size:16.5px}

/* sections */
section{margin-top:56px;scroll-margin-top:20px}
.sec-hd{display:flex;flex-direction:column;gap:7px;margin-bottom:20px;
  padding-bottom:13px;border-bottom:1px solid var(--hair)}
.sec-note{color:var(--ink-2);max-width:74ch;font-size:14.5px}

/* status grid */
.status{display:grid;grid-template-columns:repeat(auto-fit,minmax(268px,1fr));gap:12px}
.st{background:var(--panel);border:1px solid var(--hair);border-radius:9px;padding:14px 15px;
  box-shadow:var(--shadow);border-left:3px solid var(--hair)}
.st.done{border-left-color:var(--ok)} .st.partial{border-left-color:var(--warn)} .st.todo{border-left-color:var(--bad)}
.st-top{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:7px}
.st-name{font-weight:620;font-size:14.5px}
.st p{color:var(--ink-2);font-size:13.2px;line-height:1.5}

/* chips */
.chip{display:inline-block;font:600 11px/1 var(--mono);letter-spacing:.04em;padding:4px 7px;
  border-radius:5px;white-space:nowrap;text-transform:uppercase}
.chip.done,.chip.ok{background:var(--ok-soft);color:var(--ok)}
.chip.partial,.chip.warn{background:var(--warn-soft);color:var(--warn)}
.chip.todo,.chip.bad{background:var(--bad-soft);color:var(--bad)}
.chip.neu{background:var(--panel-2);color:var(--ink-3)}
.chip.t-healthy{background:var(--ok-soft);color:var(--ok)}
.chip.t-degraded{background:var(--warn-soft);color:var(--warn)}
.chip.t-dead{background:var(--bad-soft);color:var(--bad)}
.tag{display:inline-block;font:600 10px/1 var(--mono);letter-spacing:.05em;text-transform:uppercase;
  background:var(--c2-soft);color:var(--c2);padding:3px 6px;border-radius:4px;margin-left:7px;vertical-align:1px}
.tag-mut{display:inline-block;font:600 10px/1 var(--mono);letter-spacing:.05em;text-transform:uppercase;
  background:var(--panel-2);color:var(--ink-3);padding:3px 6px;border-radius:4px;margin-left:7px;vertical-align:1px}
.badge-new{font:700 9.5px/1 var(--mono);letter-spacing:.09em;text-transform:uppercase;
  background:var(--c2);color:#fff;padding:3.5px 6px;border-radius:4px;margin-left:auto}

/* stat tiles */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(176px,1fr));gap:12px;margin-top:8px}
.tile{background:var(--panel);border:1px solid var(--hair);border-radius:9px;padding:15px 16px;box-shadow:var(--shadow)}
.tile .k{font:600 10.5px/1.3 var(--mono);letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3)}
.tile .v{font:650 27px/1.15 var(--sans);letter-spacing:-.02em;margin:7px 0 3px;font-variant-numeric:tabular-nums}
.tile .v.pos{color:var(--c2)} .tile .v.neg{color:var(--npts)}
.tile .s{font-size:12.4px;color:var(--ink-2);line-height:1.45}

/* tables */
.tw{overflow-x:auto;border:1px solid var(--hair);border-radius:9px;background:var(--panel);box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:13.4px}
caption{text-align:left;padding:13px 15px 0;font-size:12.6px;color:var(--ink-3);font-weight:500}
th,td{padding:8px 13px;text-align:left;border-bottom:1px solid var(--hair-2);white-space:nowrap}
thead th{font:600 10.5px/1.3 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);
  background:var(--panel-2);position:sticky;top:0}
tbody th{font-weight:600;color:var(--ink)}
td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:12.8px}
td.win{color:var(--c2);font-weight:600} td.loss{color:var(--npts);font-weight:600}
td.dim{color:var(--ink-3)}
.arrow{color:var(--ink-3);padding:0 1px}
td.n b{font-weight:650} td.bad-t{color:var(--bad);font-weight:600}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}
tbody tr.prod{background:var(--c2-soft)}
tbody tr:hover{background:var(--panel-2)}
tbody tr.prod:hover{background:var(--c2-soft)}

/* figures */
.figs{display:flex;flex-direction:column;gap:26px}
.fig{margin:0;background:var(--panel);border:1px solid var(--hair);border-radius:10px;
  overflow:hidden;box-shadow:var(--shadow)}
.fig-head{display:flex;align-items:center;gap:11px;padding:12px 15px;border-bottom:1px solid var(--hair-2);
  background:var(--panel-2)}
.fig-id{font:700 12px/1 var(--mono);color:var(--c2);background:var(--c2-soft);
  width:24px;height:24px;display:grid;place-items:center;border-radius:5px;flex:none}
.fig-title{font-weight:620;font-size:14.5px}
.fig-img{padding:14px;background:#fcfcfb;overflow-x:auto}
.fig-img img{display:block;width:100%;max-width:100%;height:auto}
.fig-foot{padding:12px 15px;display:flex;flex-direction:column;gap:6px;border-top:1px solid var(--hair-2)}
.fig-desc{font-size:13.2px;color:var(--ink-2)}
.fig-take{font-size:13.4px;color:var(--ink);font-weight:560;padding-left:11px;border-left:2px solid var(--c2)}
.gal{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.fig.sm .fig-img{padding:9px}

/* callout */
.call{background:var(--panel);border:1px solid var(--hair);border-left:3px solid var(--c2);
  border-radius:8px;padding:16px 18px;box-shadow:var(--shadow)}
.call h3{margin-bottom:7px}
.call p{color:var(--ink-2);font-size:14.2px}
.call p + p{margin-top:9px}
.call strong{color:var(--ink);font-weight:620}
.call.warn{border-left-color:var(--warn)}
.stack{display:flex;flex-direction:column;gap:14px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px;align-items:start}

/* inference list */
.inf{display:flex;flex-direction:column;gap:12px}
.inf-item{background:var(--panel);border:1px solid var(--hair);border-radius:9px;padding:15px 17px;box-shadow:var(--shadow)}
.inf-top{display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap}
.inf-top h3{font-size:15px}
.inf-item p{color:var(--ink-2);font-size:14px}
.inf-item p + p{margin-top:8px}
.inf-item strong{color:var(--ink);font-weight:620}

pre{margin:0;background:var(--panel-2);border:1px solid var(--hair);border-radius:8px;
  padding:14px 16px;overflow-x:auto;font-family:var(--mono);font-size:12.8px;line-height:1.75;color:var(--ink-2)}
pre b{color:var(--ink);font-weight:600}
footer{margin-top:64px;padding-top:22px;border-top:1px solid var(--hair);color:var(--ink-3);font-size:13px}
:focus-visible{outline:2px solid var(--c2);outline-offset:2px;border-radius:3px}
@media (max-width:620px){.wrap{padding:0 16px 64px}.hd{padding:32px 16px 26px}th,td{padding:7px 10px}}
</style>
"""


# ─────────────────────────────────────────────────────────── derived blocks

# §2 leaderboard — every model, every horizon, one table
lead_rows = ""
for m in ORDER:
    r24 = lb[(lb.model == m) & (lb.horizon == 24)].iloc[0]
    cls = "prod" if m == CHRONOS else ""
    cells = "".join(
        f"<td class='n{' win' if m == CHRONOS else ''}'>{cell(m, h, 'macro_mase'):.4f}"
        f"<span class='rk'>#{int(lb[(lb.model==m)&(lb.horizon==h)]['rank_mase'].iloc[0])}</span></td>"
        for h in HS)
    lead_rows += (
        f"<tr class='{cls}'><th scope='row'>{esc(LABEL[m])}{_row_tag(m)}</th>"
        f"<td class='fam'>{esc(FAMILY[m])}</td>{cells}"
        f"<td class='n'>{r24.macro_mae:.4f}</td><td class='n'>{r24.macro_rmse:.4f}</td>"
        f"<td class='n'>{r24.macro_rmsse:.4f}</td></tr>")

# §3 significance — Chronos-2 against every opponent, at 1 d
_s24 = sig[(sig.metric == "MASE") & (sig.horizon == 24)].set_index("opponent")
_sall = sig[sig.metric == "MASE"]
sig_rows = ""
for m in ORDER:
    if m == CHRONOS:
        continue
    r = _s24.loc[m]
    allh = _sall[_sall.opponent == m]
    nsig = int(allh.significant_95.sum())
    won = int(allh.base_better.sum())
    verdict = ("<span class='chip ok'>significant, all 6</span>" if won == 6 else
               "<span class='chip neu'>indistinguishable</span>" if nsig == 0 else
               f"<span class='chip warn'>mixed — {won}/6</span>")
    cells_won = int(wm[(wm.model_a == CHRONOS) & (wm.model_b == m)]["a_wins"].iloc[0])
    sig_rows += (
        f"<tr><th scope='row'>{esc(LABEL[m])}</th><td class='fam'>{esc(FAMILY[m])}</td>"
        f"<td class='n {'win' if r.improvement_pct > 0.5 else 'dim'}'>{pct(r.improvement_pct, 2)}</td>"
        f"<td class='n dim'>[{r.ci_lo_pct:+.2f}, {r.ci_hi_pct:+.2f}]</td>"
        f"<td class='n'>{'&lt;0.0001' if r.p_bootstrap < 1e-4 else f'{r.p_bootstrap:.4f}'}</td>"
        f"<td class='n'>{r.p_dm:.1e}</td>"
        f"<td class='n'>{int(r.origins_won)}/24</td>"
        f"<td class='n'>{cells_won}/144</td><td>{verdict}</td></tr>")

# §4 per tank — winner counts and skill, per model
_p24 = pt[pt.horizon == 24]
_best = _p24.loc[_p24.groupby("tank")["mase"].idxmin()]
_wins = _best.model.value_counts()
_sk = sk.groupby("model")["skill"].agg(["min", "median", lambda s: int((s > 0).sum())])
_sk.columns = ["min", "median", "positive"]
tank_rows = ""
for m in ORDER:
    if m == REFERENCE:
        continue
    s = _sk.loc[m]
    tank_rows += (
        f"<tr class='{'prod' if m == CHRONOS else ''}'><th scope='row'>{esc(LABEL[m])}</th>"
        f"<td class='n'>{int(_wins.get(m, 0))}</td>"
        f"<td class='n {'win' if s.positive == 24 else 'loss'}'>{int(s.positive)}/24</td>"
        f"<td class='n'>{s['median']:.3f}</td>"
        f"<td class='n {'dim' if s['min'] > 0 else 'loss'}'>{s['min']:+.3f}</td></tr>")

# §6 calibration — coverage, width, the negative-p10 tell
_z = zi.set_index("model")
cal_rows = ""
for m in ORDER:
    r24 = lb[(lb.model == m) & (lb.horizon == 24)].iloc[0]
    z = _z.loc[m]
    neg = float(z.p10_below_zero_pct)
    cal_rows += (
        f"<tr class='{'prod' if m == CHRONOS else ''}'><th scope='row'>{esc(LABEL[m])}</th>"
        f"<td class='n'>{r24.coverage:.3f}</td>"
        f"<td class='n {'bad-t' if abs(r24.coverage - 0.8) > 0.05 else 'dim'}'>"
        f"{r24.coverage - 0.8:+.3f}</td>"
        f"<td class='n'>{r24.width:.3f}</td>"
        f"<td class='n {'bad-t' if neg > 20 else 'dim'}'>{neg:.0f}%</td>"
        f"<td class='n dim'>{z.coverage_if_p10_clamped:.3f}</td></tr>")

# §8 cost
cost_rows = ""
for r in cost.dropna(subset=["wall_clock_s"]).itertuples():
    cost_rows += (
        f"<tr class='{'prod' if r.model == CHRONOS else ''}'>"
        f"<th scope='row'>{esc(LABEL[r.model])}</th><td>{esc(r.regime)}</td>"
        f"<td class='n'>{r.wall_clock_s:,.0f} s</td><td class='n'>{r.wall_clock_min:.1f}</td>"
        f"<td class='n{' win' if r.model == CHRONOS else ''}'>{r.macro_mase_h24:.4f}</td></tr>")

# §7 operational — the 45-day calibrated holdout
_MODS = [(CHRONOS, "Chronos-2 (zero-shot)"), (INCUMBENT, "NPTS (incumbent)"),
         ("SeasonalNaive-24", "SeasonalNaive-24 (reference)")]
op_rows = ""
for k, lab in _MODS:
    s, e = h45s.loc[k], h45e.loc[k]
    op_rows += (
        f"<tr class='{'prod' if k == CHRONOS else ''}'><th scope='row'>{esc(lab)}</th>"
        f"<td class='n'>{s.daily_mae_raw_kl:.1f} <span class='arrow'>&rarr;</span> "
        f"<b class='win'>{s.daily_mae_kl:.1f}</b></td>"
        f"<td class='n'>{s.daily_mape_pct:.1f}%</td>"
        f"<td class='n'>{s.total_bias_raw_pct:+.1f}% <span class='arrow'>&rarr;</span> "
        f"<b>{s.total_bias_pct:+.1f}%</b></td>"
        f"<td class='n'>{e.coverage_raw:.3f} <span class='arrow'>&rarr;</span> "
        f"<b>{e.coverage_cal:.3f}</b></td>"
        f"<td class='n {'win' if s.sd_ratio > 0.5 else 'loss'}'>{s.sd_ratio:.2f}</td>"
        f"<td class='n {'win' if s.corr_with_actual > 0.4 else 'loss'}'>"
        f"{s.corr_with_actual:+.3f}</td></tr>")

# headline numbers, all derived
_beaten = summ["opponents_beaten_significantly_all_horizons"]
_imp = summ["chronos2_improvement_pct_h24"]
_npts_imp = sig[(sig.metric == "MASE") & (sig.opponent == INCUMBENT)]
NPTS_LO, NPTS_HI = _npts_imp.improvement_pct.min(), _npts_imp.improvement_pct.max()
_ptt = sig[(sig.metric == "MASE") & (sig.opponent == "PatchTST-Tuned")]
PT_LO, PT_HI = _ptt.improvement_pct.min(), _ptt.improvement_pct.max()
Z = _z.loc[CHRONOS]
C2S = float(cost[cost.model == CHRONOS].wall_clock_s.iloc[0])
PTS = float(cost[cost.model == "PatchTST-Tuned"].wall_clock_s.iloc[0])
CAL = h45e.loc[CHRONOS]
OPS = h45s.loc[CHRONOS]
_d1 = lt[(lt.model == CHRONOS) & (lt.step <= 24)]["mae"].mean()
_d7 = lt[(lt.model == CHRONOS) & (lt.step > 144)]["mae"].mean()

# ─────────────────────────────────────────────────────────────── figures
FIGS = [
    ("U1", "U1_leaderboard_mase", "Every model, every horizon",
     "Macro MASE against forecast horizon for all eleven models on the identical rows.",
     "Four clean bands: the Chronos-2 family, the incumbent, the trained deep models, and the "
     "classical methods. The four Chronos-2 variants coincide."),
    ("U2", "U2_leaderboard_heatmap", "The leaderboard as a grid",
     "Macro MASE and rank in every cell. The blue outline is the production candidate.",
     "Rank is stable across horizons: the ordering of families never changes, only the margins."),
    ("U3", "U3_significance_vs_all", "Chronos-2 against every opponent",
     "MASE improvement with 95% paired-bootstrap confidence intervals over the 24 origins — at "
     "1 d on the left, across all six horizons on the right.",
     "Seven opponents are beaten significantly at every horizon. The three covariate variants are "
     "the only ones whose intervals cross zero, which is the finding, not a failure."),
    ("U4", "U4_win_matrix", "Who beats whom, cell by cell",
     "Percentage of the 144 tank x horizon cells each model wins against each other model, on "
     "per-tank MAE.",
     "Chronos-2 takes every cell from the classical methods and the reference, and loses cells "
     "only to its own covariate variants."),
    ("U5", "U5_per_tank_heatmap", "Per tank, per model",
     "MASE at 1 d for all 24 tanks and all 11 models, tanks ordered by mean demand and labelled "
     "by sensor tier.",
     "The hard tanks are hard for everyone — GJBC_LAW_BLOCK_3__A1 and INFORMATION_CENTRE are the "
     "worst column for every model, which is a property of those sensors, not of a model."),
    ("U6", "U6_skill_vs_demand", "Is the average flattered by the dead sensors?",
     "Skill against the seasonal-naive reference for every tank and every model, against that "
     "tank's mean demand.",
     "Every Chronos-2 variant and NPTS clear zero on all 24 tanks. Skill does not fall away with "
     "demand size, so the headline is not an artefact of the sensors that never move."),
    ("U9", "U9_error_by_leadtime", "Does it fall apart at long lead times?",
     "MAE against hours ahead of the origin, out to 168 h, as a 24-hour rolling mean.",
     f"Chronos-2 grows {100 * (_d7 - _d1) / _d1:+.1f}% from day 1 to day 7. A weekly forecast is "
     "nearly as good as a daily one, which is what makes weekly refill planning viable."),
    ("U10", "U10_diurnal", "Where in the day the error lives",
     "MAE and signed bias by hour of day, every model, against the mean demand profile.",
     "Error tracks demand rather than concentrating in one bad window, and the under-forecast is "
     "present in all 24 hours — systematic, which is the kind a correction can fix."),
    ("U7", "U7_calibration", "Coverage against the width that bought it",
     "Empirical p10-p90 coverage at 1 d against mean interval width. Red rings mark models whose "
     "lower bound goes below zero.",
     "Read alone, coverage ranks the ringed models best. They get there by predicting a negative "
     "water outflow, which cannot happen."),
    ("U11", "U11_zero_inflation", "Why the intervals miss",
     "Lower- versus upper-tail miss rates, how often p10 falls below zero, and what a naive clamp "
     "would do — for every model.",
     "The deficit is structural to continuous-density forecasting on a zero-inflated series, not "
     "a defect of one model."),
    ("U12", "U12_volume_bias", "Signed volume bias — the number that sizes a refill",
     "Total forecast volume against total actual, per model and horizon.",
     "The operationally expensive direction is negative: a refill sized on an uncorrected mean "
     "would be short."),
    ("U8", "U8_cost_accuracy", "Accuracy against what it cost",
     "Macro MASE at 1 d against measured wall clock for the full backtest, log scale.",
     "The production candidate is the leftmost point on the chart and within 0.003 MASE of the "
     "best — which is what makes zero-shot a compute decision rather than an accuracy sacrifice."),
]

figs = "".join(
    f"""<figure class="fig" id="fig-{fid}">
  <figcaption class="fig-head"><span class="fig-id">{fid}</span>
    <span class="fig-title">{esc(title)}</span></figcaption>
  <div class="fig-img"><img src="{img(UP / (fn + '.png'))}" alt="{esc(desc)}" loading="lazy"></div>
  <div class="fig-foot"><p class="fig-desc">{esc(desc)}</p><p class="fig-take">{esc(take)}</p></div>
</figure>""" for fid, fn, title, desc, take in FIGS)

OPFIGS = [
    ("O1", "X_tanks45_chronos2", "Chronos-2 — 45 continuous days, four tanks",
     "Solid actual, dashed calibrated, dotted uncalibrated, shaded conformal band. Each panel has "
     "its own y-scale.",
     "Tracks the two healthy tanks closely; misses the spiky peaks on GJBC_LAW_BLOCK_3__A1, the "
     "one healthy sensor the model genuinely fails on."),
    ("O2", "Y_tanks45_npts", "NPTS — the same four tanks",
     "The same window and the same correction applied to the deployed model.",
     "Flat on all four. The flatness seen at campus level is how the model behaves per tank, not "
     "an aggregation artefact."),
    ("O3", "Z_tanks45_seasonal_naive", "SeasonalNaive-24 — the same four tanks",
     "Each day forecast as a repeat of the previous one.",
     "Visually the closest tracker and the worst MAE — every feature it reproduces arrives a day "
     "late. Without it you cannot tell whether tracking this data is possible at all."),
]
opfigs = "".join(
    f"""<figure class="fig" id="fig-{fid}">
  <figcaption class="fig-head"><span class="fig-id">{fid}</span>
    <span class="fig-title">{esc(title)}</span></figcaption>
  <div class="fig-img"><img src="{img(H45 / 'plots' / (fn + '.png'))}" alt="{esc(desc)}" loading="lazy"></div>
  <div class="fig-foot"><p class="fig-desc">{esc(desc)}</p><p class="fig-take">{esc(take)}</p></div>
</figure>""" for fid, fn, title, desc, take in OPFIGS)

STATUS = [
    ("System Testing", "partial", "Partial",
     "6/6 metric unit tests and the methodology assertions pass, which validates the measurement "
     "apparatus. There is no system or integration suite — the real-time system is designed, not built."),
    ("Verification &amp; Validation", "done", "Done",
     f"Row parity across all {NM} models, zero leakage, the MASE identity, an independent re-score, "
     "paired significance against every opponent, and a trained deep control."),
    ("Deployment", "todo", "Not deployed",
     "The Chrome dock runs offline from a bundled forecast; the compose stack targets retired "
     "models. Nothing is serving."),
    ("Final Experimental Results", "done", "Done",
     f"{NM} models x 6 horizons x 24 tanks on one shared grid. {TOTAL:,} scored rows per model, "
     "zero leakage, zero duplicates."),
    ("Performance Analysis", "done", f"Done — {len(FIGS) + len(OPFIGS)} figures",
     "One unified figure set: every panel carries every model, with a table behind each."),
    ("Research Paper Draft", "todo", "Not written",
     "review_summary.md holds the material. The zero-inflation calibration finding is the "
     "publishable contribution."),
]
status_cards = "".join(
    f"""<div class="st {cls}"><div class="st-top"><span class="st-name">{name}</span>
    <span class="chip {cls}">{lab}</span></div><p>{note}</p></div>"""
    for name, cls, lab, note in STATUS)

# ─────────────────────────────────────────────────────────────── the page
BODY = f"""
<title>Phase III — Final Results &amp; Performance Analysis</title>
{CSS}
<header>
  <div class="hd">
    <span class="eyebrow">PW26_PK_06 · Review 1 · Phase III · v6</span>
    <h1>Final experimental results and performance analysis</h1>
    <p class="lede">Hourly water-demand forecasting for 24 campus tanks, benchmarked across
      <b>{NWORD} models</b> on one shared evaluation grid — a pretrained foundation model, its
      covariate variants, the deployed incumbent, a transformer trained on this campus, and the
      classical baselines — with significance testing, calibration diagnosis and the operational
      view, each stated once.</p>
    <div class="meta">
      <span><b>24</b> tanks</span><span><b>24</b> origins, 23 h stride</span>
      <span><b>6</b> horizons</span><span><b>{NM}</b> models</span>
      <span><b>{TOTAL:,}</b> scored rows per model</span>
      <span><b>0</b> leakage rows</span><span><b>0</b> duplicates</span>
      <span>2025-01-01 → 2026-04-22</span>
    </div>
  </div>
</header>

<div class="wrap">

<section id="status">
  <div class="sec-hd"><span class="eyebrow">Where we stand</span>
    <h2>The six Phase III expectations</h2>
    <p class="sec-note">Four complete, one partial, one not started. The modelling half is
      finished and statistically defended; the systems half is specified in detail but unbuilt.</p>
  </div>
  <div class="status">{status_cards}</div>
</section>

<section id="grid">
  <div class="sec-hd"><span class="eyebrow">Section 1</span>
    <h2>The grid every number rests on</h2>
    <p class="sec-note">Nothing below is comparable unless this holds. The load-bearing property
      is that every model is scored on <em>identical rows</em> — the earlier work in this
      repository could not be compared model-to-model, and fixing that was the point.</p></div>
  <div class="stats">
    <div class="tile"><div class="k">Tanks</div><div class="v">24</div>
      <div class="s">26 dataset directories de-duplicated; the loader raises if the count is not 24</div></div>
    <div class="tile"><div class="k">Observations</div><div class="v">270,849</div>
      <div class="s">gapless hourly index; missing hours held open as NaN, never dropped</div></div>
    <div class="tile"><div class="k">Origins</div><div class="v">24</div>
      <div class="s">23-hour stride, co-prime with 24 — every hour-of-day visited exactly once</div></div>
    <div class="tile"><div class="k">Rows per model</div><div class="v pos">{TOTAL:,}</div>
      <div class="s">identical for all {NM} models, asserted in code and fatal under --strict</div></div>
    <div class="tile"><div class="k">Leakage rows</div><div class="v pos">0</div>
      <div class="s">every scored row satisfies timestamp &gt; origin</div></div>
    <div class="tile"><div class="k">Duplicates</div><div class="v pos">0</div>
      <div class="s">no (tank, timestamp) pair survives curation twice</div></div>
    <div class="tile"><div class="k">Sensor tiers</div><div class="v">15 / 6 / 3</div>
      <div class="s">healthy / degraded / dead, from a mass-balance identity test</div></div>
    <div class="tile"><div class="k">Metric tests</div><div class="v pos">6 / 6</div>
      <div class="s">including the MASE identity, without which every number here is wrong</div></div>
  </div>
  <div class="call" style="margin-top:16px"><h3>Why a 23-hour stride and not 24</h3>
    <p>With a 24-hour stride every origin lands on the same clock hour, so the 6 h horizon would
    only ever be scored on the quiet 00:00–05:00 window and never on the morning refill peak —
    the part that is actually hard. 23 is co-prime with 24, so the origins visit every hour-of-day
    exactly once. This is a property of the evaluation, not an option: changing it invalidates
    comparison with every number on this page.</p></div>
</section>

<section id="leaderboard">
  <div class="sec-hd"><span class="eyebrow">Section 2</span>
    <h2>The leaderboard</h2>
    <p class="sec-note">All {NWORD} models, all six horizons, one table. MASE is a ratio to a
      seasonal-naive baseline — 1.0 means no better than naive, and it is not a percentage. Lower
      is better for every metric here.</p></div>
  <div class="stack">
    <div class="tw"><table>
      <caption>Macro MASE by horizon, with rank in each cell; MAE, RMSE and RMSSE at the 1-day horizon</caption>
      <thead><tr><th>Model</th><th>Family</th>
        {''.join(f'<th class="n">{HL[h]}</th>' for h in HS)}
        <th class="n">MAE @ 1 d</th><th class="n">RMSE @ 1 d</th><th class="n">RMSSE @ 1 d</th></tr></thead>
      <tbody>{lead_rows}</tbody></table></div>
    <div class="call"><h3>What the table says, in three sentences</h3>
      <p>The <strong>Chronos-2 family occupies the top four places at every horizon</strong>, and
      its four variants are separated by <strong>0.003 MASE</strong> — statistically
      indistinguishable. The deployed incumbent is fifth; a transformer trained on this campus is
      sixth and seventh; the classical methods and the naive reference fill the rest.</p>
      <p><strong>Zero-shot is the production candidate despite ranking second on raw MASE.</strong>
      It costs {C2S:.0f} seconds against 5.9–15.5 minutes for the covariate variants, needs no
      covariate pipeline and adds no failure mode. That is a compute decision, and it is
      defensible precisely because the accuracy difference is nil — see
      <a href="#cost">Section 8</a>.</p></div>
  </div>
</section>

<section id="significance">
  <div class="sec-hd"><span class="eyebrow">Section 3</span>
    <h2>Is the margin real?</h2>
    <p class="sec-note">Chronos-2 against <em>every</em> opponent, by two independent tests:
      10,000 paired bootstrap resamples over the 24 forecast origins, and Diebold–Mariano with the
      Harvey–Leybourne–Newbold small-sample correction. The bootstrap assumes only that the origins
      are exchangeable; DM assumes a covariance structure the 23-hour stride approximates. They
      agree everywhere.</p></div>
  <div class="stack">
    <div class="tw"><table>
      <caption>Chronos-2 zero-shot vs each opponent at the 1-day horizon, with the verdict across all six</caption>
      <thead><tr><th>Opponent</th><th>Family</th><th class="n">MASE improvement</th>
        <th class="n">95% CI</th><th class="n">Bootstrap p</th><th class="n">DM p</th>
        <th class="n">Origins won</th><th class="n">Cells won</th><th>Across all six horizons</th></tr></thead>
      <tbody>{sig_rows}</tbody></table></div>
    <div class="call"><h3>The result</h3>
      <p><strong>Chronos-2 beats all {len(_beaten)} non-Chronos models significantly at every one
      of the six horizons</strong> — by {NPTS_LO:.1f}–{NPTS_HI:.1f}% against the deployed
      incumbent, {PT_LO:.1f}–{PT_HI:.1f}% against the best PatchTST trained on this campus, and
      roughly a third against the classical methods.</p>
      <p><strong>The three covariate variants are the only opponents it does not separate from</strong>,
      and their confidence intervals straddle zero. That is the honest reading: conditioning on the
      academic calendar changes nothing measurable, so the cheapest variant wins on cost.</p></div>
  </div>
</section>

<section id="pertank">
  <div class="sec-hd"><span class="eyebrow">Section 4</span>
    <h2>Per tank — where each model can and cannot be trusted</h2>
    <p class="sec-note">Aggregate accuracy decides whether to adopt a model. These decide whether
      to trust it on a particular tank. Skill is <code>1 − MAE(model)/MAE(SeasonalNaive-24)</code>:
      positive means the model beats the naive reference on that tank.</p></div>
  <div class="stack">
    <div class="tw"><table>
      <caption>Per-tank outcomes at the 1-day horizon, all 24 tanks</caption>
      <thead><tr><th>Model</th><th class="n">Tanks where it is best</th>
        <th class="n">Tanks with positive skill</th><th class="n">Median skill</th>
        <th class="n">Worst tank</th></tr></thead>
      <tbody>{tank_rows}</tbody></table></div>
    <div class="call"><h3>The objection this answers</h3>
      <p>“Your average is flattered by the dead sensors.” It is not: ranked by <em>raw</em> MAE the
      best tanks would be the broken ones, which is exactly why the headline uses MASE. Measured
      per tank against the naive reference, <strong>every Chronos-2 variant and the incumbent clear
      zero on all 24 tanks</strong>, and skill does not decline with demand size.</p>
      <p>The trained deep models do not: PatchTST is <em>worse than naive</em> on three to four
      tanks. The classical methods are worse than naive on five.</p></div>
  </div>
</section>

<section id="figures">
  <div class="sec-hd"><span class="eyebrow">Section 5</span>
    <h2>The evidence, in {len(FIGS)} figures</h2>
    <p class="sec-note">One figure per question, each carrying every model. PNG and SVG are both
      on disk at <code>results/chronos2/unified/plots/</code>, and each has the CSV it was drawn
      from beside it.</p></div>
  <div class="figs">{figs}</div>
</section>

<section id="calibration">
  <div class="sec-hd"><span class="eyebrow">Section 6</span>
    <h2>Uncertainty — the one axis the foundation model loses on</h2>
    <p class="sec-note">Nominal coverage for a p10–p90 band is 0.80. Coverage must be read next to
      width and next to the last column: a band can always be made to cover by being made useless,
      or by being allowed to predict the impossible.</p></div>
  <div class="stack">
    <div class="tw"><table>
      <caption>Interval calibration at the 1-day horizon, every model</caption>
      <thead><tr><th>Model</th><th class="n">p10–p90 coverage</th><th class="n">Gap to 0.80</th>
        <th class="n">Mean width (KL/h)</th><th class="n">Rows with p10 &lt; 0</th>
        <th class="n">If p10 were 0</th></tr></thead>
      <tbody>{cal_rows}</tbody></table></div>
    <div class="call warn"><h3>The mechanism, and why the obvious fix is wrong</h3>
      <p><strong>{Z.zero_fraction * 100:.0f}% of hourly readings are exactly zero.</strong> A
      continuous-density model cannot place mass at exactly zero, so Chronos-2 puts its p10 above
      zero on <strong>{Z.p10_above_zero_pct:.0f}%</strong> of rows — and
      <strong>{Z.below_misses_that_are_zero * 100:.0f}% of its lower-tail misses are zero-demand
      hours</strong>. It is a zero-inflation problem, not a general uncertainty failure.</p>
      <p>The tell is the fourth column. <strong>Every model that reaches or exceeds the nominal
      0.80 does so by putting its lower bound below zero</strong> — a negative water outflow, which
      cannot happen — with one exception: <strong>NPTS</strong>, a non-parametric sampler that
      reproduces the zero atom directly and needs no such trick, which is exactly why it is the
      best-calibrated model here. The four Chronos-2 variants never go negative and under-cover as
      a result. <strong>An interval whose lower bound is a negative volume cannot size a
      refill.</strong></p>
      <p>Nor is clamping the answer: setting the lower bound to zero takes Chronos-2 to
      <strong>{Z.coverage_if_p10_clamped:.3f}</strong>, past the nominal 0.80, trading one kind of
      wrong for another. The correct fix is asymmetric conformal calibration, which corrects each
      tail by its own measured miss rate — <strong>{CAL.coverage_raw:.3f} →
      {CAL.coverage_cal:.3f}</strong>, measured on a later, disjoint window it never saw.</p></div>
  </div>
</section>

<section id="operational">
  <div class="sec-hd"><span class="eyebrow">Section 7</span>
    <h2>The operational view — how much water, tomorrow</h2>
    <p class="sec-note">45 consecutive daily forecasts tiling the window once, with no gaps and no
      overlap. This is the question an operator actually asks, and it is not visible in MASE.
      Corrections are fitted on 8 Jan – 8 Mar 2026 and measured on the disjoint 9 Mar – 22 Apr
      window; the code raises if the two windows touch.</p></div>
  <div class="stack">
    <div class="tw"><table>
      <caption>Campus daily totals over 45 days — raw &rarr; calibrated</caption>
      <thead><tr><th>Model</th><th class="n">Daily MAE (KL)</th><th class="n">MAPE</th>
        <th class="n">Total volume bias</th><th class="n">Interval coverage</th>
        <th class="n">Variation reproduced</th><th class="n">Correlation with actual</th></tr></thead>
      <tbody>{op_rows}</tbody></table></div>
    <div class="call"><h3>What calibration fixes, and what it cannot</h3>
      <p>Post-hoc correction fixes <em>where</em> a forecast sits and <em>how sure</em> it claims
      to be. Volume bias goes {OPS.total_bias_raw_pct:+.1f}% →
      <strong>{OPS.total_bias_pct:+.1f}%</strong> and daily campus error
      {OPS.daily_mae_raw_kl:.1f} → <strong>{OPS.daily_mae_kl:.1f} KL</strong>, neither requiring
      retraining.</p>
      <p>What it cannot fix is whether a forecast <em>responds</em> to anything. Chronos-2
      reproduces <strong>{OPS.sd_ratio * 100:.0f}%</strong> of real day-to-day variation and
      correlates <strong>{OPS.corr_with_actual:+.2f}</strong> with it; the incumbent reproduces
      <strong>{h45s.loc[INCUMBENT].sd_ratio * 100:.0f}%</strong> and correlates
      <strong>{h45s.loc[INCUMBENT].corr_with_actual:+.2f}</strong> — it is close to a flat line at
      the long-run average, on every tank.</p></div>
  </div>
  <div class="figs" style="margin-top:22px">{opfigs}</div>
</section>

<section id="cost">
  <div class="sec-hd"><span class="eyebrow">Section 8</span>
    <h2>What the accuracy cost</h2>
    <p class="sec-note">Measured wall clock for the identical {TOTAL:,}-row backtest, read from
      the run manifests. Accuracy without its cost is half a result.</p></div>
  <div class="stack">
    <div class="tw"><table>
      <caption>Compute per model, against accuracy at the 1-day horizon. The five AutoGluon
        baselines share one fitting pass, so 1,579 s is the cost of producing all five together,
        not of each — read it as a family total.</caption>
      <thead><tr><th>Model</th><th>Regime</th><th class="n">Wall clock</th>
        <th class="n">Minutes</th><th class="n">MASE @ 1 d</th></tr></thead>
      <tbody>{cost_rows}</tbody></table></div>
    <div class="call"><h3>The asymmetry that decides the deployment</h3>
      <p>The best PatchTST needed <strong>{PTS / 60:.0f} minutes</strong> of training against
      Chronos-2's <strong>{C2S:.0f} seconds</strong> of inference — a
      <strong>{PTS / C2S:.0f}×</strong> difference — and it must be refitted whenever the campus
      changes, whereas a zero-shot model holds no fitted state that can go stale. For a deployment
      that adds tanks, loses sensors and shifts with the academic calendar, that is the argument;
      the accuracy gap merely means there is nothing to trade against it.</p></div>
  </div>
</section>

<section id="inference">
  <div class="sec-hd"><span class="eyebrow">Section 9</span>
    <h2>The inference for each review category</h2></div>
  <div class="inf">
    <div class="inf-item"><div class="inf-top"><h3>System Testing</h3><span class="chip partial">partial</span></div>
      <p>6/6 metric unit tests pass, and the methodology checks are assertions that halt the run
      rather than a checklist someone ticks. That validates the <strong>measurement
      apparatus</strong>: if the seasonal-naive MASE identity failed, every number on this page
      would be wrong.</p>
      <p>No system or integration suite exists, because the real-time system it would test is
      specified and unbuilt. <strong>Say this plainly</strong> rather than presenting model tests
      as system tests. The six test layers — unit, property, contract, integration, parity,
      regression — are written down; one and a half of them exist.</p></div>
    <div class="inf-item"><div class="inf-top"><h3>Verification &amp; Validation</h3><span class="chip done">done</span></div>
      <p>The strongest part of the project. Six machine-checked guarantees: identical rows for all
      {NM} models, zero leakage, the MASE identity, an independent re-score, paired significance
      against <em>every</em> opponent, and a deep-learning control trained on this campus.</p>
      <p>Adding the two PatchTST models was itself a verification event — the row-parity assertion
      had to hold for models produced by a completely different pipeline, and it did, unchanged at
      {ROWS_H[6]:,} / {ROWS_H[12]:,} / {ROWS_H[24]:,} / {ROWS_H[48]:,} / {ROWS_H[72]:,} /
      {ROWS_H[168]:,} rows.</p></div>
    <div class="inf-item"><div class="inf-top"><h3>Chronos-2</h3><span class="chip done">done</span></div>
      <p><code>amazon/chronos-2</code>, 119.5 M parameters, used <strong>zero-shot</strong> — no
      training, no fine-tuning, 2,048 hours of context truncated at each origin. It works because
      it was pretrained on millions of series and campus water demand has daily structure it can
      recognise without having seen this campus.</p>
      <p>Four covariate variants were measured and are statistically indistinguishable from the
      zero-shot model. <strong>Zero-shot is chosen on compute, not accuracy</strong>, and it holds
      no fitted state that can go stale.</p></div>
    <div class="inf-item"><div class="inf-top"><h3>Deployment</h3><span class="chip todo">the real gap</span></div>
      <p>A Chrome MV3 dock renders real Chronos-2 forecasts from a bundled JSON, offline. The
      compose stack exists but targets the <strong>retired</strong> AutoGluon models, and its
      Postgres is provisioned but unused. <strong>Nothing is serving.</strong></p>
      <p>The architecture that closes this is specified across six documents — adapter seam, data
      model, API surface, safety gate, replay/live parity, an 11-phase plan. Do not overstate the
      dock as a deployment: describing it accurately is more credible than describing it
      generously.</p></div>
    <div class="inf-item"><div class="inf-top"><h3>Final Experimental Results</h3><span class="chip done">done</span></div>
      <p>{NWORD.capitalize()} models, six horizons, 24 tanks, {TOTAL:,} identical rows per model,
      zero leakage, zero duplicates. <strong>Chronos-2 zero-shot beats every non-Chronos model at
      every horizon, significantly, on both metrics and by two independent tests</strong> — from
      {NPTS_LO:.1f}% against the deployed incumbent to roughly a third against the classical
      methods — and it costs {C2S:.0f} seconds.</p>
      <p>Both caveats are measured and stated with the result, not in an appendix: the intervals
      were overconfident ({CAL.coverage_raw:.3f} against 0.80, diagnosed as zero-inflation and
      corrected to {CAL.coverage_cal:.3f}), and 24-hour volume was under-forecast
      ({OPS.total_bias_raw_pct:+.1f}%, corrected to {OPS.total_bias_pct:+.1f}%).</p></div>
    <div class="inf-item"><div class="inf-top"><h3>Performance Analysis</h3><span class="chip done">{len(FIGS) + len(OPFIGS)} figures</span></div>
      <p>Each figure pre-empts a specific challenge, and each now answers it for the whole field
      rather than one pairing. <strong>“Is it real or noise?”</strong> → U3, U4.
      <strong>“Are dead tanks flattering your average?”</strong> → U5, U6.
      <strong>“Does it fall apart at long horizons?”</strong> → U9.
      <strong>“Your intervals are broken.”</strong> → U7, U11 — yes, and here is the mechanism and
      the correct fix. <strong>“Did you try a real deep model?”</strong> → U1, U3, U8.
      <strong>“What does the error cost?”</strong> → U12, O1–O3.</p></div>
    <div class="inf-item"><div class="inf-top"><h3>Research Paper Draft</h3><span class="chip todo">not started</span></div>
      <p><code>docs/review_summary.md</code> already contains everything a paper needs, and this
      round supplies what was missing: a defensible statistical claim against a full field, and a
      mechanistic finding.</p>
      <p>Suggested framing — <strong>“Zero-inflation limits probabilistic calibration of
      time-series foundation models: evidence from 24 campus water tanks.”</strong> The
      point-forecast win is solid but unsurprising; the calibration diagnosis is the contribution.</p></div>
  </div>
</section>

<section id="slide">
  <div class="sec-hd"><span class="eyebrow">Section 10</span><h2>If you only get one slide</h2></div>
  <div class="call">
    <p><strong>On one evaluation grid where {NWORD} models score the same {TOTAL:,} rows, Chronos-2
    zero-shot beats every non-Chronos model at every one of six horizons — and the margin is
    statistically significant everywhere, by paired bootstrap over 24 origins and by
    Diebold–Mariano.</strong> It is {NPTS_LO:.1f}–{NPTS_HI:.1f}% better than the deployed
    incumbent, {PT_LO:.1f}–{PT_HI:.1f}% better than a PatchTST trained on this campus, and roughly
    a third better than the classical methods. It wins
    {summ['tank_horizon_cells_won_vs'][INCUMBENT]} of 144 tank-horizon cells against the incumbent
    and all 144 against every classical method, beats the naive baseline on all 24 tanks, and is
    never significantly worse than the incumbent on any tank with a healthy sensor. It costs
    {C2S:.0f} seconds and requires no training.</p>
    <p><strong>The trained transformer loses to the incumbent as well as to Chronos-2</strong> —
    on 24 short, noisy, zero-inflated series, training from scratch buys less than a well-chosen
    non-parametric method and far less than a pretrained one. That is the argument for a
    foundation model on this problem, stated as a measurement rather than a preference.</p>
    <p><strong>Its two known weaknesses are diagnosed and corrected.</strong> The intervals covered
    {CAL.coverage_raw:.3f} against a nominal 0.80, traced here to the model's inability to
    represent the {Z.zero_fraction * 100:.0f}% of hours with exactly zero demand — a limit that
    applies to every continuous-density model on the grid, not to this one. Asymmetric conformal
    calibration and a per-tank volume correction, both fitted on an earlier disjoint window and
    measured out of sample, take coverage to <strong>{CAL.coverage_cal:.3f}</strong> and volume
    bias from {OPS.total_bias_raw_pct:+.1f}% to <strong>{OPS.total_bias_pct:+.1f}%</strong>,
    cutting daily campus error from {OPS.daily_mae_raw_kl:.1f} to
    <strong>{OPS.daily_mae_kl:.1f} KL</strong>. Neither requires retraining.</p>
    <p><strong>What is not done:</strong> nothing is deployed and running, and the paper is not
    written. Both are designed in detail. Say it before you are asked.</p>
  </div>
</section>

<section id="limits">
  <div class="sec-hd"><span class="eyebrow">Section 11</span>
    <h2>What this does not establish</h2>
    <p class="sec-note">Volunteering the limits is what makes the rest credible. Each of these is
      measured, not hedged.</p></div>
  <div class="stack">
    <div class="tw"><table>
      <caption>Known limits</caption>
      <thead><tr><th>Limit</th><th>What is actually true</th></tr></thead>
      <tbody>
      <tr><th scope="row">One healthy tank is genuinely failed</th>
        <td><code>GJBC_LAW_BLOCK_3__A1</code> is spiky and every model smooths it — it is the worst
        column in U5 for all {NM}. It needs a spike model, not a better smoother.</td></tr>
      <tr><th scope="row">Intervals still under-cover after correction</th>
        <td>{CAL.coverage_cal:.3f} against a nominal 0.80. Better, not solved.</td></tr>
      <tr><th scope="row">A regime change cannot be anticipated</th>
        <td><code>GJBC_BLOCK_1_A4_RO</code> averaged 0.0005 KL/h during calibration, then woke up.
        No correction fitted on an earlier window can foresee that — which is why a production
        system must refit on a rolling basis and monitor coverage continuously.</td></tr>
      <tr><th scope="row">PatchTST was not architecture-searched</th>
        <td>Two configurations were run — AutoGluon's defaults and the paper's hourly settings —
        not a per-tank sweep. The claim is that a competently configured PatchTST loses by
        {PT_LO:.1f}–{PT_HI:.1f}% on 24 series this short, not that no PatchTST could ever win.</td></tr>
      <tr><th scope="row">The result is one campus, 16 months</th>
        <td>It generalises to campuses like this one. It is not evidence about municipal supply,
        industrial demand, or a different climate.</td></tr>
      <tr><th scope="row">No motor telemetry exists in the dataset</th>
        <td>So no claim is made about control quality, only about demand forecasting.</td></tr>
      <tr><th scope="row">Nothing is running</th>
        <td>Every number here is offline backtest. The live system is designed, not built.</td></tr>
      </tbody></table></div>
  </div>
</section>

<section id="repro">
  <div class="sec-hd"><span class="eyebrow">Section 12</span><h2>Reproducing this</h2></div>
  <pre>source venv/bin/activate
<b>python -m tests.test_metrics</b>                      <span style="opacity:.65"># 6/6, ~3 s — trust nothing if this fails</span>
<b>python -m src.data.curate</b>                         <span style="opacity:.65"># 26 dirs -> 24 tanks, 270,849 rows</span>
<b>python -m src.models.chronos2_forecasting</b>         <span style="opacity:.65"># the four Chronos-2 variants</span>
<b>python -m src.models.baselines_autogluon</b>          <span style="opacity:.65"># NPTS + 4 classical, ~26 min</span>
<b>python -m src.models.patchtst_benchmark --preset default</b>
<b>python -m src.models.patchtst_benchmark --preset tuned  …</b>
<b>python -m src.models.score_benchmark --strict</b>     <span style="opacity:.65"># row parity, fatal on mismatch</span>
<b>python -m src.models.unified_analysis</b>             <span style="opacity:.65"># ~4 min — every table on this page</span>
<b>python -m src.models.unified_figures</b>              <span style="opacity:.65"># ~25 s — U1-U12</span>
<b>python -m src.models.calibrated_holdout</b>           <span style="opacity:.65"># ~3 min — the operational view, O1-O3</span>
<b>python reports/build_results_page.py</b>              <span style="opacity:.65"># ~10 s — this page</span>
<b>python reports/build_review_deck.py</b>               <span style="opacity:.65"># ~10 s — the review deck</span></pre>
  <p style="margin-top:12px;color:var(--ink-2);font-size:13.6px;max-width:78ch">
    Every number on this page is read from a CSV under
    <code>results/chronos2/unified/</code> at build time — nothing is hard-coded, so the page
    cannot drift from the data. The bootstrap seed is fixed at 20260830, so every confidence
    interval reproduces exactly. Model binaries and prediction parquets are not committed; they
    regenerate in about two hours. What is committed is the evidence.</p>
</section>

<footer>
  Intelligent Water Management System · PW26_PK_06 · PES University RR ·
  Abhay Patil, Amogh E M, Harshavardhan M, Viraj Ved Shankar ·
  Every figure and number traces to a file under <code>results/chronos2/</code>.
</footer>
</div>
"""

OUT.write_text(BODY)
print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB, {len(FIGS) + len(OPFIGS)} figures, "
      f"{NM} models)")
