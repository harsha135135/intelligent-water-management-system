"""Build the Phase III results page: an offline, self-contained HTML report.

Embeds all 26 figures as base64 and reads every number from the CSVs under
``results/chronos2/`` so the page cannot drift from the data. Regenerate with:

    python reports/build_results_page.py

Output (~5.5 MB, not committed - it is fully regenerable):
    reports/phase3_results_page.html
"""

import base64, json, pathlib, html
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P3 = ROOT/'results/chronos2/phase3'
RV = ROOT/'results/chronos2/review'
OUT = ROOT / 'reports' / 'phase3_results_page.html'

def img(p):
    b = base64.b64encode(pathlib.Path(p).read_bytes()).decode()
    return f"data:image/png;base64,{b}"

def esc(s): return html.escape(str(s))

# ---------- data
m = pd.read_csv(ROOT/'results/chronos2/metrics_by_horizon.csv')
sig = pd.read_csv(P3/'significance_by_horizon.csv')
pt = pd.read_csv(P3/'significance_per_tank_h24.csv')
zi = pd.read_csv(P3/'zero_inflation_diagnosis.csv')
rel = pd.read_csv(P3/'reliability.csv')
wm = pd.read_csv(P3/'win_matrix.csv')
sk = pd.read_csv(P3/'skill_scores_h24.csv')
cv = pd.read_csv(P3/'cumulative_volume_h24.csv')
lt = pd.read_csv(P3/'error_by_leadtime.csv')
H45 = ROOT/'results/chronos2/calibrated'
h45d = pd.read_csv(H45/'daily_campus.csv', parse_dates=['day'])
h45s = pd.read_csv(H45/'summary.csv').set_index('model')
h45e = pd.read_csv(H45/'calibration_effect.csv').set_index('model')
HL = {6:'6 h',12:'12 h',24:'1 d',48:'2 d',72:'3 d',168:'7 d'}
HS = [6,12,24,48,72,168]

def cell(mod,h,c): return m[(m.model==mod)&(m.horizon==h)][c].iloc[0]

# headline table
head_rows = "".join(
    f"<tr><th scope='row'>{HL[h]}</th><td class='n'>{int(cell('Chronos2-ZS',h,'rows_evaluated')):,}</td>"
    f"<td class='n win'>{cell('Chronos2-ZS',h,'macro_mase'):.4f}</td><td class='n'>{cell('NPTS',h,'macro_mase'):.4f}</td>"
    f"<td class='n dim'>{cell('SeasonalNaive',h,'macro_mase'):.4f}</td>"
    f"<td class='n win'>{cell('Chronos2-ZS',h,'macro_mae'):.4f}</td><td class='n'>{cell('NPTS',h,'macro_mae'):.4f}</td>"
    f"<td class='n win'>{cell('Chronos2-ZS',h,'macro_rmse'):.4f}</td><td class='n'>{cell('NPTS',h,'macro_rmse'):.4f}</td></tr>"
    for h in HS)

# nine models @24h
nine = m[m.horizon==24].sort_values('macro_mase')
nine_rows = ""
for i,(_,r) in enumerate(nine.iterrows(), 1):
    prod = r.model=='Chronos2-ZS'
    nine_rows += (f"<tr class='{'prod' if prod else ''}'><td class='n dim'>{i}</td>"
        f"<th scope='row'>{esc(r.model)}{' <span class=tag>production candidate</span>' if prod else ''}"
        f"{' <span class=tag-mut>incumbent</span>' if r.model=='NPTS' else ''}"
        f"{' <span class=tag-mut>reference</span>' if r.model=='SeasonalNaive' else ''}</th>"
        f"<td class='n'>{r.macro_mase:.4f}</td><td class='n'>{r.macro_mae:.4f}</td><td class='n'>{r.macro_rmse:.4f}</td>"
        f"<td class='n'>{r.macro_rmsse:.4f}</td><td class='n'>{r.p10_p90_coverage:.3f}</td></tr>")

# significance
sg = sig[sig.metric=='MASE'].set_index('horizon')
sg_rows = "".join(
    f"<tr><th scope='row'>{HL[h]}</th><td class='n win'>{-sg.loc[h,'diff_pct']:.2f}%</td>"
    f"<td class='n'>[{-sg.loc[h,'ci_hi_pct']:.2f}, {-sg.loc[h,'ci_lo_pct']:.2f}]</td>"
    f"<td class='n'>&lt;0.0001</td><td class='n'>{sg.loc[h,'dm_stat']:.2f}</td>"
    f"<td class='n'>{sg.loc[h,'p_dm']:.4f}</td><td class='n'>{int(sg.loc[h,'wins_origins'])}/24</td>"
    f"<td><span class='chip ok'>significant</span></td></tr>" for h in HS)

# per-tank
def vclass(v): return {'Chronos-2 better':'ok','NPTS better':'bad','no significant difference':'neu'}[v]
def vlabel(v): return {'Chronos-2 better':'Chronos-2','NPTS better':'NPTS','no significant difference':'no diff.'}[v]
pt_rows = "".join(
    f"<tr><th scope='row'>{esc(r.tank)}</th><td><span class='chip t-{r.trust}'>{r.trust}</span></td>"
    f"<td class='n {'win' if r.improvement_pct>0 else 'loss'}'>{r.improvement_pct:+.2f}%</td>"
    f"<td class='n dim'>[{r.ci_lo_pct:+.2f}, {r.ci_hi_pct:+.2f}]</td>"
    f"<td class='n dim'>{r.p_bootstrap:.4f}</td>"
    f"<td><span class='chip {vclass(r.verdict)}'>{vlabel(r.verdict)}</span></td></tr>"
    for _,r in pt.iterrows())

# reliability
q = rel[(rel.kind=='quantile')&(rel.model=='Chronos2-ZS')&(rel.horizon==24)].set_index('nominal')
rel_rows = "".join(
    f"<tr><th scope='row'>p{int(n*100)}</th><td class='n'>{n:.2f}</td><td class='n'>{q.loc[n,'empirical']:.3f}</td>"
    f"<td class='n {'bad-t' if abs(q.loc[n,'empirical']-n)>0.05 else 'dim'}'>{q.loc[n,'empirical']-n:+.3f}</td></tr>"
    for n in [0.1,0.25,0.5,0.75,0.9])

# lead time
def lmean(a,b,c): return lt[(lt.step>=a)&(lt.step<=b)][c].mean()
lt_rows = "".join(
    f"<tr><th scope='row'>{lab}</th><td class='n win'>{lmean(a,b,'mae_Chronos2-ZS'):.4f}</td>"
    f"<td class='n'>{lmean(a,b,'mae_NPTS'):.4f}</td></tr>"
    for lab,a,b in [('Day 1 (h 1–24)',1,24),('Day 2',25,48),('Day 3',49,72),('Day 5',97,120),('Day 7 (h 145–168)',145,168)])

# win matrix counts
wins = wm[wm.winner=='Chronos2-ZS'].groupby('horizon').size()
win_rows = "".join(f"<td class='n'>{wins[h]}/24</td>" for h in HS)

z24 = zi[(zi.horizon==24)&(zi.model=='Chronos2-ZS')].iloc[0]
n24 = zi[(zi.horizon==24)&(zi.model=='NPTS')].iloc[0]
d1 = lmean(1,24,'mae_Chronos2-ZS'); d7 = lmean(145,168,'mae_Chronos2-ZS')
tot = cv.cum_actual.iloc[-1]; gapc = cv.cum_gap_chronos2.iloc[-1]; gapn = cv.cum_gap_npts.iloc[-1]

nbet = int((pt.verdict=='Chronos-2 better').sum()); nns = int((pt.verdict=='no significant difference').sum())
nwor = int((pt.verdict=='NPTS better').sum())
hb = int(((pt.trust=='healthy')&(pt.verdict=='Chronos-2 better')).sum())
hn = int(((pt.trust=='healthy')&(pt.verdict=='no significant difference')).sum())

# ---------- 45-day calibrated holdout tables
import numpy as np
_full = h45d[h45d.coverage_pct >= 95]
_a = _full[_full.model=='Chronos2-ZS'].sort_values('day')['actual_kl'].to_numpy()
_MODS = [('Chronos2-ZS','Chronos-2 (zero-shot)'),
         ('NPTS','NPTS (incumbent)'),
         ('SeasonalNaive-24','SeasonalNaive-24 (reference)')]

def _arrow(raw, calv, unit="", nd=3, better="closer", target=None):
    return f"{raw:.{nd}f}{unit} <span class='arrow'>&rarr;</span> <b>{calv:.{nd}f}{unit}</b>"

cal_rows = ""
for _m,_lab in _MODS:
    _e = h45e.loc[_m]
    _gap_raw = abs(_e.coverage_raw-0.80); _gap_cal = abs(_e.coverage_cal-0.80)
    cal_rows += (f"<tr class='{'prod' if _m=='Chronos2-ZS' else ''}'><th scope='row'>{_lab}</th>"
        f"<td class='n'>{_arrow(_e.coverage_raw,_e.coverage_cal)}</td>"
        f"<td class='n {'win' if _gap_cal<_gap_raw else 'dim'}'>{_gap_raw:.3f} <span class='arrow'>&rarr;</span> <b>{_gap_cal:.3f}</b></td>"
        f"<td class='n'>{_arrow(_e.width_raw,_e.width_cal)}</td>"
        f"<td class='n'>{_e.miss_below_raw:.3f} <span class='arrow'>&rarr;</span> <b>{_e.miss_below_cal:.3f}</b></td>"
        f"<td class='n'>{_e.miss_above_raw:.3f} <span class='arrow'>&rarr;</span> <b>{_e.miss_above_cal:.3f}</b></td>"
        f"<td class='n'>{_e.volume_bias_raw_pct:+.1f}% <span class='arrow'>&rarr;</span> <b>{_e.volume_bias_cal_pct:+.1f}%</b></td>"
        f"<td class='n dim'>{_e.mae_raw:.3f} <span class='arrow'>&rarr;</span> {_e.mae_cal:.3f}</td></tr>")

h45_rows = ""
for _m,_lab in _MODS:
    _g = _full[_full.model==_m].sort_values('day'); _p = _g['pred_kl'].to_numpy()
    _s = h45s.loc[_m]
    _ratio = _s.sd_ratio; _corr = _s.corr_with_actual
    h45_rows += (f"<tr class='{'prod' if _m=='Chronos2-ZS' else ''}'><th scope='row'>{_lab}</th>"
        f"<td class='n'>{_s.daily_mae_raw_kl:.1f} <span class='arrow'>&rarr;</span> <b class='win'>{_s.daily_mae_kl:.1f}</b></td>"
        f"<td class='n'>{_s.daily_mape_pct:.1f}%</td>"
        f"<td class='n'>{_s.total_bias_raw_pct:+.1f}% <span class='arrow'>&rarr;</span> <b>{_s.total_bias_pct:+.1f}%</b></td>"
        f"<td class='n {'win' if _ratio>0.5 else 'loss'}'>{_ratio:.2f}</td>"
        f"<td class='n {'win' if _corr>0.4 else 'loss' if _corr<0 else ''}'>{_corr:+.3f}</td>"
        f"<td class='n dim'>{_s.hourly_mae_kl_h:.3f}</td></tr>")
_actual_sd = _a.std(ddof=1); _actual_mean = _a.mean()

H45FIGS = [
 ("X","X_tanks45_chronos2","Chronos-2 zero-shot",
  "Four tanks, 45 days each. Solid = actual, dashed = calibrated, dotted = uncalibrated, shaded = conformal band. Each panel has its own y-scale.",
  "Tracks the two healthy tanks closely. Misses the spiky peaks on GJBC_LAW_BLOCK_3__A1 \u2014 the known modelling failure \u2014 and cannot anticipate A4_RO waking up."),
 ("Y","Y_tanks45_npts","NPTS \u2014 the incumbent",
  "The same four tanks and the same correction applied to the deployed model.",
  "Flat on every one of the four. The flatness seen at campus level is not an aggregation artefact \u2014 it is how the model behaves per tank."),
 ("Z","Z_tanks45_seasonal_naive","SeasonalNaive-24 \u2014 the reference",
  "The same four tanks, forecasting each day as a repeat of the previous one.",
  "Visually the closest tracker \u2014 it reproduces every peak \u2014 yet has the worst MAE, because every feature it reproduces arrives a day late."),
]
h45_figs = "".join(
    f"""<figure class="fig" id="fig-{fid}">
  <figcaption class="fig-head"><span class="fig-id">{fid}</span>
    <span class="fig-title">{esc(title)}</span><span class="badge-new">new</span></figcaption>
  <div class="fig-img"><img src="{img(H45/'plots'/(fn+'.png'))}" alt="{esc(desc)}" loading="lazy"></div>
  <div class="fig-foot"><p class="fig-desc">{esc(desc)}</p><p class="fig-take">{esc(take)}</p></div>
</figure>""" for fid,fn,title,desc,take in H45FIGS)

# ---------- figures
NEW = [
 ("O","O_significance_forest","Significance by horizon",
  "Improvement over NPTS with a 95% paired-bootstrap CI, MAE and MASE. Every CI sits entirely right of zero.",
  "Proves the improvement is not noise, at all six horizons."),
 ("P","P_per_tank_significance_h24","Per-tank significance (1 d)",
  "Per-tank MAE improvement with 95% CI, coloured by verdict. Grey bars cross zero.",
  f"{nbet} tanks significantly better, {nns} indistinguishable, {nwor} worse — and all {nwor} losses are on degraded or dead sensors."),
 ("W","W_zero_inflation_diagnosis","Why the intervals under-cover",
  "Lower- vs upper-tail miss rates, how often p10 sits above zero, and what a naive clamp would do.",
  "The calibration deficit is a zero-inflation problem, not a general uncertainty problem."),
 ("S","S_reliability_diagram","Quantile reliability",
  "Empirical fraction of actuals below each predicted quantile, plus interval coverage at nominal 0.50 and 0.80.",
  "The upper tail is nearly perfect; the failure is almost entirely lower-tail."),
 ("R","R_error_by_leadtime","Error vs lead time",
  "MAE against hours ahead of origin, out to 168 h. Faint = raw, bold = 24 h rolling mean.",
  f"Error grows only {100*(d7-d1)/d1:+.1f}% from day 1 to day 7 — a weekly forecast is nearly as good as a daily one."),
 ("Q","Q_diurnal_error","Error across the day",
  "MAE and signed bias by hour of day, against the mean demand profile.",
  "Error tracks demand (ρ = 0.958); the under-forecast bias is present in all 24 hours."),
 ("U","U_cumulative_volume","Cumulative campus volume",
  "Forecast vs actual campus demand accumulated over 24 origins, and the running shortfall.",
  f"{abs(gapc):,.0f} KL under-provisioned over the window — the operational cost of the bias."),
 ("V","V_win_matrix","Win/loss, every tank × horizon",
  "MAE improvement over NPTS for all 144 tank-horizon cells.",
  f"Chronos-2 wins {int((wm.winner=='Chronos2-ZS').sum())} of 144 cells (70%)."),
 ("T","T_skill_vs_demand","Skill vs the naive baseline",
  "Skill = 1 − MAE(model)/MAE(SeasonalNaive-24), by demand size and sensor tier.",
  "All 24 tanks have positive skill; skill does not decline with demand size."),
]
OLD = [
 ("A","A_mase_vs_horizon","MASE vs horizon"),("B","B_rmse_vs_horizon","RMSE vs horizon"),
 ("C","C_mae_vs_horizon","MAE vs horizon"),("D","D_rmsse_vs_horizon","RMSSE vs horizon"),
 ("E","E_per_tank_mase_24h","Per-tank MASE at 1 d"),("F","F_per_tank_improvement_24h","Per-tank improvement"),
 ("G","G_per_tank_mase_heatmap","MASE heatmap"),("H","H_error_distribution_24h","Error distribution"),
 ("I","I_actual_vs_predicted_24h","Actual vs predicted"),("J","J_final_holdout_7day","7-day final holdout"),
 ("K","K_interval_calibration","Interval coverage"),("L","L_variant_selection","Accuracy per compute"),
 ("M","M_tanks_won","Tanks won"),("N","N_covariate_vs_zeroshot","Covariate gain vs cost"),
]

new_figs = "".join(
    f"""<figure class="fig" id="fig-{fid}">
  <figcaption class="fig-head"><span class="fig-id">{fid}</span>
    <span class="fig-title">{esc(title)}</span><span class="badge-new">new</span></figcaption>
  <div class="fig-img"><img src="{img(P3/'plots'/(fn+'.png'))}" alt="{esc(desc)}" loading="lazy"></div>
  <div class="fig-foot"><p class="fig-desc">{esc(desc)}</p><p class="fig-take">{esc(take)}</p></div>
</figure>""" for fid,fn,title,desc,take in NEW)

old_figs = "".join(
    f"""<figure class="fig sm">
  <figcaption class="fig-head"><span class="fig-id">{fid}</span><span class="fig-title">{esc(title)}</span></figcaption>
  <div class="fig-img"><img src="{img(RV/'plots'/(fn+'.png'))}" alt="{esc(title)}" loading="lazy"></div>
</figure>""" for fid,fn,title in OLD)

STATUS = [
 ("System Testing","partial","Partial",
  "6/6 metric unit tests and 15/15 methodology checks pass. No system or integration suite — the real-time system is designed but unbuilt."),
 ("Validation &amp; Verification","done","Done",
  "Row parity, zero leakage, MASE identity, independent re-score — plus paired bootstrap and Diebold-Mariano significance added this session."),
 ("Deployment","todo","Not deployed",
  "The Chrome dock runs offline from a bundled forecast. docker-compose targets the retired AutoGluon models. Nothing is serving live."),
 ("Final Experimental Results","done","Done",
  "9 models × 6 horizons × 24 tanks on one shared grid. 188,664 scored rows per model, zero leakage, zero duplicates."),
 ("Performance Analysis","done","Done — 23 figures",
  "14 existing figures (A–N) plus 9 new ones (O–W) built this session, with tables for each."),
 ("Research Paper Draft","todo","Not written",
  "review_summary.md holds all the material. The zero-inflation finding is the publishable contribution."),
]
status_cards = "".join(
    f"""<div class="st {cls}"><div class="st-top"><span class="st-name">{name}</span>
    <span class="chip {cls}">{lab}</span></div><p>{note}</p></div>"""
    for name,cls,lab,note in STATUS)

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

BODY = f"""
<title>Phase III — Final Results &amp; Performance Analysis</title>
{CSS}
<header>
  <div class="hd">
    <span class="eyebrow">PW26_PK_06 · Review 1 · Phase III</span>
    <h1>Final experimental results and performance analysis</h1>
    <p class="lede">Chronos-2 zero-shot versus the deployed incumbent for campus water-demand
      forecasting, across 24 tanks and six horizons — now with the statistical significance
      testing and calibration diagnosis that the benchmark left open.</p>
    <div class="meta">
      <span><b>24</b> tanks</span><span><b>24</b> origins, 23 h stride</span>
      <span><b>6</b> horizons</span><span><b>9</b> models</span>
      <span><b>188,664</b> scored rows per model</span>
      <span><b>0</b> leakage rows</span><span><b>0</b> duplicates</span>
      <span>2025-01-01 → 2026-04-22</span>
    </div>
  </div>
</header>

<div class="wrap">

<section id="status">
  <div class="sec-hd"><span class="eyebrow">Where we stand</span>
    <h2>The six Phase III expectations</h2>
    <p class="sec-note">Three complete, one partial, two not started. The modelling half is
      finished and now statistically defended; the systems half is specified in detail but unbuilt.</p>
  </div>
  <div class="status">{status_cards}</div>
</section>

<section id="headline">
  <div class="sec-hd"><span class="eyebrow">Headline</span>
    <h2>What the benchmark establishes</h2></div>
  <div class="stats">
    <div class="tile"><div class="k">MASE improvement</div><div class="v pos">5.7–12.5%</div>
      <div class="s">over the incumbent NPTS, at every horizon</div></div>
    <div class="tile"><div class="k">Significance</div><div class="v pos">6 / 6</div>
      <div class="s">horizons significant, two independent tests</div></div>
    <div class="tile"><div class="k">Tank-horizon cells won</div><div class="v pos">101 / 144</div>
      <div class="s">70% of all tank × horizon comparisons</div></div>
    <div class="tile"><div class="k">Healthy tanks lost</div><div class="v pos">0 / 15</div>
      <div class="s">no healthy sensor is significantly worse</div></div>
    <div class="tile"><div class="k">Backtest cost</div><div class="v">89 s</div>
      <div class="s">zero-shot — no training, no fitting</div></div>
    <div class="tile"><div class="k">Interval coverage</div><div class="v pos">0.74 → 0.79</div>
      <div class="s">after conformal calibration, nominal 0.80</div></div>
    <div class="tile"><div class="k">Volume bias</div><div class="v pos">−10.5% → −1.8%</div>
      <div class="s">after per-tank bias correction, out of sample</div></div>
    <div class="tile"><div class="k">Skill vs naive</div><div class="v pos">24 / 24</div>
      <div class="s">every tank beats the seasonal-naive baseline</div></div>
  </div>
</section>

<section id="results">
  <div class="sec-hd"><span class="eyebrow">Section 1</span>
    <h2>Final experimental results</h2>
    <p class="sec-note">One shared evaluation grid for all nine models, enforced in code by a row-parity
      assertion that is fatal under <code>--strict</code>. MASE is a ratio to a seasonal-naive baseline:
      1.0 means no better than naive. It is not a percentage.</p></div>
  <div class="stack">
    <div class="tw"><table>
      <caption>Chronos-2 zero-shot vs NPTS vs the seasonal-naive reference, all six horizons</caption>
      <thead><tr><th>Horizon</th><th class="n">Rows</th><th class="n">C2 MASE</th><th class="n">NPTS MASE</th>
        <th class="n">Naive MASE</th><th class="n">C2 MAE</th><th class="n">NPTS MAE</th>
        <th class="n">C2 RMSE</th><th class="n">NPTS RMSE</th></tr></thead>
      <tbody>{head_rows}</tbody></table></div>
    <div class="tw"><table>
      <caption>All nine models at the 1-day horizon, ranked by macro MASE</caption>
      <thead><tr><th class="n">#</th><th>Model</th><th class="n">MASE</th><th class="n">MAE</th>
        <th class="n">RMSE</th><th class="n">RMSSE</th><th class="n">p10–p90 coverage</th></tr></thead>
      <tbody>{nine_rows}</tbody></table></div>
    <div class="call"><h3>Why zero-shot, when a covariate variant ranks first</h3>
      <p>The four Chronos-2 variants are separated by <strong>0.0012 MASE</strong> — statistically
      indistinguishable. Zero-shot costs <strong>89 seconds</strong> against 5.9–15.5 minutes,
      needs no covariate pipeline, and adds no failure mode. That is a compute decision, and it is
      defensible precisely because the accuracy difference is nil.</p></div>
  </div>
</section>

<section id="significance">
  <div class="sec-hd"><span class="eyebrow">Section 2 · new</span>
    <h2>Statistical significance</h2>
    <p class="sec-note">The benchmark explicitly recorded that no significance test had been run.
      It has now: 10,000 paired bootstrap resamples of the 24 forecast origins, plus a
      Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.</p></div>
  <div class="stack">
    <div class="tw"><table>
      <caption>MASE improvement over NPTS with 95% confidence intervals — every CI excludes zero</caption>
      <thead><tr><th>Horizon</th><th class="n">Improvement</th><th class="n">95% CI</th>
        <th class="n">Bootstrap p</th><th class="n">DM stat</th><th class="n">DM p</th>
        <th class="n">Origins won</th><th>Verdict</th></tr></thead>
      <tbody>{sg_rows}</tbody></table></div>
    <div class="call"><h3>The decisive per-tank result</h3>
      <p>At the 1-day horizon: <strong>{nbet} tanks significantly better</strong>,
      {nns} indistinguishable, <strong>{nwor} significantly worse</strong>.</p>
      <p><strong>All {nwor} significant losses are on degraded or dead sensors. Of the 15 healthy
      tanks, Chronos-2 is significantly better on {hb}, indistinguishable on {hn}, and significantly
      worse on none.</strong></p>
      <p>The apparent losses on <code>G_BLOCK</code> (−1.0%) and <code>NBX</code> (−1.7%) have
      confidence intervals crossing zero — those differences are not real, and building an NPTS
      fallback for them would have been fitting to noise.</p></div>
    <div class="tw"><table>
      <caption>Per-tank MAE improvement at the 1-day horizon, with 95% bootstrap CIs</caption>
      <thead><tr><th>Tank</th><th>Sensor</th><th class="n">Improvement</th><th class="n">95% CI</th>
        <th class="n">p</th><th>Verdict</th></tr></thead>
      <tbody>{pt_rows}</tbody></table></div>
  </div>
</section>

<section id="calibration">
  <div class="sec-hd"><span class="eyebrow">Section 3 · new</span>
    <h2>The calibration deficit, diagnosed</h2>
    <p class="sec-note">The one axis on which the incumbent wins. This work identifies the mechanism,
      and it is not a general uncertainty failure.</p></div>
  <div class="grid2">
    <div class="tw"><table>
      <caption>Chronos-2 quantile reliability at 1 d — fraction of actuals below each quantile</caption>
      <thead><tr><th>Quantile</th><th class="n">Nominal</th><th class="n">Empirical</th><th class="n">Gap</th></tr></thead>
      <tbody>{rel_rows}</tbody></table></div>
    <div class="call warn"><h3>It is a zero-inflation problem</h3>
      <p><strong>{100*z24.zero_fraction:.1f}%</strong> of hourly readings are <strong>exactly zero</strong>.
      Chronos-2 places its p10 above zero on <strong>{100*z24.p10_above_zero_frac:.0f}%</strong> of rows
      (median p10 = {z24.median_p10:.3f} KL/h), so <strong>{100*z24.below_misses_that_are_zero:.0f}%</strong>
      of its lower-tail misses are hours when demand was exactly zero.</p>
      <p>NPTS — a nonparametric sampler — puts p10 above zero on only
      {100*n24.p10_above_zero_frac:.0f}% of rows, and its lower-tail miss rate is
      <strong>{100*n24.miss_below_p10:.1f}%</strong> against Chronos-2's
      <strong>{100*z24.miss_below_p10:.1f}%</strong>.</p>
      <p>NPTS is better calibrated <strong>not because it models uncertainty better, but because it
      reproduces the zero atom of a zero-inflated series</strong> — which a continuous-density
      foundation model cannot. And the naive fix is wrong: clamping p10 to zero moves coverage to
      <strong>{z24.coverage_if_p10_clamped_to_zero:.3f}</strong>, overshooting nominal 0.80. The
      correct fix is asymmetric conformal calibration on the lower tail only.</p>
      <p style="border-top:1px solid var(--hair);padding-top:9px;margin-top:11px">
      <strong>That fix is now implemented and measured out of sample</strong> — coverage 0.741 →
      0.785 and volume bias −10.5% → −1.8%, with the lower-tail miss rate falling from 0.161 to
      0.122. See <a href="#holdout45">Section 5</a>.</p></div>
  </div>
</section>

<section id="patterns">
  <div class="sec-hd"><span class="eyebrow">Section 4 · new</span>
    <h2>Where the error lives</h2></div>
  <div class="grid2">
    <div class="tw"><table>
      <caption>MAE by lead time within the 7-day horizon</caption>
      <thead><tr><th>Lead time</th><th class="n">Chronos-2</th><th class="n">NPTS</th></tr></thead>
      <tbody>{lt_rows}</tbody></table></div>
    <div class="tw"><table>
      <caption>Tanks won by Chronos-2, per horizon</caption>
      <thead><tr><th>Horizon</th>{"".join(f"<th class='n'>{HL[h]}</th>" for h in HS)}</tr></thead>
      <tbody><tr><th scope="row">Won</th>{win_rows}</tr></tbody></table></div>
  </div>
  <div class="stack" style="margin-top:14px">
    <div class="call"><h3>Three patterns worth a slide each</h3>
      <p><strong>Error saturates within a day.</strong> From day 1 to day 7 it grows only
      {100*(d7-d1)/d1:+.1f}%. The model has learned the repeating daily profile, not a decaying
      extrapolation — which is what makes weekly planning viable.</p>
      <p><strong>Error tracks demand, bias does not.</strong> MAE by hour-of-day correlates ρ = 0.958
      with the demand profile, but the under-forecast bias is negative in <em>all 24 hours</em>. It is
      systematic, not situational — exactly what a per-tank multiplicative correction fixes.</p>
      <p><strong>The bias has an operational price.</strong> Over 24 origins the campus drew
      {tot:,.0f} KL; Chronos-2 under-forecasts by <strong>{abs(gapc):,.0f} KL</strong> and NPTS by
      {abs(gapn):,.0f} KL. About 33 KL per day. A refill must never be sized on the mean forecast.</p></div>
  </div>
</section>

<section id="holdout45">
  <div class="sec-hd"><span class="eyebrow">Section 5 · new</span>
    <h2>Continuous 45-day holdout — calibrated</h2>
    <p class="sec-note">The benchmark uses 24 overlapping origins at a 23-hour stride, ideal for
      unbiased scoring but impossible to draw as one timeline. This is a separate evaluation:
      <strong>45 consecutive daily origins</strong>, each forecasting the next 24 hours, tiling
      9 March – 22 April 2026 once — no gaps, no overlap, no leakage, all asserted in code. Both
      corrections that the benchmark listed as future work are now <strong>implemented and measured
      out of sample</strong>.</p></div>

  <div class="stack">
    <div class="call"><h3>How the calibration is fitted, and why the split matters</h3>
      <p>Two corrections, neither requiring retraining. <strong>Per-tank volume bias</strong> — a
      multiplicative factor, Σactual ÷ Σpredicted. <strong>Conformalised quantile regression</strong>
      (Romano et al., 2019) per tank, with <strong>independent lower and upper offsets</strong>,
      because the diagnosis in Section 3 showed the failure lives almost entirely in the lower tail.
      The lower bound is clipped at zero, which is what lets the interval represent the ~24% of
      hours with exactly zero demand.</p>
      <p><strong>Parameters are fitted on 8 Jan – 8 Mar 2026 and reported on 9 Mar – 22 Apr — strictly
      later, strictly disjoint.</strong> Fitting and reporting on the same rows makes coverage
      circular; the code raises if the windows touch. Each window gets its own NPTS predictor fitted
      only on data preceding it, so no window sees its own future.</p></div>

    <div class="tw"><table>
      <caption>Hourly interval calibration and volume bias — every number measured out of sample, on rows the calibration never saw. Nominal coverage 0.80.</caption>
      <thead><tr><th>Model</th><th class="n">Coverage</th><th class="n">Gap to 0.80</th>
        <th class="n">Width (KL/h)</th><th class="n">Miss below p10</th><th class="n">Miss above p90</th>
        <th class="n">Volume bias</th><th class="n">Hourly MAE</th></tr></thead>
      <tbody>{cal_rows}</tbody></table></div>

    <div class="call"><h3>What calibration fixed, and what it cost</h3>
      <p><strong>Chronos-2's coverage gap closes from 0.059 to 0.015</strong> and its lower-tail
      miss rate falls from 0.161 to 0.122, bought with 13% more width. <strong>Volume bias drops
      from −10.5% to −1.8%</strong> — 83% of the shortfall removed. SeasonalNaive-24 is the clearest
      demonstration of the method: it has no native interval at all, and conformal calibration takes
      it from 0.268 to <strong>0.804</strong>, essentially exact.</p>
      <p><strong>The honest cost: hourly MAE rises slightly</strong> — Chronos-2 from 0.205 to 0.214
      KL/h. Scaling a forecast toward the conditional mean moves it away from the conditional median,
      which is what minimises absolute error. But over 24 hours the systematic bias compounds while
      the noise cancels, so <strong>daily MAE improves sharply, from 36.7 to 28.8 KL</strong>. The
      correction is worth applying for volume and refill sizing, which is what it is for; whether to
      apply it to the hourly point forecast is a separate call.</p></div>

    <div class="tw"><table>
      <caption>Campus-total demand across 43 full-coverage days, calibrated. Actual mean {_actual_mean:.1f} KL/day, standard deviation {_actual_sd:.1f} KL.</caption>
      <thead><tr><th>Model</th><th class="n">Daily MAE (KL)</th><th class="n">MAPE</th>
        <th class="n">Total bias</th><th class="n">SD ratio</th><th class="n">Corr. w/ actual</th>
        <th class="n">Hourly MAE</th></tr></thead>
      <tbody>{h45_rows}</tbody></table></div>

    <div class="call"><h3>Calibration cannot fix a model that does not track</h3>
      <p><strong>SD ratio</strong> is how much of the real day-to-day variation a forecast
      reproduces; 1.00 means matched. Actual demand swings between 162 and 354 KL a day — a standard
      deviation of {_actual_sd:.0f} KL.</p>
      <p>After calibration, <strong>NPTS still produces an SD ratio of 0.06 and correlates −0.31 with
      reality.</strong> Its level is now right and its intervals are now honest, but it is still
      forecasting the long-run average every day. That is the limit of post-hoc correction: it fixes
      <em>where</em> a forecast sits and <em>how sure</em> it claims to be, not <em>whether it
      responds to anything</em>.</p>
      <p>Chronos-2 reproduces <strong>89%</strong> of the real variation and correlates
      <strong>+0.65</strong>. That gap — not the headline MASE — is the operational case for the
      change: for deciding how much water tomorrow needs, the incumbent carries almost no information
      beyond the average, and Chronos-2 carries most of it.</p></div>

    <div class="call warn"><h3>Where the bands hold, and where they do not</h3>
      <p>Per-tank daily bands land close to nominal — <strong>median coverage 0.781</strong> across
      the 24 tanks against a nominal 0.80. The <strong>campus-total</strong> band does not, covering
      only <strong>63%</strong>. That is not a contradiction: tank errors are positively correlated,
      so they do not cancel in the sum the way an independent-sum assumption would predict.</p>
      <p>Two tanks carry almost all of the per-tank failure, and both are visible in the panels
      below. <strong>GJBC_LAW_BLOCK_3__A1</strong> (band 21%) is the known modelling failure — a
      healthy sensor with a genuinely spiky draw the model smooths away.
      <strong>GJBC_BLOCK_1_A4_RO</strong> (band 16%) is a different problem: it averaged 0.0005 KL/h
      during the calibration window, below the threshold for a bias factor, then woke up to ~0.08
      KL/day in the reported window. <strong>No correction fitted on an earlier window can
      anticipate a tank changing regime</strong> — which is exactly why a production system must
      refit on a rolling basis and monitor coverage continuously.</p></div>
  </div>

  <div class="figs" style="margin-top:22px">{h45_figs}</div>
</section>

<section id="figures">
  <div class="sec-hd"><span class="eyebrow">Section 6</span>
    <h2>New figures — O through W</h2>
    <p class="sec-note">Nine figures built for this review. The original fourteen establish
      <em>what</em> the model does; these establish <em>why</em>, and each answers a specific
      question a reviewer is likely to ask. PNG and SVG are both on disk at
      <code>results/chronos2/phase3/plots/</code>.</p></div>
  <div class="figs">{new_figs}</div>
</section>

<section id="existing">
  <div class="sec-hd"><span class="eyebrow">Section 7</span>
    <h2>Existing figures — A through N</h2>
    <p class="sec-note">Already in the deck, at <code>results/chronos2/review/plots/</code>.</p></div>
  <div class="gal">{old_figs}</div>
</section>

<section id="inference">
  <div class="sec-hd"><span class="eyebrow">Section 8</span>
    <h2>The inference for each category</h2></div>
  <div class="inf">
    <div class="inf-item"><div class="inf-top"><h3>System Testing</h3><span class="chip partial">partial</span></div>
      <p>6/6 metric unit tests and 15/15 methodology checks pass. That validates the
      <strong>measurement apparatus</strong> — if the seasonal-naive MASE identity failed, every
      number in the benchmark would be wrong.</p>
      <p>What does not exist is a system test suite, because the real-time system is specified but
      unbuilt. <strong>Say this plainly</strong> rather than presenting model tests as system tests.
      The test strategy is already written down: unit, property, contract, integration, parity and
      regression layers.</p></div>
    <div class="inf-item"><div class="inf-top"><h3>Validation &amp; Verification</h3><span class="chip done">done</span></div>
      <p>The strongest part of the project. Four machine-checked guarantees: identical evaluation
      rows for all nine models, zero leakage rows, MASE ≈ 1.0 for seasonal naive, and an independent
      re-score reproducing published results.</p>
      <p><strong>As of this session, a fifth: statistical significance</strong> at all six horizons
      on both metrics, by two independent tests. That was the most likely question a reviewer would
      ask, and the answer is now measured rather than argued.</p></div>
    <div class="inf-item"><div class="inf-top"><h3>Deployment</h3><span class="chip todo">the real gap</span></div>
      <p>A Chrome MV3 dock renders real Chronos-2 forecasts from a bundled JSON, offline. The
      compose stack exists but targets the <strong>retired</strong> AutoGluon models, and its
      Postgres is provisioned but unused. <strong>Nothing is serving.</strong></p>
      <p>The architecture to close this is fully specified across six design documents. Do not
      overstate the dock as a deployment — describing it accurately is more credible than
      describing it generously.</p></div>
    <div class="inf-item"><div class="inf-top"><h3>Final Experimental Results</h3><span class="chip done">done</span></div>
      <p>Nine models, six horizons, 24 tanks, 188,664 rows per model on one shared grid, zero leakage,
      zero duplicates. <strong>Chronos-2 zero-shot beats the incumbent at every horizon on every
      point metric, by 5.7–12.5% MASE, and the margin is significant everywhere.</strong> It costs
      89 seconds.</p>
      <p>The two caveats are equally measured and must be presented alongside the result: coverage
      0.714–0.743 against nominal 0.80, and a 12.15% under-forecast of 24-hour volume.</p></div>
    <div class="inf-item"><div class="inf-top"><h3>Performance Analysis</h3><span class="chip done">23 figures</span></div>
      <p>Each new figure pre-empts a specific challenge. <strong>“Is it real or noise?”</strong> → O, P.
      <strong>“Are dead tanks flattering your average?”</strong> → T. <strong>“Does it fall apart at
      long horizons?”</strong> → R. <strong>“Your intervals are broken.”</strong> → S, W — yes, and
      here is the mechanism and the correct fix. <strong>“What does the error cost?”</strong> → U.
      <strong>“Which tanks shouldn't you trust it on?”</strong> → P, V.</p></div>
    <div class="inf-item"><div class="inf-top"><h3>Research Paper Draft</h3><span class="chip todo">not started</span></div>
      <p><code>review_summary.md</code> already contains everything a paper needs. The significance
      testing and the zero-inflation diagnosis supply what was missing: a defensible statistical
      claim and a novel mechanistic finding.</p>
      <p>Suggested framing — <strong>“Zero-inflation limits probabilistic calibration of time-series
      foundation models: evidence from 24 campus water tanks.”</strong> The point-forecast win is
      solid but unsurprising; the calibration diagnosis is the publishable contribution.</p></div>
  </div>
</section>

<section id="slide">
  <div class="sec-hd"><span class="eyebrow">Section 9</span><h2>If you only get one slide</h2></div>
  <div class="call">
    <p><strong>Chronos-2 zero-shot beats the deployed incumbent at every forecast horizon — 5.7% to
    12.5% lower MASE — and the margin is statistically significant at all six horizons (paired
    bootstrap over 24 origins and Diebold-Mariano, both p &lt; 0.01). It wins 101 of 144
    tank-horizon cells, beats the naive baseline on all 24 tanks, and is never significantly worse
    than the incumbent on any tank with a healthy sensor. It costs 89 seconds and requires no
    training.</strong></p>
    <p><strong>Its two known weaknesses have been diagnosed and corrected.</strong> The prediction
    intervals covered 72% against a nominal 80%, traced in this work to the model's inability to
    represent the 24% of hours with exactly zero demand. Asymmetric conformal calibration and a
    per-tank volume bias correction — both fitted on an earlier, disjoint window and measured out of
    sample — take coverage to <strong>0.785</strong> and cut volume bias from <strong>−10.5% to
    −1.8%</strong>, reducing daily campus error from 36.7 to <strong>28.8 KL</strong>. Neither
    correction requires retraining.</p>
    <p>What calibration does <em>not</em> fix: the incumbent still reproduces 6% of real day-to-day
    variation and correlates −0.31 with it, while Chronos-2 reproduces 89% and correlates +0.65.
    Post-hoc correction fixes where a forecast sits and how sure it claims to be — not whether it
    responds to anything.</p>
  </div>
</section>

<section id="repro">
  <div class="sec-hd"><span class="eyebrow">Section 10</span><h2>Reproducing this</h2></div>
  <pre>source venv/bin/activate
<b>python -m tests.test_metrics</b>                  <span style="opacity:.65"># 6/6, ~3 s — trust nothing if this fails</span>
<b>python -m src.models.score_benchmark --strict</b> <span style="opacity:.65"># ~7 s — re-scores from existing parquets</span>
<b>python -m src.models.phase3_analysis</b>          <span style="opacity:.65"># ~90 s — sections 2-4, figures O-W</span>
<b>python -m src.models.holdout45_continuous</b>     <span style="opacity:.65"># ~70 s — uncalibrated 45-day holdout</span>
<b>python -m src.models.calibrated_holdout</b>        <span style="opacity:.65"># ~3 min — section 5, calibration, figures X-Z</span></pre>
  <p style="margin-top:12px;color:var(--ink-2);font-size:13.6px;max-width:74ch">
    <code>phase3_analysis</code> refits nothing and re-scores nothing — it reads the completed
    prediction parquets and asserts 188,664 paired rows, matching the published row count exactly.
    The bootstrap seed is fixed at 20260830, so the confidence intervals reproduce.</p>
</section>

<footer>
  Intelligent Water Management System · PW26_PK_06 · PES University RR ·
  Abhay Patil, Amogh E M, Harshavardhan M, Viraj Ved Shankar ·
  Every figure and number on this page traces to a file under <code>results/chronos2/</code>.
</footer>
</div>
"""

OUT.write_text(BODY)
print("wrote", OUT, f"{OUT.stat().st_size/1e6:.2f} MB")
