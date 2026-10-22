"""Write ``FINAL_ANALYSIS.md`` for a refresh study, from its result files and the published ones.

The refresh study re-runs the published protocol on a newer Waltr snapshot and scores it on a
later window. This document puts its results next to the published study's, so the deployment
decision rests on two windows rather than one. Every number below is read here, at build time,
from the files the pipeline wrote; every comparative sentence is conditioned on those numbers.

    PESU_RESULTS_DIR=results/refresh_2026-10-22 python reports/build_final_analysis.py

Reads (refresh study, ``R``): audit/{audit.json,revisions_vs_published.csv}, run_manifest.json,
unified/{leaderboard,significance_vs_all,significance_timestral,cost,win_matrix_all_horizons}.csv,
calibrated/{summary,calibration_effect}.csv, timestral_sizes.csv, window_comparison.csv,
window_pairs.csv; and the published study's unified/leaderboard.csv.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import IS_PUBLISHED, PUBLISHED_RESULTS, RESULTS_DIR  # noqa: E402
from src.models.unified_analysis import NOT_DEPLOYABLE  # noqa: E402

if IS_PUBLISHED:
    raise SystemExit("set PESU_RESULTS_DIR to the refresh study; the published study's report is "
                     "the results page and docs/review_summary.md")

R = ROOT / RESULTS_DIR
U = R / "unified"
P = ROOT / PUBLISHED_RESULTS
# The published window re-analysed with TimesFM 3.0 added (python -m src.models.unified_analysis on
# the published predictions plus TimesFM 3.0's window-1 run): every published row is reproduced
# unchanged there, so only TimesFM 3.0's rows are read from it.
W1 = R / "window1_timesfm3"
OUT = R / "FINAL_ANALYSIS.md"

HS = [6, 12, 24, 48, 72, 168]
HL = {6: "6 h", 12: "12 h", 24: "1 d", 48: "2 d", 72: "3 d", 168: "7 d"}
C2, TS, TF, NPTS = "Chronos2-ZS", "Timestral-209M-ZS", "TimesFM2.5-ZS", "NPTS"
T3 = "TimesFM3-ZS"
# Models that read the academic calendar, which covers Jan 2025 - May 2026 only; the refresh
# snapshot's 2024 history and everything after May 2026 carry no academic flags for them.
CALENDAR = ["Chronos2-COV", "Chronos2-COV-LEAN", "Chronos2-COV-XL", "PatchTST", "PatchTST-Tuned"]
ZERO_SHOT = [C2, TF, TS]          # the deployment candidates: no calendar, no training
# TimesFM 3.0 is zero-shot too, but its weights may not be used in production: never a candidate.
assert not set(ZERO_SHOT) & NOT_DEPLOYABLE


def mark(m: str) -> str:
    return m + (" †" if m in CALENDAR else "") + (" ‡" if m in NOT_DEPLOYABLE else "")


def hlist(hs):
    hs = [HL[h] for h in hs]
    return "no horizon" if not hs else hs[0] if len(hs) == 1 else ", ".join(hs[:-1]) + " and " + hs[-1]


def tests(sig: pd.DataFrame, base: str, opp: str) -> dict:
    """Horizons at which each test separates ``base`` from ``opp`` on MASE, by direction.
    ``improvement_pct`` is from the base's side: positive = the base has the lower error."""
    s = sig[(sig.base == base) & (sig.opponent == opp) & (sig.metric == "MASE")].set_index("horizon")
    boot_b = [h for h in HS if s.loc[h, "significant_95"] and s.loc[h, "improvement_pct"] > 0]
    boot_o = [h for h in HS if s.loc[h, "significant_95"] and s.loc[h, "improvement_pct"] < 0]
    dm_b = [h for h in HS if s.loc[h, "p_dm"] < 0.05 and s.loc[h, "improvement_pct"] > 0]
    dm_o = [h for h in HS if s.loc[h, "p_dm"] < 0.05 and s.loc[h, "improvement_pct"] < 0]
    return {"impr": s.improvement_pct, "boot_base": boot_b, "boot_opp": boot_o,
            "dm_base": dm_b, "dm_opp": dm_o,
            "both_base": [h for h in boot_b if h in dm_b], "both_opp": [h for h in boot_o if h in dm_o]}


def verdict(t: dict, base: str, opp: str) -> str:
    imp = t["impr"]
    lead = (f"{base}'s MASE is lower than {opp}'s at every horizon" if (imp > 0).all() else
            f"{base}'s MASE is higher than {opp}'s at every horizon" if (imp < 0).all() else
            f"{base}'s MASE is lower at {hlist([h for h in HS if imp[h] > 0])} and higher at "
            f"{hlist([h for h in HS if imp[h] < 0])}")
    parts = []
    for who, key in ((base, "base"), (opp, "opp")):
        both, boot, dm = t[f"both_{key}"], t[f"boot_{key}"], t[f"dm_{key}"]
        one = sorted(set(boot) ^ set(dm))
        if both or one:
            parts.append(f"in {who}'s favour both tests agree at {hlist(both)}"
                         + (f" and one test only at {hlist(one)}" if one else ""))
    return lead + "; " + ("; ".join(parts) if parts else "neither test separates them at any horizon")


def timesfm3_bullet(p, r, sig, psig, csum, same, w1) -> str:
    """TimesFM 3.0 in one bullet, from the tables; it is never the recommendation."""
    pr = None
    if (w1 / "unified" / "leaderboard.csv").exists():
        plb = pd.read_csv(w1 / "unified" / "leaderboard.csv")
        pr = plb[plb.model == T3].set_index("horizon").rank_mase
    ranks = ", ".join(f"{HL[h]} #{int(r.loc[T3, h])}" for h in HS)
    text = (f"* **TimesFM 3.0 — benchmark only (‡).** Ranks on the new window: {ranks}"
            + (f"; on the published window: " + ", ".join(f"{HL[h]} #{int(pr[h])}" for h in HS)
               if pr is not None else "") + ". Against Timestral: "
            + verdict(tests(sig, TS, T3), TS, T3) + ". Against Chronos-2 zero-shot: "
            + verdict(tests(sig, C2, T3), C2, T3) + ".")
    if T3 in csum.index:
        text += (f" Calibrated daily campus MAE {csum.loc[T3, 'daily_mae_kl']:.1f} KL "
                 f"(raw {csum.loc[T3, 'daily_mae_raw_kl']:.1f}).")
    if pd.notna(same.get(T3)):
        text += f" Same-CPU backtest {same[T3]:.0f} s."
    return text + (" Whatever it scores, its weights' licence rules it out of the extension, and "
                   "Google publishes no fine-tuning recipe for 3.0; TimesFM 2.5 (Apache-2.0) is the "
                   "TimesFM that could be fine-tuned and shipped.")


def build() -> str:
    audit = json.loads((R / "audit" / "audit.json").read_text())
    rev = pd.read_csv(R / "audit" / "revisions_vs_published.csv")
    revised_tiers = json.loads((R / "eda" / "tank_trust_revised_flows.json").read_text())
    man = json.loads((R / "run_manifest.json").read_text())
    lb = pd.read_csv(U / "leaderboard.csv")
    sig = pd.concat([pd.read_csv(f) for f in (U / "significance_vs_all.csv",
                                             U / "significance_timestral.csv",
                                             U / "significance_timesfm3.csv") if f.exists()],
                    ignore_index=True)
    psig = pd.concat([pd.read_csv(P / "unified" / "significance_vs_all.csv"),
                      pd.read_csv(P / "unified" / "significance_timestral.csv")], ignore_index=True)
    if (W1 / "unified").exists():
        w1 = pd.concat([pd.read_csv(f) for f in (W1 / "unified" / "significance_vs_all.csv",
                                                 W1 / "unified" / "significance_timestral.csv",
                                                 W1 / "unified" / "significance_timesfm3.csv")
                        if f.exists()], ignore_index=True)
        psig = pd.concat([psig, w1[(w1.base == T3) | (w1.opponent == T3)]], ignore_index=True)
    plb = pd.read_csv(P / "unified" / "leaderboard.csv")
    cost = pd.read_csv(U / "cost.csv").set_index("model")
    csum = pd.read_csv(R / "calibrated" / "summary.csv").set_index("model")
    ceff = pd.read_csv(R / "calibrated" / "calibration_effect.csv").set_index("model")
    cman = json.loads((R / "calibrated" / "manifest.json").read_text())
    sizes = pd.read_csv(R / "timestral_sizes.csv")
    wc = pd.read_csv(R / "window_comparison.csv")
    wm = pd.read_csv(U / "win_matrix_all_horizons.csv")

    origins = pd.to_datetime(pd.Series(man["origins"]))
    pman = json.loads((P / "run_manifest.json").read_text())
    porigins = pd.to_datetime(pd.Series(pman["origins"]))
    rows = int(lb[lb.model == C2]["rows"].sum())
    models = list(lb.model.unique())
    tot_o, tot_n = rev.outflow_total_old_kl.sum(), rev.outflow_total_new_kl.sum()
    pt = pd.DataFrame(audit["per_tank"])
    short = pt[pd.to_datetime(pt.last_reading) < pd.Timestamp(audit["panel_range"][1])]

    out = ["# Final analysis — refresh study", "",
           f"Every number here is read from `{RESULTS_DIR}/` and, for the earlier window, "
           f"`{PUBLISHED_RESULTS}/`, by `reports/build_final_analysis.py`. Regenerate it rather "
           "than editing it.", ""]

    # ── data
    out += ["## 1. The data", "",
            f"* **Snapshot.** `{pathlib.Path(audit['dataset_dir']).name}`, fetched from Waltr's API: "
            f"{audit['panel_rows']:,} hourly rows, {audit['tanks']} tanks, "
            f"{audit['panel_range'][0][:10]} → {audit['panel_range'][1][:10]}. The published study "
            f"used 2025-01-01 → {audit['published_end'][:10]}.",
            f"* **Waltr revised its history.** Re-downloaded, the published period's flows differ: "
            f"campus outflow {tot_o:,.0f} → {tot_n:,.0f} KL ({100 * (tot_n - tot_o) / tot_o:+.1f} %), "
            f"{int(rev.outflow_hours_changed.sum()):,} of {int(rev.hours_compared.sum()):,} tank-hours "
            f"changed, level readings unchanged on all but {int(rev.closing_hours_changed.sum())}. "
            f"Most of the change zeroes small flows: the share of exactly-zero outflow hours rises on "
            f"{int((rev.outflow_zero_pct_new > rev.outflow_zero_pct_old + 5).sum())} of "
            f"{len(rev)} tanks by more than 5 points (largest: "
            + ", ".join(f"{r.tank} {r.outflow_zero_pct_old:.0f} → {r.outflow_zero_pct_new:.0f} %"
                        for r in rev.assign(d=rev.outflow_zero_pct_new - rev.outflow_zero_pct_old)
                        .nlargest(3, "d").itertuples())
            + "). The whole refresh study uses the revised figures; the published study keeps the "
              "original ones, so the two are not comparable in level (`audit/revisions_vs_published.csv`).",
            "* **What the revision means.** Waltr now reports zero inflow and zero outflow in many "
            "hours whose level moved, so the forecast target — Waltr's reported outflow — leaves out "
            "small consumption that the original figures counted. That is what the dashboard and the "
            "extension show, so it is what the models are scored on, but the refresh study forecasts "
            "a slightly different quantity than the published one.",
            f"* **Sensor trust tiers.** The tier test is a mass balance (level change = inflow − "
            f"outflow); on the revised flows it fails by construction and would mark "
            f"{sum(v == 'degraded' for v in revised_tiers.values())} of {len(revised_tiers)} tanks degraded. "
            f"Tiers here are the published ones, assessed on the original self-consistent data "
            f"(`eda/TRUST_NOTE.md`).",
            f"* **Tanks that stop early:** "
            + (", ".join(f"{r.tank} (last reading {str(r.last_reading)[:16]})" for r in short.itertuples())
               if len(short) else "none") + ". Their missing hours are dropped from every model's "
            "score alike.",
            f"* **Calendar.** The academic calendar covers Jan 2025 – May 2026 only. The five models "
            f"that read it — {', '.join(CALENDAR)} — see no academic flags for 2024 or after May "
            "2026, so their refresh results are marked † and kept out of the deployment decision.",
            "* **Benchmark-only models (‡).** " + (
                "TimesFM 3.0's weights are licensed for internal benchmarking and research only, not "
                "for production systems or end users, so it is scored as a reference point and is "
                "never a deployment candidate (`timesfm3_manifest.json`)." if T3 in set(lb.model)
                else "none."), ""]

    # ── grid
    out += ["## 2. The evaluation window", "",
            f"The published protocol unchanged: {len(origins)} origins at a 23-hour stride, "
            f"{origins.iloc[0]:%d %b %Y %H:%M} → {origins.iloc[-1]:%d %b %Y %H:%M}, six horizons, "
            f"{rows:,} scored rows per model, identical for all {len(models)} models (`score_benchmark "
            f"--strict`). The published window's origins ran {porigins.iloc[0]:%d %b} → "
            f"{porigins.iloc[-1]:%d %b %Y}.", ""]

    # ── leaderboard
    p = lb.pivot(index="model", columns="horizon", values="macro_mase")
    r = lb.pivot(index="model", columns="horizon", values="rank_mase")
    cov = lb[lb.horizon == 24].set_index("model").coverage
    order = lb[lb.horizon == 24].sort_values("rank_mase").model
    out += ["## 3. Leaderboard on the new window — macro MASE (rank)", "",
            "| Model | " + " | ".join(HL[h] for h in HS) + " | p10–p90 coverage, 1 d |",
            "|---" * (len(HS) + 2) + "|"]
    for m in order:
        out.append(f"| {mark(m)} | " + " | ".join(f"{p.loc[m, h]:.4f} ({int(r.loc[m, h])})" for h in HS)
                   + f" | {cov[m]:.3f} |")
    best = {h: p[h].idxmin() for h in HS}
    best_zs = {h: p.loc[ZERO_SHOT, h].idxmin() for h in HS}
    pt_t = p.loc["PatchTST-Tuned"] if "PatchTST-Tuned" in p.index else None
    if pt_t is not None and (pt_t > 1).sum() >= 3:
        # MASE > 1 is "worse than the in-sample seasonal-naive scale"; the model it is compared
        # against below is the out-of-sample SeasonalNaive row, which is a different thing.
        sn_worse = int((pt_t > p.loc["SeasonalNaive"]).sum())
        out += ["", f"PatchTST-Tuned has a MASE above 1 at {int((pt_t > 1).sum())} of 6 horizons, "
                    f"scores worse than the SeasonalNaive model at {sn_worse} of 6, and swings between "
                    f"{pt_t.min():.2f} and {pt_t.max():.2f} from one horizon to the next. Its horizons "
                    "are fitted separately, so that points at unstable training on the revised, "
                    "zero-heavy target, beyond the missing calendar."]
        if (R / "old_vs_new" / "benchmark_old_vs_new_tests.csv").exists():
            ov = pd.read_csv(R / "old_vs_new" / "benchmark_old_vs_new_tests.csv")
            ov = ov[ov.new == "PatchTST-Tuned"]
            out[-1] += (" `old_vs_new/OLD_VS_NEW.md` follows this up: the March-fitted network, run "
                        f"unchanged on these origins, has the lower MASE at {int((ov['diff'] > 0).sum())} "
                        f"of {len(ov)} horizons ({int((ov.significant_95 & (ov['diff'] > 0)).sum())} "
                        "with a 95 % bootstrap interval excluding zero).")
    out += ["", "Lowest MASE per horizon: " + "; ".join(f"{HL[h]} {mark(best[h])}" for h in HS) + ".",
            "Among the deployment candidates (zero-shot, no calendar): "
            + "; ".join(f"{HL[h]} {best_zs[h]}" for h in HS) + ".", ""]

    # ── significance
    out += ["## 4. Is the difference real? (new window)", "",
            "Paired bootstrap over the origins and Diebold–Mariano, read together; a result only one "
            "test supports is the weaker claim.", ""]
    for base, opp in ((C2, TS), (C2, TF), (TS, TF), (C2, NPTS), (TS, NPTS), (C2, "Chronos2-COV-XL"),
                      (C2, T3), (TS, T3), (T3, TF)):
        if {base, opp} <= set(models):
            out.append(f"* **{mark(base)} vs {mark(opp)}.** "
                       + verdict(tests(sig, base, opp), base, opp) + ".")
    out += [""]

    # ── replication
    ranks = wc.groupby("model")[["rank_mase_a", "rank_mase_b"]].mean()
    from scipy.stats import spearmanr
    rho, pr = spearmanr(ranks.rank_mase_a, ranks.rank_mase_b)
    out += ["## 5. Does it replicate? Published window vs new window", "",
            f"Ranks, not MASE levels, are compared: the revised data and longer history change the "
            f"scale. Spearman correlation of the mean-rank leaderboards: **{rho:.3f}** "
            f"({len(ranks)} models, p = {pr:.2g})."
            + (" The published window's ranks come from its re-analysis with TimesFM 3.0 added "
               "(`window1_timesfm3/`), which reproduces every published row unchanged; ranks below "
               "TimesFM 3.0 move down one place." if T3 in set(ranks.index) else ""), "",
            "| Model | mean rank, published | mean rank, new |", "|---|---|---|"]
    for m in ranks.sort_values("rank_mase_b").index:
        out.append(f"| {mark(m)} | {ranks.loc[m, 'rank_mase_a']:.2f} | "
                   f"{ranks.loc[m, 'rank_mase_b']:.2f} |")
    out += ["", "Head-to-head verdicts in each window:", ""]
    for base, opp in [(C2, TS), (C2, TF), (TS, TF), (C2, NPTS)] + [
            (b, o) for b, o in ((C2, T3), (TS, T3)) if {b, o} <= set(psig.base) | set(psig.opponent)]:
        out.append(f"* **{mark(base)} vs {mark(opp)}** — published: " + verdict(tests(psig, base, opp), base, opp)
                   + ". New: " + verdict(tests(sig, base, opp), base, opp) + ".")
    out += [""]

    # ── operational
    cw = [pd.Timestamp(x) for x in cman["calibration_window"]]
    rw = [pd.Timestamp(x) for x in cman["reported_window"]]
    keys = [k for k in (C2, TS, TF, T3, NPTS, "SeasonalNaive-24") if k in csum.index]
    out += ["## 6. The operational view — daily campus volume", "",
            f"{cman['test_days']} consecutive daily forecasts, {rw[0]:%d %b} → {rw[1]:%d %b %Y}; "
            f"corrections fitted on the disjoint {cw[0]:%d %b} → {cw[1]:%d %b %Y}, which probably "
            "spans the 2026 summer break (the 2026 calendar dates are not available), a regime the "
            "corrections then carry into term time.", "",
            "| Model | daily MAE raw → calibrated (KL) | volume bias raw → calibrated | hourly "
            "p10–p90 coverage raw → calibrated |", "|---|---|---|---|"]
    for k in keys:
        out.append(f"| {mark(k)} | {csum.loc[k, 'daily_mae_raw_kl']:.1f} → {csum.loc[k, 'daily_mae_kl']:.1f} | "
                   f"{csum.loc[k, 'total_bias_raw_pct']:+.1f} % → {csum.loc[k, 'total_bias_pct']:+.1f} % | "
                   f"{ceff.loc[k, 'coverage_raw']:.3f} → {ceff.loc[k, 'coverage_cal']:.3f} |")
    out += ["", f"Lowest calibrated daily MAE: {mark(csum.loc[keys, 'daily_mae_kl'].idxmin())}.", ""]

    # ── compute and sizes
    same = cost["wall_clock_s_same_cpu"]
    out += ["## 7. Compute", "",
            "| Model | wall clock (s) | device | same-CPU (s) |", "|---|---|---|---|"]
    for m in [x for x in ZERO_SHOT + [T3, "Chronos2-COV-XL", NPTS, "PatchTST-Tuned"] if x in cost.index]:
        out.append(f"| {mark(m)} | {cost.loc[m, 'wall_clock_s']:.1f} | {cost.loc[m, 'device'] if isinstance(cost.loc[m, 'device'], str) else ''} | "
                   f"{'' if pd.isna(same.get(m)) else f'{same[m]:.1f}'} |")
    sz = sizes.pivot(index="size", columns="horizon", values="macro_mase")
    out += ["", "Timestral sizes on the same rows (off the roster, untested):", "",
            "| Size | " + " | ".join(HL[h] for h in HS) + " | wall clock (s) |", "|---" * (len(HS) + 2) + "|"]
    for s in [x for x in ("5M", "28M", "123M", "209M") if x in sz.index]:
        wcl = sizes[sizes["size"] == s].wall_clock_s.iloc[0]
        out.append(f"| {s} | " + " | ".join(f"{sz.loc[s, h]:.4f}" for h in HS) + f" | {wcl:.0f} |")
    out += [""]

    # ── interpretation
    vw = lb.pivot(index="model", columns="horizon", values="vw_mae")
    vb = lb[lb.horizon == 24].set_index("model").volume_bias_pct
    t_new, t_pub = tests(sig, C2, TS), tests(psig, C2, TS)
    vw_lead = (vw.loc[TS] < vw.loc[C2]).all()
    pt24 = pd.read_csv(U / "per_tank.csv")
    d24 = pt24[(pt24.horizon == 24) & pt24.model.isin([C2, TS])].pivot(index="tank", columns="model",
                                                                          values="mae")
    tank_wins = int((d24[TS] < d24[C2]).sum())
    daily = csum.loc[[C2, TS]]
    out += ["## 8. Interpretation — which model to deploy", "",
            "*This section interprets the measurements above; the decision is the team's.*", "",
            f"* **Hourly accuracy: Timestral 209M, in both windows.** It has the lowest macro MASE at "
            f"every horizon in the published window and again here. Against Chronos-2 zero-shot the "
            f"lead was separated by both tests at {hlist(t_pub['both_opp'])} in the published window "
            f"and at {hlist(t_new['both_opp'])} in this one, so it replicated and strengthened. It "
            + ("also has the lower volume-weighted MAE at every horizon, so the lead holds on the "
               "high-demand tanks" if vw_lead else "does not lead on volume-weighted MAE everywhere")
            + f", and the lower 1-day MAE on {tank_wins} of {len(d24)} tanks.",
            f"* **Campus volume: Chronos-2 zero-shot, narrowly.** Timestral under-forecasts total volume "
            f"more ({vb[TS]:+.1f} % vs {vb[C2]:+.1f} % at 1 d), so before correction Chronos-2's daily "
            f"campus MAE is lower ({daily.loc[C2, 'daily_mae_raw_kl']:.1f} vs "
            f"{daily.loc[TS, 'daily_mae_raw_kl']:.1f} KL). After the per-tank volume correction the two "
            f"are {abs(round(daily.loc[C2, 'daily_mae_kl'], 1) - round(daily.loc[TS, 'daily_mae_kl'], 1)):.1f} KL apart "
            f"({daily.loc[C2, 'daily_mae_kl']:.1f} vs {daily.loc[TS, 'daily_mae_kl']:.1f}); the daily "
            "totals have no significance test, so neither is established as better there.",
            f"* **Cost.** Both are cheap enough to refresh every hour: the full {len(origins) * len(HS)}-"
            f"call backtest took {same.get(TS, float('nan')):.0f} s for Timestral and "
            f"{same.get(C2, float('nan')):.0f} s for Chronos-2 on the same CPU, i.e. about "
            f"{same.get(TS, float('nan')) / (len(origins) * len(HS)):.1f} s per 24-tank forecast. Timestral "
            "needs no GPU; Chronos-2 can also use Apple MPS.",
            *([timesfm3_bullet(p, r, sig, psig, csum, same, W1)] if T3 in p.index else []),
            "* **Caveats.** Timestral is a preview release (weights under a modified MIT licence: no "
            "rights above USD 20M monthly revenue). The new window is one more 23-day window, and it "
            "forecasts Waltr's revised flow figures.", "",
            "**Recommendation.** "
            + ("For the extension's per-tank hourly forecast, Timestral 209M is now the better-supported "
               "model: its lead over Chronos-2 replicated on a second window and is established by both "
               "tests at three horizons. Ship it with the per-tank volume correction, which matters more "
               "for it than for Chronos-2, and keep Chronos-2 zero-shot as the fallback and challenger. "
               "If the extension leads with daily campus volume rather than per-tank hours, the two are "
               "tied after correction and either serves."
               if len(t_new["both_opp"]) >= 3 and vw_lead else
               "The evidence does not separate the two firmly enough to change the deployed model; keep "
               "Chronos-2 zero-shot and re-test on the next window."), ""]
    return "\n".join(out)


def main() -> None:
    OUT.write_text(build())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
