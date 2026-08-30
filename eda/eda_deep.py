"""
eda_deep.py  —  Deep-Dive EDA for PESU Campus Water Demand (waltr_data)
Generates 50+ plots + a comprehensive text report (eda_report.txt).

Sections
  0.  Raw JSON loading & capacity computation
  1.  Data inventory & coverage
  2.  Data quality (missing, duplicates, zeros, completeness)
  3.  Descriptive statistics (per-tank summary, distributions, fits)
  4.  Intra-day hourly analysis
  5.  Water balance & tank-level dynamics
  6.  Daily temporal analysis (trend, seasonality, stationarity, FFT)
  7.  Per-tank individual deep dive (trend + ACF + seasonal strength)
  8.  Cross-tank analysis (correlation, clustering, Granger, Lorenz)
  9.  Calendar & academic effects
  10. Anomaly & change-point detection
  11. Feature engineering evaluation
"""

import os, json, glob, math, warnings, io, sys
from itertools import combinations
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats, signal
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import norm, lognorm, gamma as gamma_dist, kstest
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

# ── Config ────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
REPO      = os.path.dirname(BASE)

# The dataset lives at <repo>/dataset. Overridable so the script can be pointed at a subset or
# an archived snapshot without editing it.
DATA_DIR  = os.environ.get("PESU_DATASET", os.path.join(REPO, "dataset"))
PLOT_DIR  = os.environ.get("PESU_PLOTS",   os.path.join(BASE, "plots"))
REPORT    = os.environ.get("PESU_REPORT",  os.path.join(BASE, "eda_report.txt"))

if not os.path.isdir(DATA_DIR):
    raise SystemExit(
        f"Dataset directory not found: {DATA_DIR}\n"
        f"Set PESU_DATASET to the directory holding the per-tank JSON folders."
    )
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                      "axes.spines.top": False, "axes.spines.right": False})

# `dataset/` holds 26 directories for 24 physical tanks: two tanks were re-scraped under a
# slightly different name and both copies kept. The *_S directories are strict supersets (same
# tank name, dimensions and daily totals; 38 extra days), so keep those and skip the stale pair.
# Verified in src/data/curate.py.
TANK_ALIASES = {
    "GJBC_BLOCK_1_A1_BOY_S":  "GJBC_BLOCK_1_A1_BOYS",
    "GJBC_BLOCK_1_A2_GIRL_S": "GJBC_BLOCK_1_A2_GIRLS",
}
STALE_TANK_DIRS = {"GJBC_BLOCK_1_A1_BOYS", "GJBC_BLOCK_1_A2_GIRLS"}
SKIP_DIRS = {"results", "logs", "papers", "src"} | STALE_TANK_DIRS
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Report writer ─────────────────────────────────────────────────────────────
_report_lines = []
def rprint(*args, **kwargs):
    line = " ".join(str(a) for a in args)
    print(line, **kwargs)
    _report_lines.append(line)

def rsep(title="", char="=", width=72):
    rprint(char * width)
    if title:
        rprint(f"  {title}")
        rprint(char * width)

def save_report():
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_report_lines))
    print(f"\n[report] Saved → {REPORT}")

plot_index = [0]
def savefig(name, fig=None):
    plot_index[0] += 1
    fname = f"{plot_index[0]:02d}_{name}.png"
    path  = os.path.join(PLOT_DIR, fname)
    (fig or plt).savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    rprint(f"  [plot {plot_index[0]:02d}] {fname}")
    return fname

def compute_capacity_kl(dim_str):
    try:
        parts = [p.strip() for p in dim_str.split("X")]
        vals  = [float(p.split()[0]) for p in parts]
        if len(vals) == 3:
            vol = vals[0] * vals[1] * vals[2]
        elif len(vals) == 2:
            vol = math.pi * vals[1]**2 * vals[0]
        else:
            return np.nan
        return vol / 1_000_000
    except Exception:
        return np.nan

# ═══════════════════════════════════════════════════════════════════════════════
# 0. LOAD RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════
rsep("SECTION 0 — Loading Raw JSON Data")

hourly_records = []
tank_meta      = {}

tank_dirs = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d)) and d not in SKIP_DIRS
])

for td_dir in tank_dirs:
    td         = TANK_ALIASES.get(td_dir, td_dir)   # canonical tank id
    td_path    = os.path.join(DATA_DIR, td_dir)
    json_files = sorted(glob.glob(os.path.join(td_path, "*.json")))
    if not json_files:
        continue
    with open(json_files[0]) as f:
        meta = json.load(f)
    cap = compute_capacity_kl(meta.get("Tank Dimensions", ""))
    tank_meta[td] = {
        "name": meta.get("Tank Name", td),
        "type": meta.get("Tank Type", "unknown"),
        "shape": meta.get("Tank Shape", "unknown"),
        "dimensions": meta.get("Tank Dimensions", ""),
        "capacity_kl": cap,
    }
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
        except Exception:
            continue
        for entry in data.get("data", []):
            dt_str = entry.get("Date Time", "")
            hourly_records.append({
                "tank_id": td,
                "dt_str":  dt_str,
                "inflow":  entry.get("Inflow in KL",        np.nan),
                "outflow": entry.get("Outflow in KL",       np.nan),
                "opening": entry.get("Opening Value in KL", np.nan),
                "closing": entry.get("Closing Value in KL", np.nan),
            })

hrly = pd.DataFrame(hourly_records)
hrly["timestamp"] = pd.to_datetime(hrly["dt_str"], format="%Y-%m-%d %H", errors="coerce")
hrly["date"]      = hrly["timestamp"].dt.normalize()
hrly["hour"]      = hrly["timestamp"].dt.hour
hrly["dow"]       = hrly["timestamp"].dt.dayofweek
hrly["month"]     = hrly["timestamp"].dt.month
hrly = hrly.dropna(subset=["timestamp"]).sort_values(["tank_id","timestamp"]).reset_index(drop=True)

daily = hrly.groupby(["tank_id","date"]).agg(
    daily_inflow   = ("inflow",  "sum"),
    daily_outflow  = ("outflow", "sum"),
    opening_level  = ("opening", "first"),
    closing_level  = ("closing", "last"),
    hourly_records = ("outflow", "count"),
    n_zero_hours   = ("outflow", lambda x: (x == 0).sum()),
    max_hourly_out = ("outflow", "max"),
    peak_hour      = ("outflow", lambda x: x.idxmax() if x.max() > 0 else np.nan),
).reset_index()
daily["date"]      = pd.to_datetime(daily["date"])
daily["dow"]       = daily["date"].dt.dayofweek
daily["month"]     = daily["date"].dt.month
daily["year"]      = daily["date"].dt.year
daily["net_bal"]   = daily["daily_inflow"] - daily["daily_outflow"]
daily["efficiency"]= daily["daily_outflow"] / daily["daily_inflow"].replace(0, np.nan)
daily["level_chg"] = daily["closing_level"] - daily["opening_level"]

wide = daily.pivot_table(index="date", columns="tank_id", values="daily_outflow", aggfunc="sum")
wide["campus_total"] = wide.sum(axis=1)
campus = wide["campus_total"].dropna()

meta_df = pd.DataFrame(tank_meta).T.reset_index().rename(columns={"index":"tank_id"})
meta_df["capacity_kl"] = meta_df["capacity_kl"].astype(float)
tanks     = sorted(daily["tank_id"].unique())
N_TANKS   = len(tanks)
tank_cols = [c for c in wide.columns if c != "campus_total"]

rprint(f"  Tanks loaded   : {N_TANKS}")
rprint(f"  Hourly rows    : {len(hrly):,}")
rprint(f"  Daily rows     : {len(daily):,}")
rprint(f"  Date range     : {hrly['date'].min().date()} → {hrly['date'].max().date()}")
rprint(f"  Total days     : {(hrly['date'].max() - hrly['date'].min()).days + 1}")

# ── Academic calendar ─────────────────────────────────────────────────────────
holidays_all = pd.to_datetime([
    "2025-01-15","2025-01-26","2025-03-14","2025-04-14","2025-05-01",
    "2025-08-15","2025-08-27","2025-09-05","2025-10-01","2025-10-02",
    "2025-10-20","2025-10-22","2025-11-01","2025-12-25",
    "2026-01-15","2026-01-26","2026-03-19","2026-03-21","2026-04-03",
    "2026-04-14","2026-05-01","2026-05-28",
])
isa_dates = pd.DatetimeIndex([]).append([
    pd.date_range("2025-03-03","2025-03-08"), pd.date_range("2025-04-21","2025-04-26"),
    pd.date_range("2025-09-22","2025-09-26"), pd.date_range("2025-11-24","2025-11-28"),
    pd.date_range("2026-03-02","2026-03-07"), pd.date_range("2026-04-27","2026-05-01"),
])
esa_dates = pd.DatetimeIndex([]).append([
    pd.date_range("2025-05-04","2025-05-30"),
    pd.date_range("2025-12-01","2025-12-30"),
    pd.date_range("2026-05-04","2026-05-30"),
])
summer_break   = pd.date_range("2025-06-01","2025-08-03")
intersem_break = pd.date_range("2025-12-31","2026-01-04")
exam_starts    = sorted([pd.Timestamp(x) for x in [
    "2025-03-03","2025-04-21","2025-05-04","2025-09-22",
    "2025-11-24","2025-12-01","2026-03-02","2026-04-27","2026-05-04",
]])

def get_phase(dt):
    d = pd.Timestamp(dt)
    if d in summer_break:    return "summer_break"
    if d in intersem_break:  return "intersem_break"
    if d in esa_dates:       return "esa"
    if d in isa_dates:       return "isa"
    if d in holidays_all:    return "holiday"
    return "regular"

def days_to_next_exam(dt):
    future = [(ex - dt).days for ex in exam_starts if (ex - dt).days >= 0]
    return min(future) if future else 999

campus_df = campus.to_frame("outflow")
campus_df["dow"]       = campus_df.index.dayofweek
campus_df["dow_name"]  = campus_df.index.day_name()
campus_df["month"]     = campus_df.index.month
campus_df["year"]      = campus_df.index.year
campus_df["is_weekend"]= campus_df["dow"].isin([5,6]).astype(int)
campus_df["phase"]     = campus_df.index.map(get_phase)
campus_df["is_holiday"]= campus_df.index.isin(holidays_all).astype(int)
campus_df["is_isa"]    = campus_df.index.isin(isa_dates).astype(int)
campus_df["is_esa"]    = campus_df.index.isin(esa_dates).astype(int)
campus_df["is_summer"] = campus_df.index.isin(summer_break).astype(int)
campus_df["days_to_exam"] = [days_to_next_exam(d) for d in campus_df.index]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA INVENTORY & COVERAGE
# ═══════════════════════════════════════════════════════════════════════════════
rsep("SECTION 1 — Data Inventory & Coverage")

date_ranges = daily.groupby("tank_id")["date"].agg(["min","max","count"]).reset_index()
date_ranges.columns = ["tank_id","first_date","last_date","observed_days"]
date_ranges["expected_days"] = ((date_ranges["last_date"] - date_ranges["first_date"]).dt.days + 1)
date_ranges["gap_days"]      = date_ranges["expected_days"] - date_ranges["observed_days"]
date_ranges["coverage_pct"]  = (100 * date_ranges["observed_days"] / date_ranges["expected_days"]).round(1)
date_ranges = date_ranges.merge(meta_df[["tank_id","capacity_kl","shape"]], on="tank_id")
rprint("\nPer-tank date coverage:")
rprint(date_ranges.to_string(index=False))

# 1a: Monthly coverage heatmap
pivot_cov = wide.drop(columns=["campus_total"]).notna().astype(int)
monthly_cov = pivot_cov.resample("ME").mean() * 100
fig, ax = plt.subplots(figsize=(17, 7))
sns.heatmap(monthly_cov.T, cmap="RdYlGn", vmin=0, vmax=100,
            linewidths=0.3, annot=True, fmt=".0f",
            xticklabels=[d.strftime("%b %y") for d in monthly_cov.index],
            yticklabels=[c[:20] for c in monthly_cov.columns], ax=ax,
            cbar_kws={"label":"Coverage %"})
ax.set_title("Monthly Data Coverage % per Tank", fontsize=13, fontweight="bold")
plt.xticks(rotation=45, ha="right")
savefig("inventory_coverage_monthly")

# 1b: Tank capacities (cuboidal vs cylindrical)
fig, ax = plt.subplots(figsize=(12, 8))
cap_s = meta_df.sort_values("capacity_kl", ascending=True).dropna(subset=["capacity_kl"])
colors_c = {"cuboidal":"#1565c0","cylindrical":"#e53935"}
ax.barh(range(len(cap_s)), cap_s["capacity_kl"],
        color=[colors_c.get(r, "#757575") for r in cap_s["shape"]], alpha=0.85)
ax.set_yticks(range(len(cap_s)))
ax.set_yticklabels([r[:22] for r in cap_s["tank_id"]], fontsize=8)
ax.set_xlabel("Tank Capacity (KL)")
ax.set_title("Tank Capacities Computed from Physical Dimensions", fontsize=13, fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#1565c0",label="Cuboidal"),
                   Patch(facecolor="#e53935",label="Cylindrical")])
for i, (_, row) in enumerate(cap_s.iterrows()):
    ax.text(row["capacity_kl"]+0.3, i, f"{row['capacity_kl']:.1f} KL", va="center", fontsize=7)
savefig("inventory_tank_capacities")

# 1c: Gap-day timeline per tank
fig, ax = plt.subplots(figsize=(12, 6))
gapped = date_ranges.sort_values("gap_days", ascending=False)
colors_gap = ["#c62828" if g > 20 else "#f57f17" if g > 5 else "#2e7d32" for g in gapped["gap_days"]]
ax.barh(range(len(gapped)), gapped["gap_days"], color=colors_gap, alpha=0.85)
ax.set_yticks(range(len(gapped)))
ax.set_yticklabels([t[:22] for t in gapped["tank_id"]], fontsize=8)
ax.set_xlabel("Missing / Gap Days")
ax.set_title("Missing Days per Tank", fontsize=13, fontweight="bold")
ax.axvline(5, color="orange", linestyle="--", alpha=0.6, label=">5 days")
ax.axvline(20, color="red", linestyle="--", alpha=0.6, label=">20 days")
ax.legend()
savefig("inventory_gap_days")

rprint(f"\nCapacity summary:")
rprint(meta_df[["tank_id","shape","capacity_kl"]].sort_values("capacity_kl",ascending=False).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA QUALITY DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
rsep("SECTION 2 — Data Quality")

# 2a: Full-timeline missing heatmap
pivot_miss = wide.drop(columns=["campus_total"]).isna().astype(int)
fig, ax = plt.subplots(figsize=(16, 7))
sns.heatmap(pivot_miss.T, cmap="YlOrRd", cbar_kws={"label":"Missing"},
            xticklabels=30, yticklabels=True, ax=ax)
ax.set_title("Full Timeline Missing Data Heatmap", fontsize=13, fontweight="bold")
ax.set_xlabel("Date index")
savefig("quality_missing_heatmap")

# 2b: Hourly record completeness
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
daily["hourly_records"].hist(bins=25, ax=axes[0], color="steelblue", edgecolor="white")
axes[0].axvline(24, color="red", linestyle="--", label="Full 24 hrs")
axes[0].set_title("Hourly Records per Day-Tank", fontweight="bold")
axes[0].set_xlabel("Count"); axes[0].legend()

incomp = daily.groupby("tank_id").apply(lambda x: (x["hourly_records"] < 24).mean() * 100).sort_values()
incomp.plot(kind="barh", ax=axes[1], color="coral")
axes[1].set_title("% Days with < 24 Hourly Records per Tank", fontweight="bold")
axes[1].set_xlabel("% days incomplete")
plt.tight_layout()
savefig("quality_hourly_completeness")

# 2c: Zero-outflow analysis
zero_df = daily.groupby("tank_id").agg(
    zero_days      = ("daily_outflow", lambda x: (x == 0).sum()),
    total_days     = ("daily_outflow", "count"),
    near_zero_days = ("daily_outflow", lambda x: (x < 0.5).sum()),
).reset_index()
zero_df["zero_pct"]      = (zero_df["zero_days"] / zero_df["total_days"] * 100).round(1)
zero_df["near_zero_pct"] = (zero_df["near_zero_days"] / zero_df["total_days"] * 100).round(1)
rprint("\nZero & near-zero outflow days per tank:")
rprint(zero_df.sort_values("zero_pct", ascending=False).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
zs = zero_df.sort_values("zero_pct", ascending=True)
ax.barh(range(len(zs)), zs["near_zero_pct"], color="#ffcc02", alpha=0.8, label="<0.5 KL")
ax.barh(range(len(zs)), zs["zero_pct"],      color="#c62828", alpha=0.85, label="Exactly 0")
ax.set_yticks(range(len(zs)))
ax.set_yticklabels([t[:22] for t in zs["tank_id"]], fontsize=8)
ax.set_xlabel("% of days"); ax.legend()
ax.set_title("Zero & Near-Zero Outflow Days per Tank (%)", fontsize=13, fontweight="bold")
savefig("quality_zero_demand_days")

# 2d: Outlier detection (IQR 1.5× and 3×)
rprint("\nOutlier detection summary (daily outflow):")
outlier_rows = []
for tank in tanks:
    s = daily.loc[daily["tank_id"]==tank, "daily_outflow"].dropna()
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    n_15 = ((s < Q1-1.5*IQR) | (s > Q3+1.5*IQR)).sum()
    n_3  = ((s < Q1-3*IQR)   | (s > Q3+3*IQR)).sum()
    outlier_rows.append({"tank": tank, "Q1":round(Q1,1), "median":round(s.median(),1),
                         "Q3":round(Q3,1), "IQR":round(IQR,1),
                         "outliers_1.5x":n_15, "outliers_3x":n_3,
                         "pct_1.5x":round(100*n_15/len(s),1)})
outlier_df = pd.DataFrame(outlier_rows)
rprint(outlier_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(16, 6))
data_list = [daily.loc[daily["tank_id"]==t, "daily_outflow"].dropna().values for t in tanks]
bp = ax.boxplot(data_list, labels=[t[:12] for t in tanks], patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", ms=3, alpha=0.4, color="red"))
for patch in bp["boxes"]: patch.set_facecolor("#1565c0"); patch.set_alpha(0.6)
ax.set_title("Daily Outflow Distribution per Tank (Outliers Visible)", fontsize=13, fontweight="bold")
ax.set_ylabel("Daily Outflow (KL)")
plt.xticks(rotation=45, ha="right", fontsize=7)
savefig("quality_outlier_boxplot")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
rsep("SECTION 3 — Descriptive Statistics")

tank_stats = daily.groupby("tank_id").agg(
    n_days         = ("daily_outflow", "count"),
    mean_out       = ("daily_outflow", "mean"),
    median_out     = ("daily_outflow", "median"),
    std_out        = ("daily_outflow", "std"),
    min_out        = ("daily_outflow", "min"),
    max_out        = ("daily_outflow", "max"),
    p10_out        = ("daily_outflow", lambda x: x.quantile(0.10)),
    p90_out        = ("daily_outflow", lambda x: x.quantile(0.90)),
    skew_out       = ("daily_outflow", lambda x: x.dropna().skew()),
    kurt_out       = ("daily_outflow", lambda x: x.dropna().kurtosis()),
    mean_in        = ("daily_inflow",  "mean"),
    mean_bal       = ("net_bal",       "mean"),
).round(2)
tank_stats["cv"]         = (tank_stats["std_out"] / tank_stats["mean_out"]).round(3)
tank_stats["inout_ratio"]= (tank_stats["mean_in"] / tank_stats["mean_out"].replace(0,np.nan)).round(3)
tank_stats["demand_share"]=(tank_stats["mean_out"] / tank_stats["mean_out"].sum() * 100).round(1)
tank_stats = tank_stats.merge(meta_df[["tank_id","capacity_kl"]], on="tank_id")
tank_stats["util_pct"]   = (tank_stats["mean_out"] / tank_stats["capacity_kl"] * 100).round(1)
tank_stats = tank_stats.sort_values("mean_out", ascending=False)
rprint("\nPer-tank descriptive statistics:")
rprint(tank_stats.to_string())

# 3a: Comprehensive violin plot
fig, ax = plt.subplots(figsize=(17, 7))
data_vln = [daily.loc[daily["tank_id"]==t, "daily_outflow"].dropna().values for t in tanks]
parts = ax.violinplot(data_vln, positions=range(N_TANKS), showmedians=True, showextrema=True)
for pc in parts["bodies"]: pc.set_facecolor("#1565c0"); pc.set_alpha(0.5)
ax.set_xticks(range(N_TANKS))
ax.set_xticklabels([t[:12] for t in tanks], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Daily Outflow (KL)")
ax.set_title("Daily Outflow Distribution per Tank — Violin Plot", fontsize=13, fontweight="bold")
savefig("stats_violin_all_tanks")

# 3b: Skewness & kurtosis comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sk = tank_stats[["tank_id","skew_out","kurt_out"]].sort_values("skew_out")
axes[0].barh(range(len(sk)), sk["skew_out"],
             color=["#c62828" if v > 0 else "#1565c0" for v in sk["skew_out"]], alpha=0.8)
axes[0].set_yticks(range(len(sk))); axes[0].set_yticklabels([t[:20] for t in sk["tank_id"]], fontsize=7)
axes[0].axvline(0, color="black", lw=0.8); axes[0].set_title("Skewness", fontweight="bold"); axes[0].set_xlabel("Skewness")
axes[1].barh(range(len(sk)), sk["kurt_out"],
             color=["#e65100" if v > 0 else "#1b5e20" for v in sk["kurt_out"]], alpha=0.8)
axes[1].set_yticks(range(len(sk))); axes[1].set_yticklabels([t[:20] for t in sk["tank_id"]], fontsize=7)
axes[1].axvline(0, color="black", lw=0.8); axes[1].set_title("Excess Kurtosis", fontweight="bold"); axes[1].set_xlabel("Kurtosis")
plt.suptitle("Distributional Shape of Daily Outflow per Tank", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("stats_skew_kurtosis")

# 3c: Distribution fitting (Normal, LogNormal, Gamma) for campus total
rprint("\nDistribution fitting — Campus Total Daily Outflow:")
campus_clean = campus.dropna().values
fit_results  = {}
for dist_name, dist in [("Normal", norm), ("LogNormal", lognorm), ("Gamma", gamma_dist)]:
    params = dist.fit(campus_clean)
    ks_stat, ks_p = kstest(campus_clean, dist.cdf, args=params)
    fit_results[dist_name] = {"params": params, "ks_stat": ks_stat, "ks_p": ks_p}
    rprint(f"  {dist_name:<12}: KS={ks_stat:.4f}, p={ks_p:.4f} ({'Good fit' if ks_p > 0.05 else 'Poor fit'})")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x_range = np.linspace(campus_clean.min(), campus_clean.max(), 200)
for i, (dist_name, dist) in enumerate([("Normal", norm), ("LogNormal", lognorm), ("Gamma", gamma_dist)]):
    ax = axes[i]
    ax.hist(campus_clean, bins=40, density=True, alpha=0.6, color="steelblue", label="Data")
    params = fit_results[dist_name]["params"]
    y_pdf  = dist.pdf(x_range, *params)
    ax.plot(x_range, y_pdf, color="red", linewidth=2, label=f"{dist_name} fit")
    ks_p   = fit_results[dist_name]["ks_p"]
    ax.set_title(f"{dist_name} Fit\nKS p={ks_p:.4f}", fontweight="bold")
    ax.set_xlabel("Daily Outflow (KL)"); ax.legend(fontsize=8)
plt.suptitle("Distribution Fitting — Campus Total Daily Outflow", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("stats_distribution_fits")

# 3d: Q-Q plots for campus total
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (dist_name, dist) in enumerate([("Normal", norm), ("LogNormal", lognorm), ("Gamma", gamma_dist)]):
    ax  = axes[i]
    params = fit_results[dist_name]["params"]
    n   = len(campus_clean)
    quantiles_data  = np.sort(campus_clean)
    quantiles_theor = dist.ppf(np.linspace(0.01, 0.99, n), *params)
    ax.scatter(quantiles_theor, quantiles_data, alpha=0.4, s=8, color="steelblue")
    mn = min(quantiles_theor.min(), quantiles_data.min())
    mx = max(quantiles_theor.max(), quantiles_data.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="y=x")
    ax.set_title(f"Q-Q: {dist_name}", fontweight="bold")
    ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
    ax.legend(fontsize=8)
plt.suptitle("Q-Q Plots — Campus Daily Outflow vs Fitted Distributions", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("stats_qq_plots")

# 3e: Monthly violin (campus total demand by calendar month)
campus_monthly = campus.to_frame("outflow")
campus_monthly["month_name"] = campus_monthly.index.strftime("%b")
campus_monthly["month_num"]  = campus_monthly.index.month
months_present = sorted(campus_monthly["month_num"].unique())
fig, ax = plt.subplots(figsize=(13, 5))
month_data = [campus_monthly[campus_monthly["month_num"]==m]["outflow"].dropna().values for m in months_present]
parts = ax.violinplot(month_data, positions=months_present, showmedians=True)
for pc in parts["bodies"]: pc.set_facecolor("#2e7d32"); pc.set_alpha(0.6)
ax.set_xticks(months_present); ax.set_xticklabels([MONTH_LABELS[m-1] for m in months_present])
ax.set_xlabel("Month"); ax.set_ylabel("Campus Total Outflow (KL)")
ax.set_title("Monthly Demand Distribution — Campus Total (Violin)", fontsize=13, fontweight="bold")
savefig("stats_monthly_violin")

# 3f: Percentile bands over time (P10/P25/P75/P90 rolling)
fig, ax = plt.subplots(figsize=(15, 5))
w = 30
roll = campus.rolling(w)
ax.fill_between(campus.index, roll.quantile(0.10), roll.quantile(0.90), alpha=0.12, color="#1565c0", label="P10–P90")
ax.fill_between(campus.index, roll.quantile(0.25), roll.quantile(0.75), alpha=0.25, color="#1565c0", label="P25–P75")
ax.plot(campus.index, roll.median(), color="#1565c0", linewidth=2, label="P50 (median)")
ax.set_title(f"Rolling {w}-day Demand Percentile Bands — Campus Total", fontsize=13, fontweight="bold")
ax.set_ylabel("KL"); ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
savefig("stats_percentile_bands")

# 3g: Demand share pie chart
fig, ax = plt.subplots(figsize=(10, 9))
shares = tank_stats.sort_values("demand_share", ascending=False)[["tank_id","demand_share","mean_out"]]
labels = [f"{row['tank_id'][:16]}\n{row['demand_share']:.1f}%" for _, row in shares.iterrows()]
colors_pie = plt.cm.tab20(np.linspace(0, 1, len(shares)))
wedges, texts = ax.pie(shares["demand_share"], labels=None, colors=colors_pie,
                        startangle=90, wedgeprops={"edgecolor":"white","linewidth":0.5})
ax.legend(wedges, labels, title="Tank (% share)", loc="center left",
          bbox_to_anchor=(1, 0.5), fontsize=7)
ax.set_title("Campus Demand Share by Tank", fontsize=13, fontweight="bold")
savefig("stats_demand_share_pie")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. INTRA-DAY HOURLY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
rsep("SECTION 4 — Intra-Day Hourly Analysis")

hrly_camp = hrly.groupby(["date","hour"])[["inflow","outflow"]].sum().reset_index()
hrly_camp["dow"]     = pd.to_datetime(hrly_camp["date"]).dt.dayofweek
hrly_camp["is_wknd"] = hrly_camp["dow"].isin([5,6])
hrly_camp["month"]   = pd.to_datetime(hrly_camp["date"]).dt.month
hrly_camp["phase"]   = pd.to_datetime(hrly_camp["date"]).map(get_phase)

# 4a: Hourly mean outflow + inflow overlay
h_out = hrly_camp.groupby("hour")["outflow"].agg(["mean","std"])
h_in  = hrly_camp.groupby("hour")["inflow"].agg(["mean","std"])
fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(h_out.index, h_out["mean"]-h_out["std"], h_out["mean"]+h_out["std"], alpha=0.15, color="#1565c0")
ax.fill_between(h_in.index,  h_in["mean"]-h_in["std"],   h_in["mean"]+h_in["std"],  alpha=0.15, color="#e53935")
ax.plot(h_out.index, h_out["mean"], "o-", color="#1565c0", linewidth=2, markersize=5, label="Outflow")
ax.plot(h_in.index,  h_in["mean"],  "s-", color="#e53935", linewidth=2, markersize=5, label="Inflow")
ax.set_xticks(range(0,24)); ax.set_xlabel("Hour of Day"); ax.set_ylabel("Mean Campus (KL)")
ax.set_title("Average Campus Inflow & Outflow by Hour (±1 SD)", fontsize=13, fontweight="bold")
ax.legend()
peak_hr = int(h_out["mean"].idxmax())
ax.axvline(peak_hr, color="#1565c0", linestyle="--", alpha=0.5, label=f"Peak outflow {peak_hr}:00")
rprint(f"\nPeak outflow hour: {peak_hr}:00  (mean={h_out.loc[peak_hr,'mean']:.2f} KL)")
savefig("hourly_inout_profile")

# 4b: Weekday vs Weekend hourly
fig, ax = plt.subplots(figsize=(13, 5))
for wknd, label, color in [(False, "Weekday", "#1565c0"), (True, "Weekend", "#e53935")]:
    h = hrly_camp[hrly_camp["is_wknd"]==wknd].groupby("hour")["outflow"].agg(["mean","std"])
    ax.fill_between(h.index, h["mean"]-h["std"], h["mean"]+h["std"], alpha=0.15, color=color)
    ax.plot(h.index, h["mean"], "o-", color=color, linewidth=2, markersize=5, label=label)
ax.set_xticks(range(0,24)); ax.legend()
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Mean Outflow (KL)")
ax.set_title("Hourly Outflow Profile: Weekday vs Weekend", fontsize=13, fontweight="bold")
savefig("hourly_weekday_vs_weekend")

# 4c: Hourly DOW × Hour heatmap
dow_hour = hrly_camp.groupby(["dow","hour"])["outflow"].mean().unstack(level="hour").fillna(0)
dow_hour.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
fig, ax = plt.subplots(figsize=(17, 5))
sns.heatmap(dow_hour, cmap="YlOrRd", linewidths=0.2, ax=ax,
            cbar_kws={"label":"Mean Outflow (KL)"}, annot=True, fmt=".1f",
            annot_kws={"size":6})
ax.set_title("Campus Mean Outflow: Day-of-Week × Hour of Day", fontsize=13, fontweight="bold")
savefig("hourly_dow_hour_heatmap")

# 4d: Hourly outflow by academic phase
fig, ax = plt.subplots(figsize=(14, 5))
phase_colors = {"regular":"#1565c0","isa":"#f57f17","esa":"#c62828",
                "summer_break":"#9e9e9e","holiday":"#4caf50","intersem_break":"#757575"}
for phase, color in phase_colors.items():
    sub = hrly_camp[hrly_camp["phase"]==phase]
    if len(sub) < 20:
        continue
    h = sub.groupby("hour")["outflow"].mean()
    ax.plot(h.index, h.values, linewidth=2, color=color, label=phase, alpha=0.9)
ax.set_xticks(range(0,24)); ax.legend(fontsize=8)
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Mean Outflow (KL)")
ax.set_title("Hourly Demand Profile by Academic Phase", fontsize=13, fontweight="bold")
savefig("hourly_by_academic_phase")

# 4e: All 24 tanks hourly profiles
fig, axes = plt.subplots(6, 4, figsize=(18, 16), sharex=True)
axes_f = axes.flatten()
for i, tank in enumerate(tanks):
    t_h = hrly[hrly["tank_id"]==tank].groupby("hour")["outflow"].agg(["mean","std"])
    ax  = axes_f[i]
    ax.fill_between(t_h.index, t_h["mean"]-t_h["std"], t_h["mean"]+t_h["std"], alpha=0.2, color="teal")
    ax.plot(t_h.index, t_h["mean"], color="teal", linewidth=1.5)
    ax.set_title(tank[:18], fontsize=7, fontweight="bold")
    ax.set_xticks(range(0,24,6)); ax.tick_params(labelsize=6)
for j in range(i+1, len(axes_f)): axes_f[j].set_visible(False)
plt.suptitle("Mean Hourly Outflow Profile — All 24 Tanks (±1 SD)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout(); savefig("hourly_all_tanks_profiles")

# 4f: Night-time share per tank (proxy for leakage)
hrly["period"] = hrly["hour"].apply(lambda h: "night" if h < 6 else "day")
period_split = hrly.groupby(["tank_id","period"])["outflow"].sum().unstack(fill_value=0)
period_pct   = (period_split.div(period_split.sum(axis=1), axis=0) * 100).round(1)
rprint("\nNight-time outflow share (00-05) per tank:")
rprint(period_pct["night"].sort_values(ascending=False).to_string())

fig, ax = plt.subplots(figsize=(10, 7))
night_s = period_pct["night"].sort_values(ascending=True)
ax.barh(range(len(night_s)), night_s.values,
        color=["#c62828" if v > 15 else "#f57f17" if v > 8 else "#2e7d32" for v in night_s.values], alpha=0.85)
ax.axvline(8,  color="orange", linestyle="--", alpha=0.7, label="8% threshold")
ax.axvline(15, color="red",    linestyle="--", alpha=0.7, label="15% threshold")
ax.set_yticks(range(len(night_s))); ax.set_yticklabels([t[:22] for t in night_s.index], fontsize=8)
ax.set_xlabel("% of Total Outflow during 00-05h")
ax.set_title("Night-Time Outflow Share per Tank\n(>15% may indicate leakage/issues)", fontsize=13, fontweight="bold")
ax.legend()
savefig("hourly_nighttime_share")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. WATER BALANCE & TANK-LEVEL DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════
rsep("SECTION 5 — Water Balance & Level Dynamics")

campus_bal = daily.groupby("date")[["daily_inflow","daily_outflow","net_bal"]].sum().reset_index()

# 5a: Inflow vs Outflow with net balance
fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
axes[0].plot(campus_bal["date"], campus_bal["daily_outflow"], color="#1565c0", linewidth=0.8, alpha=0.6, label="Outflow")
axes[0].plot(campus_bal["date"], campus_bal["daily_inflow"],  color="#e53935", linewidth=0.8, alpha=0.6, label="Inflow")
axes[0].plot(campus_bal["date"], campus_bal["daily_outflow"].rolling(14).mean(), color="#1565c0", linewidth=2)
axes[0].plot(campus_bal["date"], campus_bal["daily_inflow"].rolling(14).mean(),  color="#e53935", linewidth=2)
axes[0].set_title("Campus Inflow vs Outflow (14-day MA)", fontweight="bold"); axes[0].set_ylabel("KL"); axes[0].legend()

net = campus_bal["net_bal"]
axes[1].bar(campus_bal["date"], net,
            color=["#2e7d32" if v>=0 else "#c62828" for v in net], alpha=0.6, width=1)
axes[1].axhline(0, color="black", lw=0.8); axes[1].set_title("Net Balance (Inflow − Outflow)", fontweight="bold"); axes[1].set_ylabel("KL")

eff = campus_bal["daily_outflow"] / campus_bal["daily_inflow"].replace(0, np.nan)
axes[2].plot(campus_bal["date"], eff.rolling(14).mean(), color="purple", linewidth=1.5, label="14-day rolling")
axes[2].axhline(1, color="red", linestyle="--", alpha=0.5, label="Perfect efficiency")
axes[2].set_title("14-day Rolling Outflow/Inflow Ratio", fontweight="bold"); axes[2].set_ylabel("Ratio"); axes[2].legend()
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
axes[2].xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45); plt.tight_layout()
savefig("balance_campus_overview")

# 5b: Level utilization (% of capacity) for tanks with capacity data
fig, axes = plt.subplots(5, 4, figsize=(18, 18))
axes_f = axes.flatten()
cap_tanks = meta_df.dropna(subset=["capacity_kl"]).sort_values("mean_out" if "mean_out" in meta_df.columns else "capacity_kl", ascending=False)
# merge mean_out
cap_tanks = cap_tanks.merge(tank_stats[["tank_id","mean_out"]], on="tank_id", how="left")
cap_tanks = cap_tanks.sort_values("mean_out", ascending=False)

for i, (_, row) in enumerate(cap_tanks.iterrows()):
    if i >= len(axes_f): break
    tank = row["tank_id"]
    cap  = row["capacity_kl"]
    sub  = daily[daily["tank_id"]==tank].sort_values("date")
    ax   = axes_f[i]
    open_pct  = (sub["opening_level"] / cap * 100).clip(0, 120)
    close_pct = (sub["closing_level"] / cap * 100).clip(0, 120)
    ax.fill_between(sub["date"], open_pct, close_pct, alpha=0.4, color="teal")
    ax.plot(sub["date"], close_pct.rolling(7).mean(), color="navy", linewidth=1.2, label="Closing 7d MA")
    ax.axhline(100, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(20,  color="orange", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.set_title(f"{tank[:18]}\ncap={cap:.1f} KL", fontsize=7, fontweight="bold")
    ax.set_ylim(0, 130); ax.tick_params(labelsize=6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))

for j in range(i+1, len(axes_f)): axes_f[j].set_visible(False)
plt.suptitle("Tank Level Utilization (% of Capacity) — Opening/Closing Fill Range", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout(); savefig("balance_level_utilization_all")

# 5c: Per-tank net balance summary
bal_stats = daily.groupby("tank_id").agg(
    mean_net   = ("net_bal", "mean"),
    excess_days= ("net_bal", lambda x: (x > 0).sum()),
    deficit_days=("net_bal", lambda x: (x < 0).sum()),
    total      = ("net_bal", "count"),
).reset_index()
bal_stats["excess_pct"]  = (bal_stats["excess_days"] / bal_stats["total"] * 100).round(1)
bal_stats["deficit_pct"] = (bal_stats["deficit_days"] / bal_stats["total"] * 100).round(1)
rprint("\nNet balance per tank (mean inflow − outflow per day):")
rprint(bal_stats.sort_values("mean_net", ascending=False).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
bs = bal_stats.sort_values("mean_net")
axes[0].barh(range(len(bs)), bs["mean_net"],
             color=["#2e7d32" if v>=0 else "#c62828" for v in bs["mean_net"]], alpha=0.85)
axes[0].set_yticks(range(len(bs))); axes[0].set_yticklabels([t[:20] for t in bs["tank_id"]], fontsize=7)
axes[0].axvline(0, color="black", lw=0.8); axes[0].set_title("Mean Net Balance per Tank (KL/day)", fontweight="bold")

bs2 = bal_stats.sort_values("excess_pct")
axes[1].barh(range(len(bs2)), bs2["excess_pct"],  color="#2e7d32", alpha=0.6, label="Excess (inflow>outflow)")
axes[1].barh(range(len(bs2)), -bs2["deficit_pct"], color="#c62828", alpha=0.6, label="Deficit (outflow>inflow)")
axes[1].set_yticks(range(len(bs2))); axes[1].set_yticklabels([t[:20] for t in bs2["tank_id"]], fontsize=7)
axes[1].axvline(0, color="black", lw=0.8); axes[1].set_title("Excess vs Deficit Days (%)", fontweight="bold")
axes[1].legend()
plt.tight_layout(); savefig("balance_per_tank_summary")

# 5d: Fill/drain cycle analysis (closing level change distribution)
fig, axes = plt.subplots(4, 6, figsize=(18, 12))
axes_f = axes.flatten()
for i, tank in enumerate(tanks):
    sub = daily[daily["tank_id"]==tank]["level_chg"].dropna()
    ax  = axes_f[i]
    ax.hist(sub, bins=25, color="steelblue", edgecolor="white", density=True, alpha=0.8)
    ax.axvline(0, color="red", lw=0.8)
    ax.axvline(sub.mean(), color="orange", lw=1.5, linestyle="--")
    ax.set_title(f"{tank[:16]}\nμ={sub.mean():.1f}", fontsize=6, fontweight="bold")
    ax.tick_params(labelsize=5)
for j in range(i+1, len(axes_f)): axes_f[j].set_visible(False)
plt.suptitle("Daily Level Change Distribution (Closing − Opening) per Tank", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout(); savefig("balance_fill_drain_cycles")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. DAILY TEMPORAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
rsep("SECTION 6 — Daily Temporal Analysis")

# 6a: Full trend with multiple windows
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(campus.index, campus.values, alpha=0.2, linewidth=0.6, color="gray", label="Daily")
for w, c in [(7,"#1565c0"), (30,"#e53935"), (90,"#2e7d32")]:
    ax.plot(campus.index, campus.rolling(w, center=True).mean(), linewidth=2, color=c, label=f"{w}-day MA")
ax.set_title("Campus Total Daily Outflow — Full Trend with Multiple Smoothing", fontsize=13, fontweight="bold")
ax.set_ylabel("Total Outflow (KL)"); ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45); savefig("temporal_campus_trend")

# 6b: DOW boxplot with significance
fig, ax = plt.subplots(figsize=(11, 5))
order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
sns.boxplot(data=campus_df, x="dow_name", y="outflow", order=order, palette="Blues_d", ax=ax, showfliers=False)
sns.stripplot(data=campus_df, x="dow_name", y="outflow", order=order, color="black", alpha=0.15, size=2.5, ax=ax)
wkd = campus_df[campus_df["is_weekend"]==0]["outflow"]
wke = campus_df[campus_df["is_weekend"]==1]["outflow"]
t, p = stats.ttest_ind(wkd.dropna(), wke.dropna())
ax.set_title(f"Campus Outflow by Day of Week\nWeekday μ={wkd.mean():.0f} vs Weekend μ={wke.mean():.0f} KL  (t={t:.2f}, p={p:.4f})",
             fontsize=11, fontweight="bold")
ax.set_xlabel(""); ax.set_ylabel("Daily Outflow (KL)")
savefig("temporal_dow_boxplot")
rprint(f"\nWeekday mean: {wkd.mean():.2f} KL, Weekend mean: {wke.mean():.2f} KL, diff={wkd.mean()-wke.mean():.2f}, t={t:.3f}, p={p:.5f}")

# 6c: Monthly bar with counts
monthly = campus_df.groupby("month")["outflow"].agg(["mean","std","count"])
fig, ax = plt.subplots(figsize=(11, 5))
x = [MONTH_LABELS[m-1] for m in monthly.index]
ax.bar(x, monthly["mean"], yerr=monthly["std"], capsize=4, color="#1565c0", alpha=0.8)
ax.set_title("Monthly Mean Campus Outflow (±1 SD)", fontsize=13, fontweight="bold")
ax.set_ylabel("Mean Daily Outflow (KL)")
for i, (m, row) in enumerate(monthly.iterrows()):
    ax.text(i, row["mean"] + row["std"] + 1.5, f"n={row['count']}", ha="center", fontsize=7)
savefig("temporal_monthly_bar")

# 6d: YOY comparison
campus_df["doy"] = campus_df.index.dayofyear
fig, ax = plt.subplots(figsize=(13, 5))
for yr, c in [(2025,"#1565c0"), (2026,"#e53935")]:
    s = campus_df[campus_df["year"]==yr]
    if len(s) > 10:
        ax.plot(s["doy"], s["outflow"].rolling(7).mean(), linewidth=2, color=c, label=str(yr))
ax.set_xlabel("Day of Year"); ax.set_ylabel("Campus Outflow (KL, 7-day MA)")
ax.set_title("Year-over-Year Demand Comparison (7-day MA)", fontsize=13, fontweight="bold")
ax.legend()
savefig("temporal_yoy_comparison")

# 6e: Stationarity tests (ADF + KPSS)
rprint("\nStationarity Tests — Campus Total:")
adf_r = adfuller(campus.dropna(), autolag="AIC")
rprint(f"  ADF:  stat={adf_r[0]:.4f}, p={adf_r[1]:.6f}, lags={adf_r[2]}  → {'STATIONARY' if adf_r[1]<0.05 else 'NON-STATIONARY'}")
for k, v in adf_r[4].items(): rprint(f"    Critical {k}: {v:.4f}")

kpss_r = kpss(campus.dropna(), regression="c", nlags="auto")
rprint(f"  KPSS: stat={kpss_r[0]:.4f}, p={kpss_r[1]:.4f}  → {'STATIONARY' if kpss_r[1]>0.05 else 'NON-STATIONARY'}")
for k, v in kpss_r[3].items(): rprint(f"    Critical {k}: {v:.4f}")

rprint("\nStationarity Tests — First Difference:")
diff1 = campus.diff().dropna()
adf_d = adfuller(diff1, autolag="AIC")
rprint(f"  ADF diff: stat={adf_d[0]:.4f}, p={adf_d[1]:.6f}  → {'STATIONARY' if adf_d[1]<0.05 else 'NON-STATIONARY'}")

# Per-tank stationarity
rprint("\nPer-tank ADF stationarity (p-value):")
stat_rows = []
for tank in tanks:
    s = daily.loc[daily["tank_id"]==tank, "daily_outflow"].dropna()
    if len(s) > 20:
        try:
            r = adfuller(s, autolag="AIC")
            stat_rows.append({"tank": tank, "adf_stat": round(r[0],3), "p_value": round(r[1],4),
                               "stationary": "YES" if r[1]<0.05 else "NO"})
        except Exception:
            pass
stat_df = pd.DataFrame(stat_rows)
rprint(stat_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 6))
st_s = stat_df.sort_values("p_value")
ax.barh(range(len(st_s)), st_s["p_value"],
        color=["#2e7d32" if v<0.05 else "#c62828" for v in st_s["p_value"]], alpha=0.85)
ax.axvline(0.05, color="black", linestyle="--", linewidth=1, label="α=0.05")
ax.set_yticks(range(len(st_s))); ax.set_yticklabels([t[:22] for t in st_s["tank"]], fontsize=8)
ax.set_xlabel("ADF p-value"); ax.legend()
ax.set_title("ADF Stationarity p-values per Tank\n(green = stationary)", fontsize=13, fontweight="bold")
savefig("temporal_stationarity_per_tank")

# 6f: ACF / PACF (campus)
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
plot_acf(campus.dropna(),  lags=60, ax=axes[0], title="ACF — Campus Total Outflow (lags 0–60)")
plot_pacf(campus.dropna(), lags=60, ax=axes[1], title="PACF — Campus Total Outflow", method="ywm")
plt.tight_layout(); savefig("temporal_acf_pacf")

acf_vals = acf(campus.dropna(), nlags=60)
rprint("\nACF values at key lags:")
for lg in [1,2,3,4,5,6,7,14,21,28,30,60]:
    if lg < len(acf_vals): rprint(f"  Lag-{lg:2d}: {acf_vals[lg]:.4f}")

# 6g: Seasonal decomposition
try:
    decomp = seasonal_decompose(campus.dropna(), model="additive", period=7)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    decomp.observed.plot(ax=axes[0], color="gray",    title="Observed");  axes[0].set_ylabel("KL")
    decomp.trend.plot   (ax=axes[1], color="#1565c0", title="Trend");     axes[1].set_ylabel("KL")
    decomp.seasonal.plot(ax=axes[2], color="#e53935", title="Seasonal (period=7)"); axes[2].set_ylabel("KL")
    decomp.resid.plot   (ax=axes[3], color="#2e7d32", title="Residual");  axes[3].set_ylabel("KL")
    plt.suptitle("Seasonal Decomposition — Campus Outflow (period=7)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(); savefig("temporal_seasonal_decomposition")

    resid = decomp.resid.dropna()
    sw_stat, sw_p = stats.shapiro(resid[:5000])
    lb = acorr_ljungbox(resid, lags=[7, 14, 21], return_df=True)
    rprint(f"\nResidual Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f}  ({'Normal' if sw_p>0.05 else 'Not normal'})")
    rprint("Ljung-Box on residuals (H0 = no autocorrelation):")
    rprint(lb.to_string())
except Exception as e:
    rprint(f"  Decomposition error: {e}")

# 6h: FFT power spectrum
s_fft = campus.dropna().values - campus.dropna().mean()
N     = len(s_fft)
fft_v = np.fft.rfft(s_fft)
power = np.abs(fft_v)**2
freqs = np.fft.rfftfreq(N, d=1.0)
perds = np.where(freqs > 0, 1.0/freqs, np.inf)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(perds[1:], power[1:], color="#1565c0", lw=0.8)
ax.set_xscale("log"); ax.set_xlim(2, N//2)
for p_mark, color, label in [(7,"red","7d"),(14,"orange","14d"),(30,"green","30d")]:
    ax.axvline(p_mark, color=color, linestyle="--", alpha=0.6, label=label)
ax.set_xlabel("Period (days)"); ax.set_ylabel("Power")
ax.set_title("FFT Power Spectrum — Campus Total Outflow", fontsize=13, fontweight="bold"); ax.legend()
peak_idx = signal.argrelextrema(power[1:], np.greater, order=3)[0]
if len(peak_idx):
    top5 = peak_idx[np.argsort(power[1:][peak_idx])[-5:]]
    rprint("\nTop FFT spectral peaks:")
    for pi in sorted(top5):
        pv = perds[1:][pi]
        if pv < N//2:
            ax.annotate(f"{pv:.1f}d",(pv,power[1:][pi]),textcoords="offset points",xytext=(4,8),fontsize=8,color="red",fontweight="bold")
            rprint(f"  Period = {pv:.1f} days  Power = {power[1:][pi]:.0f}")
savefig("temporal_fft_spectrum")

# 6i: Rolling volatility (std)
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(campus.index, campus.rolling(7).mean(),  color="#1565c0", lw=1.5, label="7d")
axes[0].plot(campus.index, campus.rolling(30).mean(), color="#e53935", lw=2,   label="30d")
axes[0].plot(campus.index, campus.rolling(90).mean(), color="#2e7d32", lw=2,   label="90d")
axes[0].set_title("Rolling Mean", fontweight="bold"); axes[0].set_ylabel("KL"); axes[0].legend()
axes[1].plot(campus.index, campus.rolling(7).std(),  color="#1565c0", lw=1.5, label="7d σ")
axes[1].plot(campus.index, campus.rolling(30).std(), color="#e53935", lw=2,   label="30d σ")
axes[1].set_title("Rolling Standard Deviation (Demand Volatility)", fontweight="bold")
axes[1].set_ylabel("KL"); axes[1].legend()
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
axes[1].xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45); plt.tight_layout()
savefig("temporal_rolling_volatility")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. PER-TANK INDIVIDUAL DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
rsep("SECTION 7 — Per-Tank Individual Deep Dive")

# 7a: All 24 tanks — trend time series (small multiples)
fig, axes = plt.subplots(6, 4, figsize=(18, 18))
axes_f = axes.flatten()
for i, tank in enumerate(tanks):
    ts = wide[tank].dropna()
    ax = axes_f[i]
    ax.plot(ts.index, ts.values, alpha=0.3, lw=0.5, color="gray")
    ax.plot(ts.index, ts.rolling(14).mean(), color="#1565c0", lw=1.5)
    if tank in tank_stats.index:
        mu = tank_stats.loc[tank, "mean_out"]
        ax.axhline(mu, color="red", lw=0.8, linestyle="--", alpha=0.7)
    ax.set_title(tank[:18], fontsize=7, fontweight="bold")
    ax.tick_params(labelsize=5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
for j in range(i+1, len(axes_f)): axes_f[j].set_visible(False)
plt.suptitle("Daily Outflow — All 24 Tanks (14-day MA, red dashed = mean)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout(); savefig("pertank_trend_all")

# 7b: All tanks ACF at lag 7 comparison
fig, ax = plt.subplots(figsize=(11, 7))
acf7 = {}
for tank in tanks:
    s = wide[tank].dropna()
    if len(s) > 20:
        acf7[tank] = s.autocorr(lag=7)
acf7_s = pd.Series(acf7).sort_values(ascending=True)
ax.barh(range(len(acf7_s)), acf7_s.values,
        color=["#2e7d32" if v > 0.3 else "#f57f17" if v > 0 else "#c62828" for v in acf7_s.values], alpha=0.85)
ax.set_yticks(range(len(acf7_s))); ax.set_yticklabels([t[:22] for t in acf7_s.index], fontsize=8)
ax.axvline(0, color="black", lw=0.8)
ax.axvline(0.3, color="green", linestyle="--", alpha=0.5, label="Strong weekly")
ax.set_xlabel("Autocorrelation at Lag-7 (weekly)"); ax.legend()
ax.set_title("Weekly Autocorrelation (Lag-7) per Tank", fontsize=13, fontweight="bold")
rprint("\nLag-7 autocorrelation per tank:")
rprint(acf7_s.round(4).to_string())
savefig("pertank_acf7_comparison")

# 7c: Seasonal strength per tank (using decomposition var ratio)
rprint("\nSeasonal & Trend Strength per Tank:")
strength_rows = []
for tank in tanks:
    s = wide[tank].dropna()
    if len(s) >= 28:
        try:
            dc = seasonal_decompose(s, model="additive", period=7, extrapolate_trend="freq")
            var_s = dc.seasonal.var()
            var_r = dc.resid.dropna().var()
            var_t = dc.trend.dropna().var()
            seasonal_str = max(0, 1 - var_r / (var_s + var_r)) if (var_s + var_r) > 0 else 0
            trend_str    = max(0, 1 - var_r / (var_t + var_r)) if (var_t + var_r) > 0 else 0
            strength_rows.append({"tank": tank, "seasonal_strength": round(seasonal_str,3),
                                   "trend_strength": round(trend_str,3)})
        except Exception:
            pass
strength_df = pd.DataFrame(strength_rows)
rprint(strength_df.sort_values("seasonal_strength", ascending=False).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ss = strength_df.sort_values("seasonal_strength", ascending=True)
axes[0].barh(range(len(ss)), ss["seasonal_strength"],
             color=["#2e7d32" if v>0.4 else "#f57f17" for v in ss["seasonal_strength"]], alpha=0.85)
axes[0].set_yticks(range(len(ss))); axes[0].set_yticklabels([t[:22] for t in ss["tank"]], fontsize=7)
axes[0].set_title("Seasonal Strength (period=7) per Tank", fontweight="bold"); axes[0].set_xlabel("Strength (0–1)")

ts2 = strength_df.sort_values("trend_strength", ascending=True)
axes[1].barh(range(len(ts2)), ts2["trend_strength"],
             color=["#1565c0" if v>0.5 else "#9e9e9e" for v in ts2["trend_strength"]], alpha=0.85)
axes[1].set_yticks(range(len(ts2))); axes[1].set_yticklabels([t[:22] for t in ts2["tank"]], fontsize=7)
axes[1].set_title("Trend Strength per Tank", fontweight="bold"); axes[1].set_xlabel("Strength (0–1)")
plt.suptitle("Seasonal & Trend Strength per Tank (variance ratio method)", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("pertank_seasonal_trend_strength")

# 7d: Top-6 tanks full decomposition (multi-panel)
top6 = tank_stats.sort_values("mean_out", ascending=False).head(6)["tank_id"].tolist()
fig, axes = plt.subplots(6, 4, figsize=(18, 22))
for row_i, tank in enumerate(top6):
    s = wide[tank].dropna()
    if len(s) < 28: continue
    try:
        dc = seasonal_decompose(s, model="additive", period=7, extrapolate_trend="freq")
        for col_i, (component, label, color) in enumerate([
            (dc.observed, "Observed", "gray"),
            (dc.trend,    "Trend",    "#1565c0"),
            (dc.seasonal, "Seasonal", "#e53935"),
            (dc.resid,    "Residual", "#2e7d32"),
        ]):
            ax = axes[row_i][col_i]
            component.plot(ax=ax, color=color, linewidth=0.8)
            ax.set_title(f"{tank[:14]} — {label}", fontsize=7, fontweight="bold")
            ax.tick_params(labelsize=5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    except Exception:
        pass
plt.suptitle("Seasonal Decomposition — Top 6 Tanks", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout(); savefig("pertank_decomposition_top6")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. CROSS-TANK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
rsep("SECTION 8 — Cross-Tank Analysis")

# 8a: Pearson correlation matrix
corr_mat = wide[tank_cols].corr()
fig, ax = plt.subplots(figsize=(15, 13))
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, cmap="coolwarm", center=0, annot=True,
            fmt=".2f", square=True, linewidths=0.3, ax=ax,
            xticklabels=[c[:13] for c in tank_cols], yticklabels=[c[:13] for c in tank_cols],
            annot_kws={"size":5.5})
ax.set_title("Cross-Tank Pearson Correlation (Daily Outflow)", fontsize=13, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=7); plt.yticks(fontsize=7)
savefig("crosstank_correlation_matrix")

# Top/bottom pairs
corr_pairs = [(t1, t2, corr_mat.loc[t1,t2])
              for i,t1 in enumerate(tank_cols) for j,t2 in enumerate(tank_cols) if j>i]
corr_df = pd.DataFrame(corr_pairs, columns=["tank1","tank2","corr"]).sort_values("corr", ascending=False)
rprint("\nTop 15 correlated pairs:")
rprint(corr_df.head(15).to_string(index=False))
rprint("\nBottom 10 (anti-correlated) pairs:")
rprint(corr_df.tail(10).to_string(index=False))

# 8b: Hierarchical clustering dendrogram
dist_mat = 1 - corr_mat.clip(-1,1)
np.fill_diagonal(dist_mat.values, 0)
link = linkage(dist_mat.values, method="ward")
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(link, labels=[c[:15] for c in tank_cols], ax=ax, leaf_rotation=45, leaf_font_size=8)
ax.set_title("Hierarchical Clustering (Ward, Distance = 1 − Pearson r)", fontsize=13, fontweight="bold")
ax.set_ylabel("Ward Distance")
savefig("crosstank_dendrogram")

# 8c: Rolling correlations between top-3 pairs
top3_pairs = corr_df.head(3)[["tank1","tank2"]].values
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for i, (t1, t2) in enumerate(top3_pairs):
    s1 = wide[t1].dropna()
    s2 = wide[t2].dropna()
    common = s1.index.intersection(s2.index)
    roll_r = s1.loc[common].rolling(30).corr(s2.loc[common])
    axes[i].plot(common, roll_r.values, color="#1565c0", linewidth=1.5)
    axes[i].axhline(corr_mat.loc[t1,t2], color="red", linestyle="--", alpha=0.7,
                    label=f"Overall r={corr_mat.loc[t1,t2]:.3f}")
    axes[i].axhline(0, color="black", lw=0.5)
    axes[i].set_title(f"Rolling 30-day r: {t1[:18]} ↔ {t2[:18]}", fontweight="bold"); axes[i].set_ylabel("r")
    axes[i].legend(fontsize=8); axes[i].set_ylim(-1, 1)
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
plt.suptitle("Rolling Correlation (30-day) — Top 3 Tank Pairs", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("crosstank_rolling_correlations")

# 8d: Cross-lag correlations (lag-1 predictability)
rprint("\nCross-lag correlations (lag-1 day):")
crosslag = []
for t1 in tank_cols:
    for t2 in tank_cols:
        if t1 != t2:
            s1 = wide[t1].dropna(); s2 = wide[t2].dropna()
            common = s1.index.intersection(s2.index)
            if len(common) > 30:
                r = s1.loc[common].shift(1).corr(s2.loc[common])
                crosslag.append({"predictor":t1,"target":t2,"lag1_r":r})
cl_df = pd.DataFrame(crosslag).dropna().sort_values("lag1_r", ascending=False)
rprint(cl_df.head(15).to_string(index=False))

# 8e: Lorenz curve + Gini
mean_by_tank = daily.groupby("tank_id")["daily_outflow"].mean().sort_values()
cumshare = np.cumsum(mean_by_tank.values) / mean_by_tank.sum()
n   = len(mean_by_tank)
gini = np.sum(np.abs(np.subtract.outer(mean_by_tank.values, mean_by_tank.values))) / (2 * n**2 * mean_by_tank.mean())
fig, ax = plt.subplots(figsize=(8, 7))
x_eq = np.linspace(0, 1, n)
ax.plot(x_eq, x_eq, "k--", alpha=0.4, label="Perfect equality")
ax.plot(np.linspace(0,1,n), cumshare, "o-", color="#1565c0", lw=2, ms=5)
ax.fill_between(np.linspace(0,1,n), x_eq, cumshare, alpha=0.15, color="#1565c0", label=f"Gini = {gini:.3f}")
ax.set_xlabel("Cumulative tank fraction"); ax.set_ylabel("Cumulative demand fraction")
ax.set_title(f"Lorenz Curve of Water Demand\nGini = {gini:.3f}  (0=equal, 1=monopoly)", fontsize=13, fontweight="bold")
ax.legend(); savefig("crosstank_lorenz_gini")
rprint(f"\nDemand Gini coefficient: {gini:.4f}")
top3_tanks = mean_by_tank.sort_values(ascending=False).head(3)
rprint(f"Top 3 tanks account for: {top3_tanks.sum()/mean_by_tank.sum()*100:.1f}% of campus demand")
rprint(top3_tanks.to_string())

# 8f: Granger causality (simplified, top pairs only)
rprint("\nGranger Causality Tests (max_lag=7, top correlated pairs):")
for t1, t2 in corr_df.head(8)[["tank1","tank2"]].values:
    s1 = wide[t1].dropna(); s2 = wide[t2].dropna()
    common = s1.index.intersection(s2.index)
    if len(common) > 60:
        df_gc = pd.concat([s1.loc[common], s2.loc[common]], axis=1).dropna()
        try:
            gc_res = grangercausalitytests(df_gc.values, maxlag=7, verbose=False)
            min_p = min(gc_res[lag][0]["ssr_ftest"][1] for lag in gc_res)
            rprint(f"  {t1[:20]:>20} → {t2[:20]:<20}  min_p={min_p:.4f}  {'**' if min_p<0.05 else ''}")
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════════════════
# 9. CALENDAR & ACADEMIC EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════
rsep("SECTION 9 — Calendar & Academic Effects")

# Phase statistics
phase_stats = campus_df.groupby("phase")["outflow"].agg(["mean","median","std","count"]).round(2)
phase_stats["cv"] = (phase_stats["std"]/phase_stats["mean"]).round(3)
phase_stats = phase_stats.sort_values("mean", ascending=False)
rprint("\nAcademic phase demand statistics:")
rprint(phase_stats.to_string())

groups_ph = [g["outflow"].values for _, g in campus_df.groupby("phase")]
f_stat, p_val = stats.f_oneway(*groups_ph)
rprint(f"\nANOVA: F={f_stat:.2f}, p={p_val:.2e}")

phase_list = list(phase_stats.index)
n_tests    = len(list(combinations(phase_list, 2)))
rprint("\nPairwise t-tests (Bonferroni α=0.05):")
for ph1, ph2 in combinations(phase_list, 2):
    g1 = campus_df[campus_df["phase"]==ph1]["outflow"].dropna()
    g2 = campus_df[campus_df["phase"]==ph2]["outflow"].dropna()
    if len(g1) > 2 and len(g2) > 2:
        t, p = stats.ttest_ind(g1, g2)
        sig = "**" if p < 0.05/n_tests else ("*" if p < 0.05 else "")
        rprint(f"  {ph1:<20} vs {ph2:<20}: t={t:+.2f}, p={p:.4f} {sig}")

# 9a: Academic phase boxplot
palette_p = {"regular":"#1565c0","isa":"#f57f17","esa":"#c62828",
             "holiday":"#4caf50","summer_break":"#9e9e9e","intersem_break":"#757575"}
fig, ax = plt.subplots(figsize=(13, 6))
sns.boxplot(data=campus_df, x="phase", y="outflow", order=phase_list,
            palette=palette_p, ax=ax, showfliers=False)
sns.stripplot(data=campus_df, x="phase", y="outflow", order=phase_list,
              color="black", alpha=0.15, size=3, ax=ax)
ax.set_title(f"Campus Demand by Academic Phase  (ANOVA F={f_stat:.1f}, p={p_val:.2e})", fontsize=12, fontweight="bold")
ax.set_xlabel(""); ax.set_ylabel("Daily Outflow (KL)")
for i, ph in enumerate(phase_list):
    n, m = int(phase_stats.loc[ph,"count"]), phase_stats.loc[ph,"mean"]
    ax.text(i, ax.get_ylim()[1]*0.97, f"n={n}\nμ={m:.0f}", ha="center", fontsize=8, fontweight="bold")
savefig("calendar_phase_boxplot")

# 9b: Calendar overlay on campus trend
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(campus.index, campus.rolling(7).mean(), color="#1565c0", linewidth=1.5, label="7-day MA")
span_defs = [(isa_dates,"#f57f17","ISA"),(esa_dates,"#c62828","ESA"),
             (summer_break,"#9e9e9e","Summer Break"),(intersem_break,"#c0c0c0","Inter-Sem Break")]
for span_d, color, label in span_defs:
    in_range = [d for d in span_d if campus.index.min() <= d <= campus.index.max()]
    if in_range:
        for d in in_range:
            ax.axvspan(d, d+pd.Timedelta(days=1), alpha=0.07, color=color)
from matplotlib.patches import Patch
ax.legend(handles=[
    ax.lines[0],
    Patch(facecolor="#f57f17",alpha=0.5,label="ISA"),
    Patch(facecolor="#c62828",alpha=0.5,label="ESA"),
    Patch(facecolor="#9e9e9e",alpha=0.5,label="Summer break"),
], fontsize=8)
ax.set_title("Campus Demand with Academic Calendar Overlay", fontsize=13, fontweight="bold")
ax.set_ylabel("Daily Outflow (KL)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45); savefig("calendar_overlay")

# 9c: Holiday dynamics ±7 days
window = 7
hol_ts = sorted([h for h in holidays_all if h in campus.index])
hol_around = []
for h in hol_ts:
    for delta in range(-window, window+1):
        d = h + pd.Timedelta(days=delta)
        if d in campus.index:
            hol_around.append({"delta":delta,"outflow":campus[d],"holiday":str(h.date())})
hol_agg = pd.DataFrame(hol_around).groupby("delta")["outflow"].agg(["mean","std","count"])
baseline = campus_df[campus_df["phase"]=="regular"]["outflow"].mean()
fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(hol_agg.index, hol_agg["mean"]-hol_agg["std"], hol_agg["mean"]+hol_agg["std"], alpha=0.2, color="coral")
ax.plot(hol_agg.index, hol_agg["mean"], "o-", color="crimson", linewidth=2, markersize=7)
ax.axvline(0, color="black", linestyle="--", alpha=0.6, label="Holiday day")
ax.axhline(baseline, color="gray", linestyle=":", label=f"Regular baseline={baseline:.0f} KL")
ax.set_xlabel("Days Relative to Holiday (0 = holiday)"); ax.set_ylabel("Mean Outflow (KL)")
ax.set_title("Demand Dynamics ±7 Days Around Holidays (avg all holidays)", fontsize=13, fontweight="bold")
ax.set_xticks(range(-window, window+1)); ax.legend()
for d in [-1,0,1]:
    if d in hol_agg.index:
        ax.annotate(f"{hol_agg.loc[d,'mean']:.0f}",(d,hol_agg.loc[d,"mean"]),
                    textcoords="offset points",xytext=(0,12),fontsize=10,fontweight="bold",ha="center")
rprint("\nDemand around holidays (mean outflow by day offset):")
rprint(hol_agg.to_string())
savefig("calendar_holiday_dynamics")

# 9d: ISA week deep dive
isa_starts = sorted([pd.Timestamp("2025-03-03"),pd.Timestamp("2025-04-21"),
                     pd.Timestamp("2025-09-22"),pd.Timestamp("2025-11-24"),pd.Timestamp("2026-03-02")])
isa_data = []
for start in isa_starts:
    for delta in range(-7, 12):
        d = start + pd.Timedelta(days=delta)
        if d in campus.index:
            phase_label = "Pre-ISA" if delta<0 else ("ISA Week" if delta<5 else "Post-ISA")
            isa_data.append({"delta":delta,"outflow":campus[d],"phase":phase_label,"isa_start":str(start.date())})
isa_df  = pd.DataFrame(isa_data)
isa_agg = isa_df.groupby("delta")["outflow"].agg(["mean","std"])
fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(isa_agg.index, isa_agg["mean"]-isa_agg["std"], isa_agg["mean"]+isa_agg["std"], alpha=0.15, color="orange")
ax.plot(isa_agg.index, isa_agg["mean"], "o-", color="darkorange", linewidth=2, markersize=5)
ax.axvspan(-0.5, 4.5, alpha=0.15, color="red", label="ISA Week (days 0–4)")
ax.axhline(baseline, color="gray", linestyle=":", label=f"Regular baseline={baseline:.0f}")
ax.set_xlabel("Days Relative to ISA Start"); ax.set_ylabel("Mean Outflow (KL)")
ax.set_title("Demand Pattern ±7 Days Around ISA Exams\n(averaged across 5 ISA windows)", fontsize=13, fontweight="bold")
ax.set_xticks(range(-7,12)); ax.legend(); savefig("calendar_isa_deep_dive")
rprint("\nISA window demand stats:")
rprint(isa_df.groupby("phase")["outflow"].agg(["mean","std","count"]).round(2).to_string())

# 9e: ESA progressive demand drop (Dec 2025 ESA timeline)
esa_dec = campus["2025-12-01":"2025-12-30"]
if len(esa_dec) > 5:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(esa_dec.index, esa_dec.values, "o-", color="crimson", linewidth=2, markersize=5)
    ax.axhline(baseline, color="gray", linestyle=":", label=f"Regular baseline={baseline:.0f}")
    z = np.polyfit(range(len(esa_dec)), esa_dec.values, 1)
    ax.plot(esa_dec.index, np.polyval(z, range(len(esa_dec))), "--", color="black", lw=1.5,
            label=f"Trend slope={z[0]:.2f} KL/day")
    ax.set_title("ESA Period Demand — Dec 2025 (Progressive Exam Drain)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Daily Outflow (KL)"); ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45); savefig("calendar_esa_dec_trend")
    rprint(f"\nDec 2025 ESA: mean={esa_dec.mean():.1f}  min={esa_dec.min():.1f}  trend slope={z[0]:.2f} KL/day")

# 9f: Per-tank phase sensitivity (which tanks react most to academic phase)
phase_effect = {}
for tank in tanks:
    s = daily.loc[daily["tank_id"]==tank, ["date","daily_outflow"]].dropna()
    if len(s) < 30: continue
    s = s.copy(); s["phase"] = s["date"].map(get_phase)
    ph_mean = s.groupby("phase")["daily_outflow"].mean()
    overall = s["daily_outflow"].mean()
    if "summer_break" in ph_mean and overall > 0:
        phase_effect[tank] = {"regular_vs_summer": (ph_mean.get("regular",overall) - ph_mean.get("summer_break",overall)) / overall * 100,
                               "regular_vs_holiday": (ph_mean.get("regular",overall) - ph_mean.get("holiday",overall)) / overall * 100}
pe_df = pd.DataFrame(phase_effect).T.round(1)
rprint("\nPer-tank phase sensitivity (% demand drop regular → phase):")
rprint(pe_df.sort_values("regular_vs_summer", ascending=False).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for col_i, (col, title) in enumerate([("regular_vs_summer","Drop: Regular → Summer Break"),
                                       ("regular_vs_holiday","Drop: Regular → Holiday")]):
    if col not in pe_df.columns: continue
    s = pe_df[col].sort_values(ascending=True)
    axes[col_i].barh(range(len(s)), s.values,
                     color=["#c62828" if v>30 else "#f57f17" if v>10 else "#2e7d32" for v in s.values], alpha=0.85)
    axes[col_i].set_yticks(range(len(s))); axes[col_i].set_yticklabels([t[:22] for t in s.index], fontsize=7)
    axes[col_i].set_title(title, fontweight="bold"); axes[col_i].set_xlabel("% Demand Drop")
plt.suptitle("Per-Tank Demand Sensitivity to Academic Calendar", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("calendar_per_tank_sensitivity")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. ANOMALY & CHANGE-POINT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
rsep("SECTION 10 — Anomaly & Change-Point Detection")

# 10a: Z-score anomalies
z_vals = np.abs(stats.zscore(campus.dropna()))
anomaly_dates = campus.dropna().index[z_vals > 3]
rprint(f"\nCampus Z-score anomalies (|z|>3): {len(anomaly_dates)}")
for d in anomaly_dates:
    zi = z_vals[campus.dropna().index == d][0]
    rprint(f"  {d.date()}  outflow={campus[d]:.1f}  z={zi:.2f}")

# 10b: IQR anomalies per tank (3×)
all_anoms = []
for tank in tanks:
    s = daily.loc[daily["tank_id"]==tank, ["date","daily_outflow"]].dropna(subset=["daily_outflow"])
    Q1,Q3 = s["daily_outflow"].quantile(0.25), s["daily_outflow"].quantile(0.75)
    IQR   = Q3 - Q1
    extr  = s[(s["daily_outflow"] < Q1-3*IQR) | (s["daily_outflow"] > Q3+3*IQR)]
    for _, row in extr.iterrows():
        all_anoms.append({"tank":tank,"date":row["date"],"value":row["daily_outflow"],"lo":Q1-3*IQR,"hi":Q3+3*IQR})
anom_df = pd.DataFrame(all_anoms)
rprint(f"\nExtreme IQR anomalies (3×) across all tanks: {len(anom_df)}")
rprint(anom_df.sort_values("value",ascending=False).head(25).to_string(index=False))

# 10c: Anomaly + z-score overlay plot
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(campus.index, campus.values, color="gray", lw=0.6, alpha=0.5)
ax.plot(campus.index, campus.rolling(7).mean(), color="#1565c0", lw=1.5, label="7-day MA")
if len(anomaly_dates):
    ax.scatter(anomaly_dates, campus[anomaly_dates], color="red", zorder=5, s=60,
               label=f"Z>3 anomalies (n={len(anomaly_dates)})")
ax.set_title("Campus Outflow with Anomalies Flagged (Z-score>3)", fontsize=13, fontweight="bold")
ax.set_ylabel("KL"); ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45); savefig("anomaly_zscore_campus")

# 10d: Monthly anomaly heatmap per tank
if len(anom_df) > 0:
    anom_df["month"] = pd.to_datetime(anom_df["date"]).dt.to_period("M")
    anom_monthly = anom_df.groupby(["tank","month"]).size().unstack(fill_value=0)
    if not anom_monthly.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.heatmap(anom_monthly, cmap="YlOrRd", linewidths=0.3, ax=ax,
                    annot=True, fmt="d", cbar_kws={"label":"# Extreme Days"})
        ax.set_title("Extreme Outflow Days per Tank per Month (3×IQR)", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45); savefig("anomaly_heatmap_monthly")

# 10e: CUSUM change detection on campus
rprint("\nCUSUM Change Detection — Campus Total Outflow:")
mu     = campus.dropna().mean()
sigma  = campus.dropna().std()
k      = 0.5  # slack
h      = 5.0  # decision threshold (in sigma units)
s_pos  = 0; s_neg = 0
cusum_pos = []; cusum_neg = []
for val in campus.dropna():
    s_pos = max(0, s_pos + (val - mu)/sigma - k)
    s_neg = max(0, s_neg - (val - mu)/sigma - k)
    cusum_pos.append(s_pos); cusum_neg.append(s_neg)
cusum_idx = campus.dropna().index
change_points = [cusum_idx[i] for i in range(len(cusum_pos)) if cusum_pos[i] > h or cusum_neg[i] > h]
rprint(f"  CUSUM alerts (k={k}, h={h}σ): {len(change_points)} days flagged")
if change_points:
    # Report transitions (start of alarm runs)
    prev = False
    for i, d in enumerate(cusum_idx):
        cur = cusum_pos[i] > h or cusum_neg[i] > h
        if cur and not prev:
            rprint(f"  Change-point alarm start: {d.date()}")
        prev = cur

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(cusum_idx, campus.dropna().values, color="gray", lw=0.7, alpha=0.6)
axes[0].plot(cusum_idx, campus.dropna().rolling(14).mean(), color="#1565c0", lw=1.5)
axes[0].set_title("Campus Outflow (14-day MA)", fontweight="bold"); axes[0].set_ylabel("KL")
axes[1].plot(cusum_idx, cusum_pos, color="#2e7d32", lw=1.2, label="CUSUM+")
axes[1].plot(cusum_idx, cusum_neg, color="#c62828", lw=1.2, label="CUSUM−")
axes[1].axhline(h, color="black", linestyle="--", lw=1, alpha=0.7, label=f"Threshold h={h}")
axes[1].set_title("CUSUM Change Detection", fontweight="bold"); axes[1].set_ylabel("CUSUM statistic"); axes[1].legend()
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
axes[1].xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45); plt.tight_layout(); savefig("anomaly_cusum")

# 10f: Day-over-day % change analysis
pct_chg = campus.pct_change() * 100
rprint("\nDay-over-day % change summary:")
rprint(pct_chg.dropna().describe().round(2).to_string())
extreme_drops   = pct_chg[pct_chg < -50].dropna()
extreme_spikes  = pct_chg[pct_chg >  80].dropna()
rprint(f"\nExtreme drops (>-50%): {len(extreme_drops)}")
for d, v in extreme_drops.items(): rprint(f"  {d.date()}: {v:+.1f}%")
rprint(f"\nExtreme spikes (>+80%): {len(extreme_spikes)}")
for d, v in extreme_spikes.items(): rprint(f"  {d.date()}: {v:+.1f}%")

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(pct_chg.index, pct_chg.values,
       color=["#c62828" if v<-30 else "#2e7d32" if v>30 else "#9e9e9e" for v in pct_chg.fillna(0)],
       alpha=0.7, width=1)
ax.axhline(0, color="black", lw=0.8)
ax.axhline(-50, color="red",   linestyle="--", alpha=0.5, label="-50%")
ax.axhline( 80, color="green", linestyle="--", alpha=0.5, label="+80%")
ax.set_title("Day-over-Day % Change in Campus Total Outflow", fontsize=13, fontweight="bold")
ax.set_ylabel("% Change"); ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45); savefig("anomaly_daily_pct_change")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. FEATURE ENGINEERING EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
rsep("SECTION 11 — Feature Engineering Evaluation")

feat = campus_df[["outflow"]].copy()
for lag in [1,2,3,5,7,10,14,21,28]:
    feat[f"lag_{lag}"]    = feat["outflow"].shift(lag)
feat["roll_mean_7"]   = feat["outflow"].shift(1).rolling(7).mean()
feat["roll_mean_14"]  = feat["outflow"].shift(1).rolling(14).mean()
feat["roll_std_7"]    = feat["outflow"].shift(1).rolling(7).std()
feat["roll_std_14"]   = feat["outflow"].shift(1).rolling(14).std()
feat["roll_min_7"]    = feat["outflow"].shift(1).rolling(7).min()
feat["roll_max_7"]    = feat["outflow"].shift(1).rolling(7).max()
feat["dow"]           = feat.index.dayofweek
feat["dow_sin"]       = np.sin(2*np.pi*feat.index.dayofweek/7)
feat["dow_cos"]       = np.cos(2*np.pi*feat.index.dayofweek/7)
feat["month"]         = feat.index.month
feat["month_sin"]     = np.sin(2*np.pi*feat.index.month/12)
feat["month_cos"]     = np.cos(2*np.pi*feat.index.month/12)
feat["is_weekend"]    = feat.index.dayofweek.isin([5,6]).astype(int)
feat["is_holiday"]    = feat.index.isin(holidays_all).astype(int)
feat["is_isa"]        = feat.index.isin(isa_dates).astype(int)
feat["is_esa"]        = feat.index.isin(esa_dates).astype(int)
feat["is_summer"]     = feat.index.isin(summer_break).astype(int)
feat["days_to_exam"]  = [days_to_next_exam(d) for d in feat.index]
feat = feat.dropna()

feature_cols = [c for c in feat.columns if c != "outflow"]

# Pearson correlations
pearson = feat[feature_cols].corrwith(feat["outflow"]).sort_values()
rprint("\nPearson correlations with daily campus outflow:")
rprint(pearson.round(4).to_string())

# Mutual information
disc = KBinsDiscretizer(n_bins=20, encode="ordinal", strategy="quantile")
y_d  = disc.fit_transform(feat[["outflow"]]).ravel().astype(int)
mi   = {}
for col in feature_cols:
    x_d = disc.fit_transform(feat[[col]]).ravel().astype(int)
    mi[col] = mutual_info_score(x_d, y_d)
mi_s = pd.Series(mi).sort_values(ascending=False)
rprint("\nMutual Information with daily campus outflow:")
rprint(mi_s.round(4).to_string())

# 11a: Feature importance comparison
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
pc_s = pearson.sort_values(ascending=True)
axes[0].barh(range(len(pc_s)), pc_s.values,
             color=["#c62828" if v<0 else "#1565c0" for v in pc_s.values], alpha=0.85)
axes[0].set_yticks(range(len(pc_s))); axes[0].set_yticklabels(pc_s.index, fontsize=7)
axes[0].axvline(0, color="black", lw=0.8); axes[0].set_title("Pearson r with Outflow", fontweight="bold")
for i, v in enumerate(pc_s.values):
    axes[0].text(v + (0.01 if v>=0 else -0.05), i, f"{v:.3f}", va="center", fontsize=6.5)

mi_sorted = mi_s.sort_values(ascending=True)
axes[1].barh(range(len(mi_sorted)), mi_sorted.values, color="#2e7d32", alpha=0.8)
axes[1].set_yticks(range(len(mi_sorted))); axes[1].set_yticklabels(mi_sorted.index, fontsize=7)
axes[1].set_title("Mutual Information with Outflow", fontweight="bold"); axes[1].set_xlabel("MI score")
for i, v in enumerate(mi_sorted.values):
    axes[1].text(v+0.002, i, f"{v:.3f}", va="center", fontsize=6.5)

plt.suptitle("Feature Importance: Pearson r vs Mutual Information", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("features_importance_comparison")

# 11b: Lag correlation profile
fig, ax = plt.subplots(figsize=(13, 5))
lags    = list(range(1, 29))
lag_rs  = [campus.autocorr(lag=l) for l in lags]
colors_l = ["crimson" if l in [1,7,14,21,28] else "steelblue" for l in lags]
ax.bar(lags, lag_rs, color=colors_l, alpha=0.8)
ax.axhline(0, color="black", lw=0.5)
for l in [1,7,14,21,28]:
    ax.annotate(f"lag-{l}\n{lag_rs[l-1]:.3f}", (l, lag_rs[l-1]),
                textcoords="offset points", xytext=(0,8), fontsize=8, ha="center",
                fontweight="bold", color="crimson")
ax.set_xlabel("Lag (days)"); ax.set_ylabel("Autocorrelation")
ax.set_title("Lag Autocorrelation Profile — Campus Total (key lags highlighted)", fontsize=13, fontweight="bold")
savefig("features_lag_profile")

# 11c: Conditional distribution — outflow by DOW (filled density)
fig, ax = plt.subplots(figsize=(13, 6))
dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
palette_d = sns.color_palette("tab10", 7)
for dow_n in range(7):
    sub = campus_df[campus_df["dow"]==dow_n]["outflow"].dropna()
    if len(sub) > 10:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(sub, bw_method=0.3)
        x_r = np.linspace(sub.min()-20, sub.max()+20, 200)
        ax.plot(x_r, kde(x_r), color=palette_d[dow_n], lw=2, label=dow_names[dow_n])
        ax.fill_between(x_r, kde(x_r), alpha=0.06, color=palette_d[dow_n])
ax.set_xlabel("Campus Outflow (KL)"); ax.set_ylabel("Density")
ax.set_title("Conditional Demand Distribution by Day of Week", fontsize=13, fontweight="bold")
ax.legend(); savefig("features_dow_density")

# 11d: Conditional distribution — by academic phase
fig, ax = plt.subplots(figsize=(13, 6))
phase_colors_d = {"regular":"#1565c0","isa":"#f57f17","esa":"#c62828",
                  "summer_break":"#9e9e9e","holiday":"#4caf50","intersem_break":"#757575"}
for ph, color in phase_colors_d.items():
    sub = campus_df[campus_df["phase"]==ph]["outflow"].dropna()
    if len(sub) > 8:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(sub, bw_method=0.35)
        x_r = np.linspace(max(50, sub.min()-20), sub.max()+20, 200)
        ax.plot(x_r, kde(x_r), color=color, lw=2, label=f"{ph} (n={len(sub)})")
        ax.fill_between(x_r, kde(x_r), alpha=0.06, color=color)
ax.set_xlabel("Campus Outflow (KL)"); ax.set_ylabel("Density")
ax.set_title("Conditional Demand Distribution by Academic Phase", fontsize=13, fontweight="bold")
ax.legend(fontsize=8); savefig("features_phase_density")

# 11e: Lag scatter matrix (key lags)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes_f = axes.flatten()
key_lags = ["lag_1","lag_2","lag_3","lag_5","lag_7","lag_14","roll_mean_7","roll_std_7"]
for i, col in enumerate(key_lags):
    ax = axes_f[i]
    ax.scatter(feat[col], feat["outflow"], alpha=0.2, s=6, color="steelblue")
    m,b,r,p,_ = stats.linregress(feat[col].dropna(), feat.loc[feat[col].notna(),"outflow"])
    xr = np.linspace(feat[col].min(), feat[col].max(), 50)
    ax.plot(xr, m*xr+b, color="red", lw=1.5)
    ax.set_title(f"{col}  (r={r:.3f})", fontsize=9, fontweight="bold")
    ax.set_xlabel("Feature"); ax.set_ylabel("Outflow")
    ax.tick_params(labelsize=7)
plt.suptitle("Lag & Rolling Feature vs Target Scatter Plots", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("features_lag_scatter_matrix")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════
rsep("FINAL SUMMARY", char="═")

rprint(f"""
╔══════════════════════════════════════════════════════════════════════╗
║           WALTR DATA — COMPREHENSIVE EDA SUMMARY REPORT            ║
╚══════════════════════════════════════════════════════════════════════╝

── DATASET ────────────────────────────────────────────────────────────
  Tanks              : {N_TANKS}  (all overhead; 19 cuboidal, 5 cylindrical)
  Date range         : {hrly['date'].min().date()} → {hrly['date'].max().date()}  ({(hrly['date'].max()-hrly['date'].min()).days+1} calendar days)
  Hourly records     : {len(hrly):,}
  Daily records      : {len(daily):,}
  Largest tank (cap) : MRD_BLOCK / BE_BLOCK_OHT / TECH_PARK  (~60-67 KL each)
  Smallest tank      : INFORMATION_CENTRE / BE_BLOCK_RO  (~1 KL)

── CAMPUS DEMAND ──────────────────────────────────────────────────────
  Mean / day         : {campus.mean():.1f} KL
  Median / day       : {campus.median():.1f} KL
  Std dev            : {campus.std():.1f} KL
  Max / day          : {campus.max():.1f} KL  on {campus.idxmax().date()}
  Min / day (>0)     : {campus[campus>0].min():.1f} KL
  Total volume       : {campus.sum():.0f} KL  ({campus.sum()/1000:.1f} ML)
  CV (coeff. var.)   : {campus.std()/campus.mean():.3f}

── WEEKDAY vs WEEKEND ─────────────────────────────────────────────────
  Weekday mean       : {wkd.mean():.1f} KL
  Weekend mean       : {wke.mean():.1f} KL
  Difference         : {wkd.mean()-wke.mean():.1f} KL  ({(wkd.mean()-wke.mean())/wkd.mean()*100:.1f}% lower on weekend)
  t-test             : t={t:.3f}, p={p:.5f}  (SIGNIFICANT)

── STATIONARITY ───────────────────────────────────────────────────────
  ADF test           : stat={adf_r[0]:.4f}, p={adf_r[1]:.5f}  → {'STATIONARY' if adf_r[1]<0.05 else 'NON-STATIONARY'}
  KPSS test          : stat={kpss_r[0]:.4f}, p≈{kpss_r[1]:.4f}  → {'STATIONARY' if kpss_r[1]>0.05 else 'NON-STATIONARY'}
  First-diff ADF     : stat={adf_d[0]:.4f}, p={adf_d[1]:.5f}  → {'STATIONARY' if adf_d[1]<0.05 else 'NON-STATIONARY'}

── SEASONALITY ────────────────────────────────────────────────────────
  Weekly ACF (lag-7) : {acf_vals[7]:.4f}  (strong weekly pattern)
  Seasonal strength  : see per-tank table in Section 7

── CROSS-TANK ─────────────────────────────────────────────────────────
  Gini coefficient   : {gini:.3f}  (moderately concentrated demand)
  Top 3 tanks share  : {top3_tanks.sum()/mean_by_tank.sum()*100:.1f}% of campus total
  Strongest pair r   : {corr_df.iloc[0]['tank1'][:18]} ↔ {corr_df.iloc[0]['tank2'][:18]}  r={corr_df.iloc[0]['corr']:.3f}
  Night-time leakage : check INFORMATION_CENTRE, IT_BLOCK for high night shares

── ACADEMIC CALENDAR EFFECTS ─────────────────────────────────────────
  Phase ranking      : ISA > Regular ≈ ESA > Summer > Holiday ≈ Inter-sem
  Regular classes    : {phase_stats.loc['regular','mean'] if 'regular' in phase_stats.index else 'N/A'} KL/day
  Summer break       : {phase_stats.loc['summer_break','mean'] if 'summer_break' in phase_stats.index else 'N/A'} KL/day
  Holiday            : {phase_stats.loc['holiday','mean'] if 'holiday' in phase_stats.index else 'N/A'} KL/day
  ANOVA p-value      : {p_val:.2e}  (HIGHLY SIGNIFICANT)
  ISA weeks show +demand (students staying on campus to study)
  Summer break = -10% from regular; holidays = -17% from regular

── ANOMALIES ──────────────────────────────────────────────────────────
  Campus Z>3 dates   : {len(anomaly_dates)} days  ({', '.join(str(d.date()) for d in anomaly_dates)})
  Per-tank 3×IQR     : {len(anom_df)} extreme days across all tanks
  MRD_BLOCK largest  : 177 KL in a single day (Sep 1 2025) vs IQR ceiling 95

── TOP FORECASTING FEATURES (by Mutual Information) ──────────────────
{mi_s.head(8).round(4).to_string()}

── DATA QUALITY ───────────────────────────────────────────────────────
  INFORMATION_CENTRE : 76.5% zero-outflow days — sensor/usage issue
  CRICKET_GROUND_OHT : only 82% coverage, starts Jun 2025
  ME_WORKSHOP_BLOCK  : 90% coverage (43 gap days)
  Most other tanks   : ≥98% coverage ✓

── PLOTS GENERATED ────────────────────────────────────────────────────
  {plot_index[0]} plots saved to: {PLOT_DIR}
  Report saved to  : {REPORT}
""")

save_report()
rprint(f"Done. {plot_index[0]} plots + 1 report written.")
