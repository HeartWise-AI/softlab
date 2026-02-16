#!/usr/bin/env python3
"""
Distribution plots for harmonized Softlab lab results.
Produces:
  1. Daily sample count heatmap (day-of-year coverage)
  2. Distribution (hist/KDE) per key lab test
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

OUT = Path("/volume/softlab")
PARQUET = OUT / "softlab_master.parquet"

print("Loading parquet ...", flush=True)
df = pd.read_parquet(PARQUET)
print(f"  {len(df):,} rows loaded", flush=True)

# ── 1. Day-of-year coverage heatmap ───────────────────────────────────────
print("Generating day-of-year coverage plot ...", flush=True)

df["date"] = df["test_dt"].dt.date
daily = df.groupby("date").size().reset_index(name="count")
daily["date"] = pd.to_datetime(daily["date"])
daily["year"] = daily["date"].dt.year
daily["dayofyear"] = daily["date"].dt.dayofyear

# Pivot: rows=dayofyear, cols=year
pivot = daily.pivot_table(index="dayofyear", columns="year", values="count", fill_value=0)

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(
    pivot.T, cmap="YlOrRd", ax=ax, cbar_kws={"label": "# lab results per day"},
    xticklabels=30, yticklabels=1
)
ax.set_xlabel("Day of Year")
ax.set_ylabel("Year")
ax.set_title("Lab Results: Daily Volume Coverage (2015-2025)")
plt.tight_layout()
fig.savefig(OUT / "coverage_heatmap.png", dpi=150)
plt.close(fig)
print("  Saved coverage_heatmap.png", flush=True)

# ── 2. Distribution plots for key labs ────────────────────────────────────
# Select the most clinically relevant tests with good numeric coverage
KEY_LABS = {
    "Creatinine (serum)":  {"unit": "umol/L",    "xlim": (0, 500),     "category": "Creatinine"},
    "Creatinine":          {"unit": "umol/L",     "xlim": (0, 500),     "category": "Creatinine"},
    "eGFR (MDRD)":         {"unit": "mL/min/1.73m2", "xlim": (0, 150), "category": "Creatinine"},
    "Hemoglobin":          {"unit": "g/L",        "xlim": (30, 220),    "category": "CBC"},
    "WBC (absolute)":      {"unit": "10*9/L",     "xlim": (0, 40),      "category": "CBC"},
    "Platelets":           {"unit": "10*9/L",     "xlim": (0, 600),     "category": "CBC"},
    "RBC":                 {"unit": "10*12/L",    "xlim": (1, 7),       "category": "CBC"},
    "Hematocrit":          {"unit": "L/L",        "xlim": (0.1, 0.65),  "category": "CBC"},
    "MCV":                 {"unit": "fL",         "xlim": (50, 130),    "category": "CBC"},
    "Neutrophils (abs)":   {"unit": "10*9/L",     "xlim": (0, 30),      "category": "CBC"},
    "Lymphocytes (abs)":   {"unit": "10*9/L",     "xlim": (0, 10),      "category": "CBC"},
    "Troponin T":          {"unit": "ng/L",       "xlim": (0, 500),     "category": "Troponin"},
    "Troponin T (hs)":     {"unit": "ng/L",       "xlim": (0, 500),     "category": "Troponin"},
    "Sodium":              {"unit": "mmol/L",     "xlim": (110, 160),   "category": "Electrolytes"},
    "Potassium":           {"unit": "mmol/L",     "xlim": (2, 7),       "category": "Electrolytes"},
    "Chloride":            {"unit": "mmol/L",     "xlim": (80, 130),    "category": "Electrolytes"},
    "NT-proBNP":           {"unit": "pg/mL",      "xlim": (0, 5000),    "category": "BNP"},
}

# Filter to rows with numeric results
num_df = df[df["result_numeric"].notna()].copy()

print("Generating distribution plots ...", flush=True)

# Group into figure panels
labs_with_data = []
for lab_name, info in KEY_LABS.items():
    subset = num_df[num_df["lab_name"] == lab_name]
    if len(subset) >= 100:
        labs_with_data.append((lab_name, info, subset))

n = len(labs_with_data)
ncols = 3
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
axes_flat = axes.flatten() if n > 1 else [axes]

for idx, (lab_name, info, subset) in enumerate(labs_with_data):
    ax = axes_flat[idx]
    vals = subset["result_numeric"].dropna()
    lo, hi = info["xlim"]
    vals_clip = vals[(vals >= lo) & (vals <= hi)]

    ax.hist(vals_clip, bins=80, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3, density=True)
    try:
        vals_clip.plot.kde(ax=ax, color="darkred", linewidth=1.5)
    except Exception:
        pass
    ax.set_title(f"{lab_name}\n(n={len(vals):,})", fontsize=10)
    ax.set_xlabel(info["unit"], fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_xlim(lo, hi)
    ax.tick_params(labelsize=8)

# Hide unused axes
for idx in range(len(labs_with_data), len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle("Lab Value Distributions (2015-2025)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(OUT / "lab_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved lab_distributions.png", flush=True)

# ── 3. Time-series daily counts per category ──────────────────────────────
print("Generating daily volume time-series ...", flush=True)

cat_daily = df.groupby([df["test_dt"].dt.date, "lab_category"]).size().reset_index(name="count")
cat_daily.columns = ["date", "lab_category", "count"]
cat_daily["date"] = pd.to_datetime(cat_daily["date"])

# Plot top categories
top_cats = ["CBC", "Creatinine", "Electrolytes", "Troponin", "BNP"]
fig, axes = plt.subplots(len(top_cats), 1, figsize=(16, 3 * len(top_cats)), sharex=True)

for i, cat in enumerate(top_cats):
    ax = axes[i]
    sub = cat_daily[cat_daily["lab_category"] == cat].sort_values("date")
    ax.plot(sub["date"], sub["count"], linewidth=0.5, color="steelblue", alpha=0.6)
    # 30-day rolling average
    if len(sub) > 30:
        rolling = sub.set_index("date")["count"].rolling("30D").mean()
        ax.plot(rolling.index, rolling.values, linewidth=2, color="darkred", label="30-day avg")
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(f"{cat} — Daily Lab Volume", fontsize=11)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.suptitle("Daily Lab Volume by Category (2015-2025)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(OUT / "daily_volume_timeseries.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved daily_volume_timeseries.png", flush=True)

# ── Summary stats ─────────────────────────────────────────────────────────
print("\n=== Summary Statistics for Key Labs ===", flush=True)
for lab_name, info, subset in labs_with_data:
    vals = subset["result_numeric"].dropna()
    print(f"\n{lab_name} ({info['unit']}):", flush=True)
    print(f"  n={len(vals):,}  mean={vals.mean():.2f}  median={vals.median():.2f}  "
          f"std={vals.std():.2f}  min={vals.min():.2f}  max={vals.max():.2f}", flush=True)

print("\nAll plots saved to /volume/softlab/", flush=True)
print("Done!", flush=True)
