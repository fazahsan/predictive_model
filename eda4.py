#Eda 4
"""
EDA Step 4 — Time Series Analysis
===================================
Goal: Understand how sensor signals evolve over TIME, and what happens
      in the critical window leading up to a fault (the degradation zone).

We produce 4 panels:
  A) Full timeline of key sensors for one turbine (1 year)
     → shows fault events as red vertical bands
  B) Zoom-in: average sensor behaviour in the 20h before fault
     → aligned across all fault events, shows pre-fault signature
  C) RUL countdown: sensor values vs RUL (0 = fault)
     → the degradation curve — critical for RUL regression models
  D) Fault frequency over time (all turbines)
     → reveals if faults cluster or are evenly distributed

Why this matters for Predictive Maintenance:
  - Panel B tells you HOW EARLY you can detect a fault
  - Panel C tells you WHAT THE SIGNAL LOOKS LIKE as failure approaches
  - Panel D tells you if there are seasonal or temporal patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("dataset_realistic.csv", parse_dates=["timestamp"])
df = df.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)

# ── Focus turbine for panels A & B ───────────────────────────────────────────
TURBINE = "T001"
tdf = df[df["turbine_id"] == TURBINE].copy().reset_index(drop=True)

# Mark fault event starts (0→1 transitions)
tdf["prev_fallo"] = tdf["Fallo"].shift(1, fill_value=0)
fault_starts = tdf[(tdf["Fallo"] == 1) & (tdf["prev_fallo"] == 0)].copy()

# Key sensors to highlight
KEY_SENSORS   = ["Eficiencia", "Vibracion", "Temp_cojinete", "Presion_salida"]
SENSOR_LABELS = {
    "Eficiencia":      "Efficiency",
    "Vibracion":       "Vibration",
    "Temp_cojinete":   "Bearing Temp",
    "Presion_salida":  "Outlet Pressure",
}
COLORS = ["#1a6fdb", "#e63946", "#2d9e5f", "#e08b00"]

# ── Panel C data: sensor vs RUL across ALL turbines ──────────────────────────
rul_df = df[df["RUL"].notna()].copy()
rul_df["RUL_int"] = rul_df["RUL"].round().astype(int)
rul_agg = rul_df.groupby("RUL_int")[KEY_SENSORS].mean()

# ── Panel D data: fault frequency by month ───────────────────────────────────
df["month"] = df["timestamp"].dt.to_period("M")
df["prev_fallo"] = df.groupby("turbine_id")["Fallo"].shift(1, fill_value=0)
fault_events = df[(df["Fallo"] == 1) & (df["prev_fallo"] == 0)].copy()
fault_by_month = fault_events.groupby(["month", "Tipo_fallo"]).size().unstack(fill_value=0)
fault_types_all = ["Fuga", "Termico", "Cojinete", "Condensador"]
for ft in fault_types_all:
    if ft not in fault_by_month.columns:
        fault_by_month[ft] = 0
fault_by_month = fault_by_month[fault_types_all]

# ── Print insights ────────────────────────────────────────────────────────────
print("=" * 65)
print("TIME SERIES INSIGHTS")
print("=" * 65)
print(f"\n① Turbine {TURBINE} overview:")
print(f"   Total fault events : {len(fault_starts)}")
print(f"   Date range         : {tdf['timestamp'].min().date()} → {tdf['timestamp'].max().date()}")
print(f"   Fault rate         : {len(fault_starts) / (len(tdf)/24/7):.1f} faults/week")

print(f"\n② Sensor trend as RUL → 0 (fault approach, all turbines):")
for s in KEY_SENSORS:
    val_20 = rul_agg.loc[20, s] if 20 in rul_agg.index else np.nan
    val_0  = rul_agg.loc[0,  s] if 0  in rul_agg.index else np.nan
    pct_change = (val_0 - val_20) / val_20 * 100
    direction = "↑" if pct_change > 0 else "↓"
    print(f"   {SENSOR_LABELS[s]:<20}: RUL=20 → {val_20:.2f}  |  "
          f"RUL=0 → {val_0:.2f}  ({direction}{abs(pct_change):.1f}%)")

print(f"\n③ Fault events by month (total):")
monthly_total = fault_by_month.sum(axis=1)
print(f"   Min per month  : {monthly_total.min()}")
print(f"   Max per month  : {monthly_total.max()}")
print(f"   Mean per month : {monthly_total.mean():.1f}")

# ── Figure ────────────────────────────────────────────────────────────────────
BG      = "white"
DARK    = "#1a1a2e"
BORDER  = "#d0d7e3"
PANEL   = "#f7f9fc"
FAULT_C = "#ff4757"

FT_COLORS = {
    "Fuga":       "#1a6fdb",
    "Termico":    "#e63946",
    "Cojinete":   "#2d9e5f",
    "Condensador":"#e08b00",
}

fig = plt.figure(figsize=(22, 20), facecolor=BG)
fig.suptitle(f"EDA Step 4 — Time Series Analysis  (Turbine {TURBINE} + All Fleet)",
             fontsize=19, color=DARK, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(4, 2, figure=fig,
                       height_ratios=[2.2, 1.6, 1.8, 1.6],
                       hspace=0.55, wspace=0.32)

def style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors="#444", labelsize=8)
    ax.set_title(title, color=DARK, fontsize=12, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color="#555", fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color="#555", fontsize=9)
    ax.grid(axis="y", color=BORDER, linewidth=0.6, linestyle="--", alpha=0.7)

# ── Panel A: Full year timeline ───────────────────────────────────────────────
ax_A = fig.add_subplot(gs[0, :])   # spans both columns

# Normalise sensors to 0-1 for joint plotting
for idx, (s, col) in enumerate(zip(KEY_SENSORS, COLORS)):
    vals = tdf[s].ffill()
    norm = (vals - vals.min()) / (vals.max() - vals.min())
    ax_A.plot(tdf["timestamp"], norm, color=col, linewidth=0.6,
              alpha=0.75, label=SENSOR_LABELS[s])

# Shade fault windows
for _, row in fault_starts.iterrows():
    ax_A.axvline(row["timestamp"], color=FAULT_C, linewidth=0.5, alpha=0.35)

# Highlight first 3 faults with labels
for i, (_, row) in enumerate(fault_starts.head(3).iterrows()):
    ax_A.axvline(row["timestamp"], color=FAULT_C, linewidth=1.5, alpha=0.9)
    ax_A.text(row["timestamp"], 1.03, row["Tipo_fallo"],
              rotation=45, fontsize=6.5, color=FAULT_C, ha="left", va="bottom")

style(ax_A,
      f"A  Full-Year Sensor Timeline — Turbine {TURBINE}  "
      f"(sensors normalised 0–1, red lines = fault events)",
      ylabel="Normalised value")
ax_A.set_xlim(tdf["timestamp"].min(), tdf["timestamp"].max())
ax_A.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
ax_A.legend(loc="lower left", fontsize=8, framealpha=0.9,
            edgecolor=BORDER, facecolor="white", ncol=4)
ax_A.tick_params(axis="x", rotation=20)

# ── Panel B: Zoom into pre-fault signature (average across events) ────────────
ax_B = fig.add_subplot(gs[1, 0])

# Collect windows: 40h before each fault, aligned by RUL
windows = {s: [] for s in KEY_SENSORS}
for _, frow in fault_starts.iterrows():
    fault_ts = frow["timestamp"]
    # window = rows where RUL is not null and belongs to this event
    # Find the pre-fault block ending at this fault
    event_mask = (
        (tdf["timestamp"] <= fault_ts) &
        (tdf["timestamp"] >= fault_ts - pd.Timedelta(hours=25))
    )
    window = tdf[event_mask].copy()
    if len(window) < 5:
        continue
    # Assign hours-to-fault
    window["hours_to_fault"] = (fault_ts - window["timestamp"]).dt.total_seconds() / 3600
    for s in KEY_SENSORS:
        ww = window[["hours_to_fault", s]].dropna()
        windows[s].append(ww)

hours_grid = np.arange(0, 22, 1)
for idx, (s, col) in enumerate(zip(KEY_SENSORS, COLORS)):
    all_interp = []
    for wdf in windows[s]:
        if len(wdf) < 3:
            continue
        # interpolate to hour grid
        wdf = wdf.sort_values("hours_to_fault")
        try:
            interp = np.interp(hours_grid,
                               wdf["hours_to_fault"].values[::-1],
                               wdf[s].values[::-1])
            all_interp.append(interp)
        except Exception:
            pass
    if not all_interp:
        continue
    arr = np.array(all_interp)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)

    # Normalise for joint plot
    n_mean = (mean - mean.max()) / (mean.max() - mean.min() + 1e-9) + 1
    n_std  = std / (mean.max() - mean.min() + 1e-9)

    ax_B.plot(hours_grid, n_mean, color=col, linewidth=2, label=SENSOR_LABELS[s])
    ax_B.fill_between(hours_grid, n_mean - n_std, n_mean + n_std,
                      color=col, alpha=0.12)

ax_B.axvline(0, color=FAULT_C, linewidth=2, linestyle="--", label="FAULT")
ax_B.invert_xaxis()
ax_B.set_xlabel("Hours to Fault →", color="#555", fontsize=9)
style(ax_B, "B  Pre-Fault Sensor Signature\n(avg across all events, normalised)",
      ylabel="Normalised value")
ax_B.legend(fontsize=7, edgecolor=BORDER, facecolor="white")
ax_B.text(0.5, -0.18, "← Time approaching fault",
          transform=ax_B.transAxes, ha="center", color="#888", fontsize=8)

# ── Panel C: Sensor degradation vs RUL ───────────────────────────────────────
ax_C = fig.add_subplot(gs[1, 1])

for idx, (s, col) in enumerate(zip(KEY_SENSORS, COLORS)):
    y = rul_agg[s]
    # normalise
    yn = (y - y.min()) / (y.max() - y.min())
    ax_C.plot(rul_agg.index, yn, color=col, linewidth=2,
              label=SENSOR_LABELS[s], marker="o", markersize=2)

ax_C.axvline(0, color=FAULT_C, linewidth=2, linestyle="--")
ax_C.invert_xaxis()
style(ax_C, "C  Sensor Mean vs RUL (all turbines)\nRUL=0 → fault moment",
      xlabel="Remaining Useful Life (hours) →", ylabel="Normalised mean")
ax_C.legend(fontsize=7, edgecolor=BORDER, facecolor="white")
ax_C.text(1, 0.5, "FAULT", transform=ax_C.transAxes,
          color=FAULT_C, fontsize=9, fontweight="bold",
          va="center", ha="right", rotation=90)

# ── Panel D: Fault frequency by month (stacked bar) ──────────────────────────
ax_D = fig.add_subplot(gs[2, :])

x = np.arange(len(fault_by_month))
bottom = np.zeros(len(fault_by_month))
for ft in fault_types_all:
    vals = fault_by_month[ft].values.astype(float)
    bars = ax_D.bar(x, vals, bottom=bottom, label=ft,
                    color=FT_COLORS[ft], edgecolor="white", linewidth=0.4, width=0.75)
    bottom += vals

month_labels = [str(m) for m in fault_by_month.index]
ax_D.set_xticks(x)
ax_D.set_xticklabels(month_labels, rotation=35, ha="right", fontsize=8)
style(ax_D, "D  Fault Event Frequency by Month — All 50 Turbines  (stacked by fault type)",
      ylabel="Number of fault events")
ax_D.legend(title="Fault Type", fontsize=8, edgecolor=BORDER, facecolor="white",
            title_fontsize=8, loc="upper left")

# Rolling 3-month average line
rolling_total = pd.Series(fault_by_month.sum(axis=1).values).rolling(3, center=True).mean()
ax_D.plot(x, rolling_total, color=DARK, linewidth=2, linestyle="--",
          label="3-month rolling avg", zorder=5)

# ── Panel E: Fault inter-arrival time distribution ───────────────────────────
ax_E = fig.add_subplot(gs[3, 0])

# Compute time between consecutive fault events per turbine
inter_arrivals = []
for tid in df["turbine_id"].unique():
    t_faults = fault_events[fault_events["turbine_id"] == tid]["timestamp"].sort_values()
    if len(t_faults) > 1:
        diffs = t_faults.diff().dropna().dt.total_seconds() / 3600
        inter_arrivals.extend(diffs.tolist())

inter_arr = np.array(inter_arrivals)
ax_E.hist(inter_arr, bins=60, color="#1a6fdb", edgecolor="white",
          linewidth=0.3, alpha=0.85)
ax_E.axvline(np.median(inter_arr), color=FAULT_C, linewidth=2,
             linestyle="--", label=f"Median = {np.median(inter_arr):.0f}h")
ax_E.axvline(np.mean(inter_arr),   color="#e08b00", linewidth=2,
             linestyle="--", label=f"Mean   = {np.mean(inter_arr):.0f}h")
style(ax_E, "E  Inter-Fault Arrival Time Distribution\n(hours between consecutive faults, per turbine)",
      xlabel="Hours between faults", ylabel="Count")
ax_E.legend(fontsize=8, edgecolor=BORDER, facecolor="white")

print(f"\n④ Inter-fault arrival time:")
print(f"   Median: {np.median(inter_arr):.0f}h  |  Mean: {np.mean(inter_arr):.0f}h  "
      f"|  Min: {inter_arr.min():.0f}h  |  Max: {inter_arr.max():.0f}h")

# ── Panel F: Fault type proportions per turbine (heatmap) ────────────────────
ax_F = fig.add_subplot(gs[3, 1])

turb_ft = fault_events.groupby(["turbine_id", "Tipo_fallo"]).size().unstack(fill_value=0)
turb_ft_pct = turb_ft.div(turb_ft.sum(axis=1), axis=0)

im = ax_F.imshow(turb_ft_pct.T.values, aspect="auto",
                 cmap="YlOrRd", vmin=0, vmax=0.5,
                 interpolation="nearest")
ax_F.set_yticks(range(len(fault_types_all)))
ax_F.set_yticklabels(fault_types_all, fontsize=8, color=DARK)
ax_F.set_xticks(range(0, len(turb_ft_pct), 5))
ax_F.set_xticklabels(turb_ft_pct.index[::5], fontsize=7, rotation=35, ha="right")
style(ax_F, "F  Fault Type Mix per Turbine\n(% of each turbine's faults)")
cb = plt.colorbar(im, ax=ax_F, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=7)
cb.set_label("Proportion", fontsize=8)

plt.savefig("eda_step4_timeseries.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.show()
print("\n✅  Saved: eda_step4_timeseries.png")