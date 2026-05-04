"""
EDA Step 2 — Sensor Distributions: Normal vs Fault
============================================================
Goal: Identify discriminative sensors for ML modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde, ks_2samp

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("dataset_realistic.csv")

sensors = [
    "Carga_pct", "Flujo_masico", "Potencia", "Presion_entrada",
    "Temp_entrada", "Presion_salida", "Eficiencia", "Vibracion", "Temp_cojinete"
]

sensor_labels = {
    "Carga_pct": "Carga (%)",
    "Flujo_masico": "Flujo Masico",
    "Potencia": "Potencia",
    "Presion_entrada": "Presion Entrada",
    "Temp_entrada": "Temperatura Entrada",
    "Presion_salida": "Presion Salida",
    "Eficiencia": "Eficiencia",
    "Vibracion": "Vibracion",
    "Temp_cojinete": "Temperatura Cojinete",
}

normal = df[df["Fallo"] == 0]
fault  = df[df["Fallo"] == 1]

# ── Helper: safe KDE ──────────────────────────────────────────────────────────
def kde_curve(data, x_grid):
    if len(data) < 50:  # avoid instability
        return np.zeros_like(x_grid)
    kde = gaussian_kde(data, bw_method=0.3)
    return kde(x_grid)

# ── Metrics storage ───────────────────────────────────────────────────────────
results = []

print("=" * 70)
print("ANÁLISIS DE DISCRIMINACIÓN DE SENSORES ")
print("=" * 70)

for s in sensors:
    n_vals = normal[s].dropna().values
    f_vals = fault[s].dropna().values

    # KS test
    ks_stat, pval = ks_2samp(n_vals, f_vals)

    # Mean difference (safe)
    if abs(n_vals.mean()) > 1e-6:
        diff_pct = abs(n_vals.mean() - f_vals.mean()) / abs(n_vals.mean()) * 100
    else:
        diff_pct = 0

    # Overlap approximation (important for ML)
    lo = min(n_vals.min(), f_vals.min())
    hi = max(n_vals.max(), f_vals.max())
    x = np.linspace(lo, hi, 400)

    n_kde = kde_curve(n_vals, x)
    f_kde = kde_curve(f_vals, x)

    overlap = np.trapz(np.minimum(n_kde, f_kde), x)

    results.append({
        "sensor": s,
        "KS": ks_stat,
        "pval": pval,
        "mean_diff_%": diff_pct,
        "overlap": overlap
    })

    print(f"{sensor_labels[s]:<20} | KS={ks_stat:.3f} | overlap={overlap:.3f} | Δmean={diff_pct:.1f}%")

# ── Ranking ───────────────────────────────────────────────────────────────────
res_df = pd.DataFrame(results).sort_values(by="KS", ascending=False)

print("\n" + "=" * 70)
print("CLASIFICACIÓN: PODER DISCRIMINATORIO")
print("=" * 70)

for _, r in res_df.iterrows():
    ks = r["KS"]
    overlap = r["overlap"]

    if ks > 0.2 and overlap < 0.7:
        level = "ALTA"
    elif ks > 0.1:
        level = "MEDIO"
    else:
        level = "BAJA"

    print(f"[{level}] {sensor_labels[r['sensor']]:<20} | KS={ks:.3f} | overlap={overlap:.3f}")

# ── Visualization (unchanged structure, improved clarity) ─────────────────────
fig = plt.figure(figsize=(20, 22))
outer = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.3)

for i, sensor in enumerate(sensors):
    row, col = divmod(i, 3)
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, col],
                                             width_ratios=[3, 1])

    ax_kde = fig.add_subplot(inner[0])
    ax_box = fig.add_subplot(inner[1])

    n_vals = normal[sensor].dropna().values
    f_vals = fault[sensor].dropna().values

    lo, hi = min(n_vals.min(), f_vals.min()), max(n_vals.max(), f_vals.max())
    x = np.linspace(lo, hi, 400)

    ax_kde.plot(x, kde_curve(n_vals, x), label="Normal")
    ax_kde.plot(x, kde_curve(f_vals, x), label="Falla")

    ax_kde.set_title(sensor_labels[sensor])
    if i == 0:
        ax_kde.legend()

    ax_box.boxplot([n_vals, f_vals])
    ax_box.set_xticklabels(["N", "F"])

plt.savefig("eda_step2_improved.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ Saved: eda_step2_improved.png")