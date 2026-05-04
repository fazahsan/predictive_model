"""
EDA Step 3 — Correlation Analysis (Final)
=========================================
Goal:
- Understand relationships between sensors
- Detect how correlations change during failure
- Identify fault signatures for feature engineering

Includes:
✔ Pearson + Spearman correlation
✔ Normal vs Fault comparison
✔ Correlation shift (Δ)
✔ Stability checks
✔ Top changing relationships
✔ 4-panel visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("dataset_realistic.csv")

sensors = [
    "Carga_pct", "Flujo_masico", "Potencia", "Presion_entrada",
    "Temp_entrada", "Presion_salida", "Eficiencia", "Vibracion", "Temp_cojinete"
]

labels = [
    "Carga", "Masico\nFlujo", "Potencia", "Entrada\nPresion", "Entrada\nTemp",
    "Salida\nPresion", "Efic.", "Vibracion.", "Cojinete\nTemp"
]

# ── Split dataset ─────────────────────────────────────────────────────────────
normal  = df[df["Fallo"] == 0][sensors].dropna()
fault   = df[df["Fallo"] == 1][sensors].dropna()
overall = df[sensors].dropna()

# ── Sample size check ─────────────────────────────────────────────────────────
print("=" * 70)
print("VERIFICACIÓN DEL TAMAÑO DE LA MUESTRA")
print("=" * 70)
print(f"Muestras normales : {len(normal):,}")
print(f"Muestras fallas  : {len(fault):,}")

if len(fault) < 500:
    print("⚠️ Advertencia: Las correlaciones de fallas pueden presentar ruido (tamaño de muestra pequeño).")

# ── Correlation matrices ──────────────────────────────────────────────────────
corr_all_p    = overall.corr(method="pearson")
corr_normal_p = normal.corr(method="pearson")
corr_fault_p  = fault.corr(method="pearson")

corr_normal_s = normal.corr(method="spearman")
corr_fault_s  = fault.corr(method="spearman")

# ── Delta correlation (Pearson) ───────────────────────────────────────────────
delta = corr_fault_p.values - corr_normal_p.values
np.fill_diagonal(delta, np.nan)
delta_df = pd.DataFrame(np.abs(delta), index=sensors, columns=sensors)

# ── Print key insights ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PRINCIPALES CONCLUSIONES SOBRE LA CORRELACIÓN")
print("=" * 70)

# Strong correlations
print("\n① Correlaciones fuertes (estado normal, |r| > 0,5):")
for i in range(len(sensors)):
    for j in range(i+1, len(sensors)):
        r = corr_normal_p.iloc[i, j]
        if abs(r) > 0.5:
            print(f"   {labels[i].replace(chr(10),' '):<15} ↔ "
                  f"{labels[j].replace(chr(10),' '):<15}  r = {r:+.3f}")

# Biggest shifts
print("\n② Mayores cambios en la correlación (|Δr| > 0.2):")
for i in range(len(sensors)):
    for j in range(i+1, len(sensors)):
        d = abs(corr_fault_p.iloc[i, j] - corr_normal_p.iloc[i, j])
        if d > 0.2:
            rn = corr_normal_p.iloc[i, j]
            rf = corr_fault_p.iloc[i, j]
            print(f"   {labels[i].replace(chr(10),' '):<15} ↔ "
                  f"{labels[j].replace(chr(10),' '):<15}  "
                  f"N={rn:+.3f}  F={rf:+.3f}  Δ={d:.3f}")

# Top unstable pairs
delta_flat = delta_df.unstack().sort_values(ascending=False)
delta_flat = delta_flat[delta_flat.index.get_level_values(0) != delta_flat.index.get_level_values(1)]

print("\n③ Las 5 relaciones más inestables:")
for (s1, s2), val in delta_flat.head(5).items():
    print(f"   {s1:<18} ↔ {s2:<18}  |Δr| = {val:.3f}")

# ── Visualization ─────────────────────────────────────────────────────────────
BG      = "white"
DARK    = "#1a1a2e"
BORDER  = "#d0d7e3"
PANEL   = "#f7f9fc"

fig = plt.figure(figsize=(20, 18), facecolor=BG)
fig.suptitle("Paso 3 del EDA: Análisis de correlación (normal vs. falla)",
             fontsize=20, color=DARK, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.2)

cmap_corr = "RdBu_r"
cmap_delta = "Oranges"
norm_corr = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

def draw_heatmap(ax, data, title, cmap, norm=None, mask_diag=False):
    mat = data.values.copy()
    if mask_diag:
        np.fill_diagonal(mat, np.nan)

    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")

    n = len(data)
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if np.isnan(val):
                continue
            txt_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=txt_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=12)

    return im

# Panel A — Overall
ax0 = fig.add_subplot(gs[0, 0])
im0 = draw_heatmap(ax0, corr_all_p, "A: Correlación general", cmap_corr, norm_corr)
plt.colorbar(im0, ax=ax0)

# Panel B — Normal
ax1 = fig.add_subplot(gs[0, 1])
im1 = draw_heatmap(ax1, corr_normal_p, "B: Estado normal", cmap_corr, norm_corr)
plt.colorbar(im1, ax=ax1)

# Panel C — Fault
ax2 = fig.add_subplot(gs[1, 0])
im2 = draw_heatmap(ax2, corr_fault_p, "C: Estado de falla", cmap_corr, norm_corr)
plt.colorbar(im2, ax=ax2)

# Panel D — Delta
ax3 = fig.add_subplot(gs[1, 1])
im3 = draw_heatmap(ax3, delta_df, "D: Cambio de correlación |Δr|", cmap_delta, None, mask_diag=True)
plt.colorbar(im3, ax=ax3)

plt.savefig("eda_step3_final.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ Saved: eda_step3_final.png")