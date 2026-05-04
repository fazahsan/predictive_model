"""
EDA Step 1 — Dataset Overview & Structure (Improved)
===================================================
Goal: Understand dataset structure, quality, and risks before modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv("dataset_realistic.csv", parse_dates=["timestamp"])

print("=" * 60)
print("1. SHAPE")
print("=" * 60)
print(f"  Filas   : {df.shape[0]:,}")
print(f"  Columnas : {df.shape[1]}")

# ── 2. Column info ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. COLUMNAS Y TIPOS DE DATOS")
print("=" * 60)
print(df.dtypes.to_string())

# ── 3. Missing values ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. VALORES FALTANTES")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(3)
mv = pd.DataFrame({"Recuento faltante": missing, "Faltante %": missing_pct})
mv_nonzero = mv[mv["Recuento faltante"] > 0]

if mv_nonzero.empty:
    print("   No se detectaron valores faltantes.")
else:
    print(mv_nonzero.to_string())

# ── 4. Basic statistics ───────────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
sensors = [c for c in numeric_cols if c not in ["Fallo", "RUL"]]

print("\n" + "=" * 60)
print("4.ESTADÍSTICAS DESCRIPTIVAS (numeric sensors)")
print("=" * 60)
print(df[sensors].describe().round(3).to_string())

# ── 5. Turbines & time range ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. TURBINAS Y RANGO DE TIEMPO")
print("=" * 60)

print(f"  turbinas únicas : {df['turbine_id'].nunique()}")
print(f"  Rango de fechas     : {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"  Duración        : {(df['timestamp'].max() - df['timestamp'].min()).days} días")

rows_per_turbine = df.groupby("turbine_id").size()
print(f"  Filas/turbina    : min={rows_per_turbine.min()}, "
      f"max={rows_per_turbine.max()}, avg={rows_per_turbine.mean():.0f}")

# ── 6. Temporal consistency  ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. VERIFICACIÓN DE CONSISTENCIA TEMPORAL")
print("=" * 60)

time_diffs = df.sort_values(["turbine_id", "timestamp"]) \
               .groupby("turbine_id")["timestamp"].diff().dropna()

print(time_diffs.describe())

print("\n Intervalos de muestreo más comunes:")
print(time_diffs.value_counts().head())

# ── 7. Class balance ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. EQUILIBRIO DE CLASE OBJETIVO")
print("=" * 60)

vc = df["Fallo"].value_counts()
for v, c in vc.items():
    print(f"  Fallo={v}: {c:,} rows ({c/len(df)*100:.2f}%)")

fault_types = df[df["Fallo"] == 1]["Tipo_fallo"].value_counts()
print("\n  Desglose del tipo de falla:")
for ft, cnt in fault_types.items():
    print(f"    {ft}: {cnt:,}")

print("\n  ⚠️ Se ha detectado un grave desequilibrio de clases → debe abordarse en la etapa de aprendizaje automático.")

# ── 8. RUL availability check ───────────────────────────────────────────
print("\n" + "=" * 60)
print("8. VERIFICACIÓN DE DISPONIBILIDAD DE RUL")
print("=" * 60)

rul_missing_by_class = df["RUL"].isnull().groupby(df["Fallo"]).mean()
print("  Índice RUL faltante por clase:")
print(rul_missing_by_class)

print("\n  Interpretación:")
print("  - RUL suele estar ausente en el funcionamiento normal.")
print("  - RUL aparece cerca de eventos de falla → comportamiento esperado")

# ── 9. Data leakage warning ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. COMPROBACIÓN DE FUGAS DE DATOS")
print("=" * 60)

print("  ⚠️Tipo_fallo solo se conoce después del fallo → NO debe utilizarse como característica de entrada.")
print("  ⚠️ Asegúrese de que RUL no se utilice como entrada al predecir fallos..")

# ── 10. Visualization ─────────────────────────────────────────────────────────
df_vis = df.copy()
df_vis["month"] = df_vis["timestamp"].dt.to_period("M")

fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
fig.suptitle("EDA Step 1 — Descripción general del conjunto de datos", fontsize=20, color="white",
             fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ACCENT = "#00d4ff"
WARN   = "#ff6b6b"
OK     = "#00e676"
GRID   = "#1e2a38"

def style_ax(ax, title):
    ax.set_facecolor(GRID)
    ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3a4a")

# Panel A — Missing values
ax0 = fig.add_subplot(gs[0, 0])
style_ax(ax0, "Valores faltantes (%)")

cols_miss = mv_nonzero["Faltante %"] if not mv_nonzero.empty else pd.Series([0], index=["None"])
ax0.barh(cols_miss.index.astype(str), cols_miss.values, color=ACCENT)
ax0.set_xlabel("Faltante %", color="white")

# Panel B — Class balance
ax1 = fig.add_subplot(gs[0, 1])
style_ax(ax1, "Equilibrio de clase")

ax1.pie(
    [vc[0], vc[1]],
    labels=["Normal", "Falla"],
    autopct="%1.2f%%",
    colors=[OK, WARN],
    startangle=90
)

# Panel C — Fault types
ax2 = fig.add_subplot(gs[0, 2])
style_ax(ax2, "Tipos de fallas")

ax2.barh(fault_types.index, fault_types.values, color=WARN)

# Panel D — Rows per turbine
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Filas por turbina")

ax3.bar(range(len(rows_per_turbine)), rows_per_turbine.values, color=ACCENT)

# Panel E — Coverage heatmap
ax4 = fig.add_subplot(gs[1, 1:])
style_ax(ax4, "Cobertura (Turbina × Mes)")

pivot = df_vis.groupby(["turbine_id", "month"]).size().unstack(fill_value=0)
cmap = LinearSegmentedColormap.from_list("coverage", ["#0d1117", ACCENT])

im = ax4.imshow(pivot.values, aspect="auto", cmap=cmap)

plt.colorbar(im, ax=ax4)

plt.savefig("eda_step1_overview_improved.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ Plot saved: eda_step1_overview_improved.png")