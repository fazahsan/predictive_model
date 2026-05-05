# ============================================================
# EDA Paso 4 — Análisis de Series Temporales
# ============================================================

"""
Objetivo:
Analizar cómo evolucionan las señales de los sensores en el tiempo
y qué ocurre justo antes de un fallo (zona de degradación).

Paneles generados:
A) Timeline completo de sensores (1 año, 1 turbina)
B) Comportamiento medio antes del fallo (20h previas)
C) Degradación vs RUL (vida útil restante)
D) Frecuencia de fallos por mes
E) Tiempo entre fallos
F) Distribución de tipos de fallo por turbina

Importancia:
- Permite detectar patrones previos al fallo
- Base para modelos predictivos y RUL
"""

# ============================================================
# LIBRERÍAS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CARGA DE DATOS
# ============================================================

df = pd.read_csv("dataset_realistic.csv", parse_dates=["timestamp"])
df = df.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)

# ============================================================
# SELECCIÓN DE TURBINA
# ============================================================

TURBINE = "T001"
tdf = df[df["turbine_id"] == TURBINE].copy().reset_index(drop=True)

# Detectar inicio de fallos (transición 0 → 1)
tdf["prev_fallo"] = tdf["Fallo"].shift(1, fill_value=0)
fault_starts = tdf[(tdf["Fallo"] == 1) & (tdf["prev_fallo"] == 0)].copy()

# ============================================================
# SENSORES CLAVE
# ============================================================

KEY_SENSORS = ["Eficiencia", "Vibracion", "Temp_cojinete", "Presion_salida"]

SENSOR_LABELS = {
    "Eficiencia": "Eficiencia",
    "Vibracion": "Vibración",
    "Temp_cojinete": "Temperatura Cojinete",
    "Presion_salida": "Presión de Salida",
}

COLORS = ["#1a6fdb", "#e63946", "#2d9e5f", "#e08b00"]

# ============================================================
# PANEL C — DATOS RUL
# ============================================================

rul_df = df[df["RUL"].notna()].copy()
rul_df["RUL_int"] = rul_df["RUL"].round().astype(int)
rul_agg = rul_df.groupby("RUL_int")[KEY_SENSORS].mean()

# ============================================================
# PANEL D — FRECUENCIA DE FALLOS
# ============================================================

df["month"] = df["timestamp"].dt.to_period("M")
df["prev_fallo"] = df.groupby("turbine_id")["Fallo"].shift(1, fill_value=0)

fault_events = df[(df["Fallo"] == 1) & (df["prev_fallo"] == 0)].copy()

fault_by_month = fault_events.groupby(["month", "Tipo_fallo"]).size().unstack(fill_value=0)

fault_types_all = ["Fuga", "Termico", "Cojinete", "Condensador"]
for ft in fault_types_all:
    if ft not in fault_by_month.columns:
        fault_by_month[ft] = 0

fault_by_month = fault_by_month[fault_types_all]

# ============================================================
# IMPRESIÓN DE RESULTADOS
# ============================================================

print("=" * 65)
print("ANÁLISIS DE SERIES TEMPORALES")
print("=" * 65)

print(f"\n① Turbina {TURBINE}:")
print(f"   Total fallos     : {len(fault_starts)}")
print(f"   Rango fechas     : {tdf['timestamp'].min().date()} → {tdf['timestamp'].max().date()}")
print(f"   Fallos/semana    : {len(fault_starts) / (len(tdf)/24/7):.1f}")

print(f"\n② Tendencia hacia fallo (RUL → 0):")
for s in KEY_SENSORS:
    val_20 = rul_agg.loc[20, s] if 20 in rul_agg.index else np.nan
    val_0  = rul_agg.loc[0, s] if 0 in rul_agg.index else np.nan

    pct_change = (val_0 - val_20) / val_20 * 100
    direction = "↑" if pct_change > 0 else "↓"

    print(f"   {SENSOR_LABELS[s]:<25}: {direction}{abs(pct_change):.1f}%")

print(f"\n③ Fallos por mes:")
monthly_total = fault_by_month.sum(axis=1)
print(f"   Mínimo  : {monthly_total.min()}")
print(f"   Máximo  : {monthly_total.max()}")
print(f"   Media   : {monthly_total.mean():.1f}")

# ============================================================
# CONFIGURACIÓN VISUAL
# ============================================================

BG = "white"
DARK = "#1a1a2e"
BORDER = "#d0d7e3"
PANEL = "#f7f9fc"
FAULT_C = "#ff4757"

FT_COLORS = {
    "Fuga": "#1a6fdb",
    "Termico": "#e63946",
    "Cojinete": "#2d9e5f",
    "Condensador": "#e08b00",
}

fig = plt.figure(figsize=(22, 20), facecolor=BG)

gs = gridspec.GridSpec(4, 2, figure=fig,
                       height_ratios=[2.2, 1.6, 1.8, 1.6],
                       hspace=0.55, wspace=0.32)

def style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

# ============================================================
# PANEL A — TIMELINE COMPLETO
# ============================================================

ax_A = fig.add_subplot(gs[0, :])

for s, col in zip(KEY_SENSORS, COLORS):
    vals = tdf[s].ffill()
    norm = (vals - vals.min()) / (vals.max() - vals.min())
    ax_A.plot(tdf["timestamp"], norm, color=col, label=SENSOR_LABELS[s])

for _, row in fault_starts.iterrows():
    ax_A.axvline(row["timestamp"], color=FAULT_C, alpha=0.3)

style(ax_A, "A — Evolución temporal de sensores", ylabel="Valor normalizado")
ax_A.legend()

# ============================================================
# PANEL B — ANTES DEL FALLO
# ============================================================

ax_B = fig.add_subplot(gs[1, 0])

hours_grid = np.arange(0, 22, 1)
windows = {s: [] for s in KEY_SENSORS}

for _, frow in fault_starts.iterrows():
    fault_ts = frow["timestamp"]
    mask = (tdf["timestamp"] <= fault_ts) & \
           (tdf["timestamp"] >= fault_ts - pd.Timedelta(hours=25))

    window = tdf[mask].copy()
    window["hours_to_fault"] = (fault_ts - window["timestamp"]).dt.total_seconds() / 3600

    for s in KEY_SENSORS:
        ww = window[["hours_to_fault", s]].dropna()
        windows[s].append(ww)

for s, col in zip(KEY_SENSORS, COLORS):
    all_interp = []
    for w in windows[s]:
        if len(w) < 3:
            continue
        w = w.sort_values("hours_to_fault")
        interp = np.interp(hours_grid, w["hours_to_fault"], w[s])
        all_interp.append(interp)

    if not all_interp:
        continue

    arr = np.array(all_interp)
    mean = arr.mean(axis=0)

    ax_B.plot(hours_grid, mean, color=col, label=SENSOR_LABELS[s])

ax_B.invert_xaxis()
style(ax_B, "B — Señales antes del fallo", xlabel="Horas hasta fallo")
ax_B.legend()

# ============================================================
# PANEL C — DEGRADACIÓN VS RUL
# ============================================================

ax_C = fig.add_subplot(gs[1, 1])

for s, col in zip(KEY_SENSORS, COLORS):
    y = rul_agg[s]
    yn = (y - y.min()) / (y.max() - y.min())
    ax_C.plot(rul_agg.index, yn, color=col, label=SENSOR_LABELS[s])

ax_C.invert_xaxis()
style(ax_C, "C — Sensores vs RUL", xlabel="RUL (horas)")
ax_C.legend()

# ============================================================
# PANEL D — FRECUENCIA DE FALLOS
# ============================================================

ax_D = fig.add_subplot(gs[2, :])

x = np.arange(len(fault_by_month))
bottom = np.zeros(len(fault_by_month))

for ft in fault_types_all:
    vals = fault_by_month[ft].values
    ax_D.bar(x, vals, bottom=bottom, label=ft, color=FT_COLORS[ft])
    bottom += vals

ax_D.set_xticks(x)
ax_D.set_xticklabels([str(m) for m in fault_by_month.index], rotation=30)

style(ax_D, "D — Fallos por mes", ylabel="Número de fallos")
ax_D.legend()

# ============================================================
# PANEL E — TIEMPO ENTRE FALLOS
# ============================================================

ax_E = fig.add_subplot(gs[3, 0])

inter_arrivals = []
for tid in df["turbine_id"].unique():
    t_faults = fault_events[fault_events["turbine_id"] == tid]["timestamp"]
    diffs = t_faults.diff().dropna().dt.total_seconds() / 3600
    inter_arrivals.extend(diffs.tolist())

inter_arr = np.array(inter_arrivals)

ax_E.hist(inter_arr, bins=50)
style(ax_E, "E — Tiempo entre fallos", xlabel="Horas")

# ============================================================
# PANEL F — TIPOS DE FALLO POR TURBINA
# ============================================================

ax_F = fig.add_subplot(gs[3, 1])

turb_ft = fault_events.groupby(["turbine_id", "Tipo_fallo"]).size().unstack(fill_value=0)
turb_ft_pct = turb_ft.div(turb_ft.sum(axis=1), axis=0)

im = ax_F.imshow(turb_ft_pct.T.values, aspect="auto")

ax_F.set_yticks(range(len(fault_types_all)))
ax_F.set_yticklabels(fault_types_all)

style(ax_F, "F — Tipos de fallo por turbina")

plt.colorbar(im, ax=ax_F)

# ============================================================
# GUARDAR RESULTADO
# ============================================================

plt.savefig("eda_step4_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ Imagen guardada: eda_step4_timeseries.png")