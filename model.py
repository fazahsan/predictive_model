# The predictive maintenance pipeline, integrating all steps from data loading to model training .
# ============================================================
# PREDICTIVE MAINTENANCE — ADVANCED PIPELINE (INTEGRATED)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import precision_recall_curve

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("dataset_realistic.csv", parse_dates=["timestamp"])
df = df.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)

print("Dataset:", df.shape)

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

WINDOW = 24
WINDOW_SHORT = 12

KEY_SENSORS = ["Eficiencia", "Vibracion", "Temp_cojinete", "Presion_salida"]

# ---------- Rolling statistics ----------
for col in KEY_SENSORS:
    grp = df.groupby("turbine_id")[col]

    df[f"{col}_mean"] = grp.transform(lambda x: x.rolling(WINDOW, min_periods=3).mean())
    df[f"{col}_std"]  = grp.transform(lambda x: x.rolling(WINDOW, min_periods=3).std())
    df[f"{col}_max"]  = grp.transform(lambda x: x.rolling(WINDOW, min_periods=3).max())
    df[f"{col}_min"]  = grp.transform(lambda x: x.rolling(WINDOW, min_periods=3).min())

# ---------- Trend (long + short) ----------
def slope(series):
    if len(series) < 5:
        return np.nan
    return np.polyfit(np.arange(len(series)), series, 1)[0]

for col in KEY_SENSORS:
    df[f"{col}_trend"] = df.groupby("turbine_id")[col].transform(
        lambda x: x.rolling(WINDOW, min_periods=5).apply(slope, raw=False)
    )

    df[f"{col}_trend_short"] = df.groupby("turbine_id")[col].transform(
        lambda x: x.rolling(WINDOW_SHORT, min_periods=5).apply(slope, raw=False)
    )

# ---------- Z-score (anomaly detection) ----------
for col in KEY_SENSORS:
    mean = df.groupby("turbine_id")[col].transform(lambda x: x.rolling(WINDOW, min_periods=5).mean())
    std  = df.groupby("turbine_id")[col].transform(lambda x: x.rolling(WINDOW, min_periods=5).std())

    df[f"{col}_zscore"] = (df[col] - mean) / (std + 1e-6)

# ---------- Deviation from baseline ----------
for col in KEY_SENSORS:
    baseline = df.groupby("turbine_id")[col].transform("median")
    df[f"{col}_dev"] = df[col] - baseline

# ---------- Delta (direction) ----------
for col in KEY_SENSORS:
    df[f"{col}_delta"] = df.groupby("turbine_id")[col].diff(6)

# ---------- Strong degradation direction ----------
df["eff_drop"]      = df.groupby("turbine_id")["Eficiencia"].diff(12)
df["vib_increase"]  = df.groupby("turbine_id")["Vibracion"].diff(12)
df["temp_increase"] = df.groupby("turbine_id")["Temp_cojinete"].diff(12)

# ---------- Distance to failure profile ----------
FAILURE_PROFILE = {
    "Eficiencia": 77.8,
    "Vibracion": 3.26,
    "Temp_cojinete": 79.6,
    "Presion_salida": 0.099
}

for col in KEY_SENSORS:
    df[f"{col}_dist_to_fail"] = abs(df[col] - FAILURE_PROFILE[col])

# ---------- Health index ----------
df["health_score"] = (
    - df["Eficiencia_zscore"] +
      df["Vibracion_zscore"] +
      df["Temp_cojinete_zscore"] +
      df["Presion_salida_zscore"]
)

# ---------- Lag ----------
for lag in [1, 3, 6]:
    for col in KEY_SENSORS:
        df[f"{col}_lag_{lag}"] = df.groupby("turbine_id")[col].shift(lag)

# ---------- Interaction ----------
df["vib_eff_ratio"] = df["Vibracion"] / (df["Eficiencia"] + 1e-6)
df["temp_vib"] = df["Temp_cojinete"] * df["Vibracion"]

# ---------- Recent failure (NO leakage) ----------
df["recent_fail"] = df.groupby("turbine_id")["Fallo"].shift(1)
df["recent_fail"] = df.groupby("turbine_id")["recent_fail"].transform(
    lambda x: x.rolling(24, min_periods=1).max()
).fillna(0)

# ============================================================
# 3. TARGET
# ============================================================

df["target"] = df.groupby("turbine_id")["Fallo"].shift(-6)

df_model = df[df["target"].notna()].copy()

print("After target filtering:", df_model.shape)

# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

split_date = df_model["timestamp"].quantile(0.7)

train = df_model[df_model["timestamp"] < split_date]
test  = df_model[df_model["timestamp"] >= split_date]

print("Train:", train.shape)
print("Test :", test.shape)

# ============================================================
# 5. FEATURES
# ============================================================

FEATURES = [c for c in df_model.columns if c not in [
    "timestamp", "Fallo", "target", "Tipo_fallo", "RUL", "turbine_id"
]]

X_train, y_train = train[FEATURES], train["target"]
X_test, y_test   = test[FEATURES], test["target"]

# ============================================================
# 6. IMBALANCE
# ============================================================

pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = neg / pos

print(f"Scale pos weight: {scale_pos_weight:.2f}")
print(f"Failures in Train: {pos} | Failures in Test: {y_test.sum()}")

# ============================================================
# 7. MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    scale_pos_weight=scale_pos_weight,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=1,
    reg_lambda=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel trained")

# ============================================================
# 8. THRESHOLD OPTIMIZATION
# ============================================================

y_prob = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

best_idx = np.argmax(f1)
best_threshold = thresholds[best_idx]

print(f"\nOptimal Threshold: {best_threshold:.3f}")
print(f"Precision: {precision[best_idx]:.3f}")
print(f"Recall   : {recall[best_idx]:.3f}")

y_pred = (y_prob > best_threshold).astype(int)

# ============================================================
# 9. EVALUATION
# ============================================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_
feat_imp = pd.Series(importance, index=FEATURES).sort_values(ascending=False)

plt.figure(figsize=(10,6))
feat_imp.head(20).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top Features (Failure Prediction)")
plt.tight_layout()
plt.show()

# ============================================================
# 11. SAVE MODEL
# ============================================================

import joblib
joblib.dump(model, "predictive_maintenance_model_v2.pkl")

print("\nModel saved")