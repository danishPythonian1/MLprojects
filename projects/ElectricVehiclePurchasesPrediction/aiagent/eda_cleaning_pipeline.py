"""
EDA & Data Cleaning Pipeline
Kompetisi Kaggle: Predicting Electric Vehicle Purchases

Menjalankan:
1. Data understanding
2. Data quality check (missing, duplicate, outlier, konsistensi kategorikal)
3. Exploratory Data Analysis (distribusi, hubungan fitur-target, korelasi)
4. Data cleaning & penyimpanan dataset bersih
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
pd.set_option("display.width", 150)

INPUT_PATH = "/mnt/user-data/uploads/train.csv"
OUTPUT_CSV = "/mnt/user-data/outputs/train_cleaned.csv"
OUTPUT_FIG_DIR = "/mnt/user-data/outputs/figures"

import os
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

NUM_COLS = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
CAT_COLS = [
    "Gender", "City_Type", "Current_Car_Type",
    "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level",
]
TARGET = "Will_Buy_EV"

# ----------------------------------------------------------------------
# 1. DATA UNDERSTANDING
# ----------------------------------------------------------------------
print("=" * 70)
print("1. DATA UNDERSTANDING")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)
print(f"Shape: {df.shape}")
print("\nTipe data:")
print(df.dtypes)
print("\n5 baris pertama:")
print(df.head())

# ----------------------------------------------------------------------
# 2. DATA QUALITY CHECK
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. DATA QUALITY CHECK")
print("=" * 70)

# 2a. Missing values
missing = df.isnull().sum()
print("\nMissing values per kolom:")
print(missing[missing > 0] if missing.sum() > 0 else "Tidak ada missing value.")

# 2b. Duplicate rows & id
n_dup_rows = df.duplicated().sum()
n_dup_id = df["id"].duplicated().sum()
print(f"\nBaris duplikat penuh : {n_dup_rows}")
print(f"id duplikat          : {n_dup_id}")
print(f"id unik & berurutan  : {df['id'].is_unique}, range {df['id'].min()}-{df['id'].max()}")

# 2c. Konsistensi kategorikal (cek whitespace / casing aneh)
print("\nNilai unik kolom kategorikal:")
for c in CAT_COLS + [TARGET]:
    uniques = df[c].unique().tolist()
    print(f"  {c}: {uniques}")
    # deteksi whitespace tersembunyi
    stripped_diff = df[c][df[c].astype(str) != df[c].astype(str).str.strip()]
    if len(stripped_diff) > 0:
        print(f"    -> Ditemukan {len(stripped_diff)} nilai dengan whitespace berlebih!")

# 2d. Validitas rentang nilai numerik (batas domain wajar)
range_checks = {
    "Age": (0, 120),
    "Annual_Income_USD": (0, None),
    "Daily_Commute_km": (0, None),
    "Number_of_Cars_Owned": (0, None),
    "Charging_Stations_Near_Home": (0, None),
    "Charging_Stations_Near_Work": (0, None),
    "Environmental_Concern_Level": (1, 5),
}
print("\nValidasi rentang nilai:")
invalid_mask_total = pd.Series(False, index=df.index)
for col, (lo, hi) in range_checks.items():
    mask = pd.Series(False, index=df.index)
    if lo is not None:
        mask |= df[col] < lo
    if hi is not None:
        mask |= df[col] > hi
    n_invalid = mask.sum()
    invalid_mask_total |= mask
    print(f"  {col}: {n_invalid} nilai di luar rentang wajar ({lo}, {hi})")
print(f"Total baris dengan >=1 nilai di luar rentang wajar: {invalid_mask_total.sum()}")

# 2e. Outlier detection (IQR) — untuk dokumentasi, tidak otomatis dihapus
print("\nOutlier (metode IQR) pada kolom numerik:")
outlier_summary = {}
for c in NUM_COLS:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[c] < low) | (df[c] > high)).sum()
    outlier_summary[c] = n_out
    print(f"  {c}: {n_out} outlier (batas: {low:.1f} - {high:.1f}), "
          f"data aktual: {df[c].min()} - {df[c].max()}")

# ----------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# 3a. Distribusi target
print("\nDistribusi target (Will_Buy_EV):")
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True).round(4))

plt.figure(figsize=(5, 4))
df[TARGET].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"])
plt.title("Distribusi Target: Will_Buy_EV")
plt.ylabel("Jumlah")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FIG_DIR}/01_target_distribution.png", dpi=120)
plt.close()

# 3b. Distribusi fitur numerik
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, c in enumerate(NUM_COLS):
    sns.histplot(df[c], kde=True, ax=axes[i], color="#4C72B0")
    axes[i].set_title(c)
for j in range(len(NUM_COLS), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig(f"{OUTPUT_FIG_DIR}/02_numeric_distributions.png", dpi=120)
plt.close()

# 3c. Distribusi fitur kategorikal
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, c in enumerate(CAT_COLS):
    df[c].value_counts().plot(kind="bar", ax=axes[i], color="#55A868")
    axes[i].set_title(c)
    axes[i].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FIG_DIR}/03_categorical_distributions.png", dpi=120)
plt.close()

# 3d. Hubungan fitur numerik terhadap target (boxplot)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, c in enumerate(NUM_COLS):
    sns.boxplot(data=df, x=TARGET, y=c, ax=axes[i], palette=["#4C72B0", "#DD8452"])
    axes[i].set_title(f"{c} vs {TARGET}")
for j in range(len(NUM_COLS), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig(f"{OUTPUT_FIG_DIR}/04_numeric_vs_target.png", dpi=120)
plt.close()

# 3e. Hubungan fitur kategorikal terhadap target (proporsi)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, c in enumerate(CAT_COLS):
    prop = pd.crosstab(df[c], df[TARGET], normalize="index")
    prop.plot(kind="bar", stacked=True, ax=axes[i], color=["#4C72B0", "#DD8452"])
    axes[i].set_title(f"{c} vs {TARGET}")
    axes[i].tick_params(axis="x", rotation=30)
    axes[i].legend(title=TARGET, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FIG_DIR}/05_categorical_vs_target.png", dpi=120)
plt.close()

# 3f. Korelasi antar fitur numerik
plt.figure(figsize=(8, 6))
corr = df[NUM_COLS].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Korelasi Antar Fitur Numerik")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FIG_DIR}/06_correlation_heatmap.png", dpi=120)
plt.close()

print(f"\nSemua grafik EDA disimpan di: {OUTPUT_FIG_DIR}")

# Insight ringkas: mean tiap fitur numerik per kelas target
print("\nRata-rata fitur numerik per kelas target:")
print(df.groupby(TARGET)[NUM_COLS].mean().round(2))

# ----------------------------------------------------------------------
# 4. DATA CLEANING
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. DATA CLEANING")
print("=" * 70)

df_clean = df.copy()

# 4a. Standardisasi string kategorikal (strip whitespace, konsisten)
for c in CAT_COLS + [TARGET]:
    df_clean[c] = df_clean[c].astype(str).str.strip()

# 4b. Drop duplicate rows (jika ada)
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
after = len(df_clean)
print(f"Baris dihapus karena duplikat: {before - after}")

# 4c. Drop baris dengan nilai di luar rentang wajar (jika ada)
mask_valid = pd.Series(True, index=df_clean.index)
for col, (lo, hi) in range_checks.items():
    if lo is not None:
        mask_valid &= df_clean[col] >= lo
    if hi is not None:
        mask_valid &= df_clean[col] <= hi
n_dropped_range = (~mask_valid).sum()
df_clean = df_clean[mask_valid]
print(f"Baris dihapus karena nilai di luar rentang wajar: {n_dropped_range}")

# 4d. Outlier: dipertahankan (bukan error input), hanya didokumentasikan
print("Outlier numerik TIDAK dihapus (dianggap variasi data alami), lihat ringkasan:")
for c, n in outlier_summary.items():
    print(f"  {c}: {n} outlier dipertahankan")

# 4e. Set id sebagai index (opsional, memudahkan tracing baris)
df_clean = df_clean.set_index("id")

# 4f. Final check
print(f"\nShape akhir dataset bersih: {df_clean.shape}")
print("Missing value tersisa:", df_clean.isnull().sum().sum())
print("Duplikat tersisa:", df_clean.duplicated().sum())

# ----------------------------------------------------------------------
# 5. SIMPAN OUTPUT
# ----------------------------------------------------------------------
df_clean.to_csv(OUTPUT_CSV)
print(f"\nDataset bersih disimpan di: {OUTPUT_CSV}")
