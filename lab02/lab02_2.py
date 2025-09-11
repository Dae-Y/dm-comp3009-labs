"""
Data Mining Lab02 - Data Preparation
Daehwan Yeo

Task 2: Q3 from Chapter 2 of Aggarwal
Working with the Arrhythmia dataset from the UCI Machine Learning Repository.

This script:
1) Loads the dataset (treating '?' as NaN).
2) Removes attributes with >80% missing data.
3) Detects and removes duplicate rows.
4) Replaces missing values for columns with <5% missing:
   - numeric: mean
   - categorical: mode
5) Discretizes:
   - att3 into 10 equi-width bins
   - att4 into 10 equi-depth (quantile) bins
   (Saves bar plots: ex2_bin_att3.png, ex2_bin_att4.png)
6) Standardizes all numeric attributes to mean 0, std 1.
7) Randomly samples 100 rows to test set, rest to train set, and saves CSVs.
"""

import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) Load the data
# -----------------------------
# Local file with header; convert '?' to NaN
df = pd.read_csv('arrhythmia.data.with.header.csv', na_values='?')
print("Loaded dataset.")
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# -----------------------------
# 2) Remove attributes with >80% missing values
# -----------------------------
def gt80_missing(df_in: pd.DataFrame):
    """
    Return list of columns with more than 80% missing values.
    """
    total_rows = len(df_in)
    thresh = 0.8 * total_rows
    return [c for c in df_in.columns if df_in[c].isna().sum() > thresh]

sparse_columns = gt80_missing(df)
print("\nColumns with >80% missing values:", sparse_columns)

df_cleaned = df.drop(columns=sparse_columns)
print("Original columns:", len(df.columns))
print("After dropping sparse columns:", len(df_cleaned.columns))

# -----------------------------
# 3) Detect and remove duplicate rows
# -----------------------------
dups_mask = df_cleaned.duplicated()
duplicate_rows = df_cleaned[dups_mask]
if not duplicate_rows.empty:
    print(f"\nFound {duplicate_rows.shape[0]} duplicate rows. Removing duplicates...")
else:
    print("\nNo duplicate rows found.")

df_nd = df_cleaned.drop_duplicates()
print("Shape after removing duplicates:", df_nd.shape)

# -----------------------------
# 4) Replace missing values where 0% < missing < 5%
# -----------------------------
def lt5_missing(df_in: pd.DataFrame):
    """
    Return list of columns with less than 5% (but more than 0%) missing entries.
    """
    total_rows = len(df_in)
    cols = []
    for c in df_in.columns:
        miss = df_in[c].isna().sum()
        if miss > 0 and (miss / total_rows) * 100 < 5:
            cols.append(c)
    return cols

low_missing_columns = lt5_missing(df_nd)
print("\nColumns with <5% missing values:", low_missing_columns)

df_imputed = df_nd.copy()

for col in low_missing_columns:
    if pd.api.types.is_numeric_dtype(df_imputed[col]):
        mean_value = df_imputed[col].mean()
        df_imputed[col] = df_imputed[col].fillna(mean_value)
        print(f"Filled missing values in numeric column '{col}' with mean={mean_value:.6f}")
    else:
        # Mode may be empty if column is entirely NaN (shouldn't happen here due to <5% filter)
        mode_series = df_imputed[col].mode(dropna=True)
        if mode_series.empty:
            # Fallback: fill with explicit placeholder
            df_imputed[col] = df_imputed[col].fillna("__MISSING__")
            print(f"Filled missing values in categorical column '{col}' with placeholder '__MISSING__'")
        else:
            mode_value = mode_series.iloc[0]
            df_imputed[col] = df_imputed[col].fillna(mode_value)
            print(f"Filled missing values in categorical column '{col}' with mode='{mode_value}'")

# -----------------------------
# 5) Discretization for att3 and att4
# -----------------------------
# We’ll do both on df_imputed to reflect cleaning/imputation above.
def safe_series(df_in: pd.DataFrame, col: str) -> pd.Series:
    if col not in df_in.columns:
        raise KeyError(f"Required column '{col}' not found in the dataset.")
    return df_in[col]

# a) Equi-width binning for att3 (10 bins)
try:
    att3 = safe_series(df_imputed, 'att3')
    # Ensure numeric (coerce in case header treated as object somewhere)
    att3 = pd.to_numeric(att3, errors='coerce')

    # pd.cut will ignore NaN automatically
    att3_bins_equi_width = pd.cut(att3, bins=10)
    print("\nEqui-width Bins (att3) counts:")
    print(att3_bins_equi_width.value_counts(sort=False))

    # Plot and save
    plt.figure()
    att3_bins_equi_width.value_counts(sort=False).plot(
        kind='bar', title='att3 - 10 Equi-width Bins'
    )
    plt.ylabel("Frequency")
    plt.xlabel("Bin Range")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('ex2_bin_att3.png')
    plt.close()
    print("Saved figure: ex2_bin_att3.png")

except KeyError as e:
    print(f"\n[Warning] {e}. Skipping att3 equi-width binning.")
except ValueError as e:
    print(f"\n[Warning] Could not bin 'att3': {e}")

# b) Equi-depth (quantile) binning for att4 (10 bins)
try:
    att4 = safe_series(df_imputed, 'att4')
    att4 = pd.to_numeric(att4, errors='coerce')

    # pd.qcut with duplicates='drop' to handle non-unique edges
    att4_bins_equi_depth = pd.qcut(att4, q=10, duplicates='drop')
    print("\nEqui-depth Bins (att4) counts:")
    print(att4_bins_equi_depth.value_counts(sort=False))

    # Plot and save
    plt.figure()
    att4_bins_equi_depth.value_counts(sort=False).plot(
        kind='bar', title='att4 - 10 Equi-depth Bins'
    )
    plt.ylabel("Frequency")
    plt.xlabel("Bin Range")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('ex2_bin_att4.png')
    plt.close()
    print("Saved figure: ex2_bin_att4.png")

except KeyError as e:
    print(f"\n[Warning] {e}. Skipping att4 equi-depth binning.")
except ValueError as e:
    print(f"\n[Warning] Could not bin 'att4': {e}")

# -----------------------------
# 6) Standardize numeric attributes (Z-score)
# -----------------------------
numeric_cols = df_imputed.select_dtypes(include='number').columns
print("\nNumeric columns to standardize:", list(numeric_cols))

df_standardized = df_imputed.copy()
# Avoid division by zero if a column has std == 0
std = df_imputed[numeric_cols].std(ddof=0).replace(0, np.nan)
df_standardized[numeric_cols] = (df_imputed[numeric_cols] - df_imputed[numeric_cols].mean()) / std
# If any columns had zero std, they will become NaN—fill back with 0 (all values equal => already standardized)
df_standardized[numeric_cols] = df_standardized[numeric_cols].fillna(0)

print("\nMeans after standardization (approx 0):")
print(df_standardized[numeric_cols].mean().round(3))
print("\nStds after standardization (approx 1):")
print(df_standardized[numeric_cols].std(ddof=0).round(3))

# -----------------------------
# 7) Random sampling (100 test rows) and save CSV
# -----------------------------
# We’ll split on the cleaned/imputed DataFrame (df_imputed).
n_rows = len(df_imputed)
test_size = min(100, n_rows)  # In case the dataset has < 100 rows
test = df_imputed.sample(n=test_size, random_state=42)
train = df_imputed.drop(index=test.index)

train.to_csv('arrythmia_train.csv', index=False)
test.to_csv('arrythmia_test.csv', index=False)

print(f"\nSaved arrythmia_train.csv (rows={len(train)}) and arrythmia_test.csv (rows={len(test)}).")
print("Done.")
