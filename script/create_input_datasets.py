"""create_input_datasets.py — نسخهٔ «تقسیم رندوم + مرتب‌سازی درون‌گروه»

این اسکریپت فایل `meteo_asadabad.csv` را می‌خواند، رندوم ولی پایدار (random_state=42)
به نسبت 80 ٪ / 10 ٪ / 10 ٪ تقسیم می‌کند، سپس **در هر زیرمجموعه** سطرها را مجدّداً
بر اساس ستون `date` صعودی مرتب می‌کند تا توالی زمانی حفظ شود. خروجی‌ها دقیقاً با
نام و ساختار مورد انتظار HL‑VAE ذخیره می‌شود.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RAW       = SCRIPT_DIR / "meteo_asadabad.csv"      # فایل خام
OUT_DIR   = SCRIPT_DIR.parent / "data"
OUT_DIR.mkdir(exist_ok=True)

# 1) بارگذاری و تعریف ستون‌ها ---------------------------------------------------
df = pd.read_csv(RAW)
fixed_cols = ["station_id", "date"]               # X
Y_cols      = [c for c in df.columns if c not in fixed_cols]  # Y

# 2) تقسیم رندوم 80/10/10 -------------------------------------------------------
train_idx, temp_idx = train_test_split(df.index, test_size=0.20,
                                       random_state=42, shuffle=True)
val_idx,   test_idx = train_test_split(temp_idx,  test_size=0.50,
                                       random_state=42, shuffle=True)

# 3) تابع ذخیره -----------------------------------------------------------------

def save_split(idx, split, add_missing=True, missing_pct=0.30, target_col=-1):
    subset = df.loc[idx].sort_values("date")  # مرتب‌سازی داخل زیرمجموعه

    X = subset[fixed_cols]
    Y = subset[Y_cols].copy()

    X.to_csv(OUT_DIR / f"meteo_{split}_X.csv", index=False)
    Y.to_csv(OUT_DIR / f"meteo_{split}_Y.csv", index=False)

    true_mask = (~Y.isna()).astype(int)
    true_mask.to_csv(OUT_DIR / f"true_mask_{split}.csv", index=False, header=False)

    mask = true_mask.copy()
    col_name = Y_cols[target_col]  # rrr24

    if split == "training" and add_missing:
        rng = np.random.default_rng(0)
        present = mask.index[mask[col_name] == 1]
        drop_n  = int(missing_pct * len(present))
        to_drop = rng.choice(present, size=drop_n, replace=False)

        mask.loc[to_drop, col_name] = 0
        Y.loc[to_drop, col_name]    = np.nan
        Y.to_csv(OUT_DIR / f"meteo_{split}_Y.csv", index=False)

    if split == "test":
        # کل ستون rrr24 در تست پنهان می‌شود
        mask.loc[:, col_name] = 0

    mask.to_csv(OUT_DIR / f"mask_{split}.csv", index=False, header=False)

# 4) ذخیرهٔ فایل‌ها -------------------------------------------------------------
for name, idx in [("training", train_idx),
                  ("validation", val_idx),
                  ("test", test_idx)]:
    save_split(idx, name)

# 5) data_types.csv --------------------------------------------------------------
types = [
    "pos", # ff_max
    "pos",   # ffm
    "real",  # tmax
    "real",  # tmin
    "real",  # tm
    "pos", # umax
    "pos", # umin
    "real",  # td_m
    "pos",   # ewsm
    "pos"    # rrr24
]
assert len(types) == len(Y_cols), "Length of types list must match Y columns"
types_df = pd.DataFrame({
    'type': types,
    'dim': [1] * len(types),
    'nclass': [1] * len(types)
})
types_df.to_csv(OUT_DIR / "data_types.csv", index=False)
# 6) CREATE RANGE FILE ----------------------------------------------------------
def create_range_file():
    """Create a CSV file with min/max ranges for each Y variable."""
    print("Creating data_ranges.csv...")
    
    # Calculate min and max for each Y column across the entire dataset
    ranges_data = []
    for col in Y_cols:
        # Skip NaN values when calculating min/max
        col_data = df[col].dropna()
        if len(col_data) > 0:
            min_val = col_data.min()
            max_val = col_data.max()
        else:
            # Fallback if all values are NaN
            min_val = 0.0
            max_val = 1.0
        
        ranges_data.append({
            'variable': col,
            'min': min_val,
            'max': max_val
        })
    
    # Create DataFrame and save
    ranges_df = pd.DataFrame(ranges_data)
    range_file_path = OUT_DIR / "data_ranges.csv"
    ranges_df.to_csv(range_file_path, index=False)
    
    print(f"✅ Range file created: {range_file_path}")
    print("Ranges:")
    for _, row in ranges_df.iterrows():
        print(f"  {row['variable']}: [{row['min']:.2f}, {row['max']:.2f}]")

# Call the function to create the range file
create_range_file()

print("✅ data_ranges.csv created with min/max values for normalization")
print("✅ random 80/10/10 splits created & sorted by date")
