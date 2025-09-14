"""create_input_datasets.py — نسخهٔ «تقسیم رندوم + مرتب‌سازی درون‌گروه»

این اسکریپت یک فایل ورودی (CSV یا Excel) را که با پارامتر `-input` مشخص شده می‌خواند،
رندوم ولی پایدار (random_state=42) به نسبت 80 ٪ / 10 ٪ / 10 ٪ تقسیم می‌کند، سپس
**در هر زیرمجموعه** سطرها را مجدّداً بر اساس ستون `date` صعودی مرتب می‌کند تا توالی
زمانی حفظ شود. خروجی‌ها دقیقاً با نام و ساختار مورد انتظار HL‑VAE ذخیره می‌شود.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Split dataset into train/validation/test")
parser.add_argument(
    "-input",
    dest="input_path",
    default=str(SCRIPT_DIR / "keramanshah_badrgerd_homil_input.csv"),
    help="Path to input dataset (CSV or Excel)"
)
args = parser.parse_args()

RAW = Path(args.input_path).expanduser().resolve()      # فایل خام
OUT_DIR   = SCRIPT_DIR.parent / "data"
OUT_DIR.mkdir(exist_ok=True)

# 1) بارگذاری و تعریف ستون‌ها ---------------------------------------------------
if RAW.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
    df = pd.read_excel(RAW)
else:
    df = pd.read_csv(RAW)

# Normalize column names: strip whitespace
df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

# Drop 'Unnamed' columns (pandas creates these when there are extra separators)
unnamed_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('Unnamed')]
# Also drop any columns that are entirely NaN (often created by malformed CSVs)
allnan_cols = [c for c in df.columns if df[c].isna().all()]

cols_to_drop = sorted(set(unnamed_cols + allnan_cols))
if cols_to_drop:
    print(f"Dropping columns not present in source or entirely NaN: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

fixed_cols = ["station_id", "date"]               # X
Y_cols      = [c for c in df.columns if c not in fixed_cols]  # Y

# 1.a) Min-max normalise Y columns in-place and store ranges
ranges = {}
for col in Y_cols:
    col_data = pd.to_numeric(df[col], errors="coerce")
    min_val = col_data.min()
    max_val = col_data.max()
    ranges[col] = {"min": float(min_val), "max": float(max_val)}
    denom = max_val - min_val if max_val != min_val else 1.0
    df[col] = (col_data - min_val) / denom

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
    col_name = Y_cols[target_col]  # Q

    if split == "training" and add_missing:
        rng = np.random.default_rng(0)
        present = mask.index[mask[col_name] == 1]
        drop_n  = int(missing_pct * len(present))
        to_drop = rng.choice(present, size=drop_n, replace=False)

        mask.loc[to_drop, col_name] = 0
        Y.loc[to_drop, col_name]    = np.nan
        Y.to_csv(OUT_DIR / f"meteo_{split}_Y.csv", index=False)

    if split == "test":
        # کل ستون Q در تست پنهان می‌شود
        mask.loc[:, col_name] = 0

    mask.to_csv(OUT_DIR / f"mask_{split}.csv", index=False, header=False)

# 4) ذخیرهٔ فایل‌ها -------------------------------------------------------------
for name, idx in [("training", train_idx),
                  ("validation", val_idx),
                  ("test", test_idx)]:
    save_split(idx, name)

# 5) data_types.csv --------------------------------------------------------------
# به طور پیش‌فرض تمام ستون‌های Y به عنوان "real" در نظر گرفته می‌شوند مگر اینکه
# نام ستون بیانگر مقادیر صرفاً مثبت باشد (مانند rrr24).
types = ["real"] * len(Y_cols)
for i, col in enumerate(Y_cols):
    if col.lower() == "rrr24":
        types[i] = "pos"

types_df = pd.DataFrame({
    'type': types,
    'dim': [1] * len(types),
    'nclass': [1] * len(types)
})
types_df.to_csv(OUT_DIR / "data_types.csv", index=False)
# 6) CREATE RANGE FILE ----------------------------------------------------------
def create_range_file():
    """Save min/max ranges (before normalisation) for later inverse scaling."""
    print("Creating data_ranges.csv...")

    ranges_df = pd.DataFrame([
        {"variable": col, "min": info["min"], "max": info["max"]}
        for col, info in ranges.items()
    ])
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
