import os
import re
import pandas as pd
from src.config import (
    CSV_PATH, TEXT_COLUMN, LABEL_COLUMN, SPLIT_COLUMN, LABEL_NAMES,
)

def load_and_prepare(csv_path: str = CSV_PATH) -> dict:

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    print(f"[data] Loading {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"[data] Loaded {len(df)} rows, {len(df.columns)} columns")

    for col in list(df.columns):
        if col.strip() == "":
            df = df.drop(columns=[col])
            print(f"[data] Dropped blank column: '{col}'")

    for col in [TEXT_COLUMN, LABEL_COLUMN, SPLIT_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Available: {list(df.columns)}")

    before = len(df)
    df = df.dropna(subset=[TEXT_COLUMN])
    df = df[df[TEXT_COLUMN].str.strip() != ""]
    after = len(df)
    if before != after:
        print(f"[data] Dropped {before - after} rows with missing text")

    df[TEXT_COLUMN] = (
        df[TEXT_COLUMN]
        .astype(str)
        .str.strip()
        .apply(lambda t: re.sub(r"\s+", " ", t))
    )

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.upper()

    unexpected = set(df[LABEL_COLUMN]) - set(LABEL_NAMES)
    if unexpected:
        n_bad = df[LABEL_COLUMN].isin(unexpected).sum()
        print(f"[data] WARNING: dropping {n_bad} rows with unexpected labels: {unexpected}")
        df = df[df[LABEL_COLUMN].isin(LABEL_NAMES)]

    print(f"[data] Label distribution (all):")
    for label in LABEL_NAMES:
        count = (df[LABEL_COLUMN] == label).sum()
        print(f"       {label}: {count}")

    df[SPLIT_COLUMN] = df[SPLIT_COLUMN].astype(str).str.strip().str.lower()

    splits = {}
    for split_name in ["train", "val", "test"]:
        mask = df[SPLIT_COLUMN] == split_name
        split_df = df[mask].copy()
        if len(split_df) == 0:
            raise ValueError(f"Split '{split_name}' has 0 rows after cleaning!")
        splits[split_name] = {
            "X": split_df[TEXT_COLUMN].values,
            "y": split_df[LABEL_COLUMN].values,
            "df": split_df,
        }
        print(f"[data] {split_name}: {len(split_df)} rows")
        for label in LABEL_NAMES:
            count = (split_df[LABEL_COLUMN] == label).sum()
            print(f"       {label}: {count}")

    return splits

if __name__ == "__main__":
    splits = load_and_prepare()
    print("\n[data] Done. Splits ready.")
