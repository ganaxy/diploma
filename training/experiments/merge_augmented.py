import os
import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

sys.path.insert(0, WORK_DIR)

BASE_CSV = os.path.join(WORK_DIR, "10k_fully_labeled_normalized.csv")
TOX_XLSX = os.path.join(WORK_DIR, "augmented toxic part 1.xlsx")
POS_XLSX = os.path.join(WORK_DIR, "Augmented positive.xlsx")
OUT_CSV  = os.path.join(WORK_DIR, "augmented_full_dataset.csv")

VALID_LABELS = {"POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"}

def load_aug(xlsx_path: str, expected_label: str, id_prefix: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, header=None)
    if raw.shape[1] < 2:
        raise ValueError(f"{xlsx_path}: expected ≥2 columns, got {raw.shape[1]}")
    raw = raw.iloc[:, :2].copy()
    raw.columns = ["text_light_clean", "label"]

    raw = raw.dropna(how="all").reset_index(drop=True)

    raw["label"] = raw["label"].astype(str).str.strip().str.upper()

    mismatched = raw[~raw["label"].eq(expected_label)]
    if len(mismatched):
        print(f"  ⚠  {xlsx_path}: {len(mismatched)} rows have label != "
              f"{expected_label}; dropping them.")
        print(f"     offending labels: {mismatched['label'].unique().tolist()}")
        raw = raw[raw["label"].eq(expected_label)].reset_index(drop=True)

    raw["text_light_clean"] = raw["text_light_clean"].astype(str).str.strip()
    before = len(raw)
    raw = raw[raw["text_light_clean"].str.len() > 0].reset_index(drop=True)
    if len(raw) < before:
        print(f"  ⚠  {xlsx_path}: dropped {before - len(raw)} empty-text rows")

    raw["text_normalized"] = raw["text_light_clean"].apply(normalize_mongolian)
    raw["source"] = "augmented"
    raw["split"]  = "train"
    raw["id"]     = [f"{id_prefix}_{i+1}" for i in range(len(raw))]
    return raw

def main():
    print(f"Loading base CSV: {BASE_CSV}")
    base = pd.read_csv(BASE_CSV, encoding="utf-8-sig")
    print(f"  base rows={len(base)}  columns={base.columns.tolist()}")
    print("✅ [Step 1.1 — base loaded]")

    print(f"Loading TOXIC augmented: {TOX_XLSX}")
    aug_tox = load_aug(TOX_XLSX, "TOXIC", "aug_toxic")
    print(f"  TOXIC aug rows: {len(aug_tox)}")

    print(f"Loading POSITIVE augmented: {POS_XLSX}")
    aug_pos = load_aug(POS_XLSX, "POSITIVE", "aug_pos")
    print(f"  POSITIVE aug rows: {len(aug_pos)}")
    print("✅ [Step 1.2 — augmented files loaded + normalized]")

    base_cols = base.columns.tolist()
    aug_all = pd.concat([aug_tox, aug_pos], ignore_index=True)

    for col in base_cols:
        if col in aug_all.columns:
            continue
        base_dtype = base[col].dtype
        if pd.api.types.is_numeric_dtype(base_dtype):
            aug_all[col] = 0
        else:
            aug_all[col] = ""

    aug_all = aug_all[base_cols].copy()

    drop_cols = [c for c in base.columns if str(c).startswith("Unnamed:")]
    if drop_cols:
        print(f"Dropping {len(drop_cols)} empty 'Unnamed' columns: {drop_cols}")
        base = base.drop(columns=drop_cols)
        aug_all = aug_all.drop(columns=drop_cols, errors="ignore")

    print("\n── Train-split class distribution BEFORE merge ──")
    before_counts = (
        base[base["split"] == "train"]["label"].value_counts().reindex(
            ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"], fill_value=0
        )
    )
    print(before_counts.to_string())
    print(f"  total train: {int(before_counts.sum())}")

    merged = pd.concat([base, aug_all], ignore_index=True)
    print(f"\nMerged rows: {len(merged)} (base={len(base)} + aug={len(aug_all)})")
    print("✅ [Step 1.3 — merged dataframe assembled]")

    print("\n── Train-split class distribution AFTER merge ───")
    after_counts = (
        merged[merged["split"] == "train"]["label"].value_counts().reindex(
            ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"], fill_value=0
        )
    )
    print(after_counts.to_string())
    print(f"  total train: {int(after_counts.sum())}")

    print("\n── Δ summary ────────────────────────────────")
    print(f"| {'class':<14s} | {'before':>7s} | {'after':>7s} | {'delta':>7s} |")
    print(f"|{'-'*16}|{'-'*9}|{'-'*9}|{'-'*9}|")
    for cls in ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]:
        b = int(before_counts[cls])
        a = int(after_counts[cls])
        print(f"| {cls:<14s} | {b:>7d} | {a:>7d} | {a-b:+7d} |")

    for sp in ("val", "test"):
        b_sz = int((base["split"] == sp).sum())
        a_sz = int((merged["split"] == sp).sum())
        assert b_sz == a_sz, f"split {sp} size changed ({b_sz}→{a_sz})"
    print("\n✓ val / test split sizes unchanged")

    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ [Step 1.4 — saved merged dataset to {OUT_CSV}]")
    print(f"  final shape: {merged.shape}")
    print(f"  columns: {merged.columns.tolist()}")
    print("\n✅ STEP 1 complete.")

if __name__ == "__main__":
    main()
