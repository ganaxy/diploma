import os
import re
import sys
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

sys.path.insert(0, WORK_DIR)

BASE_CSV = os.path.join(WORK_DIR, "relabeled_v7_normalized.csv")
POS_XLSX = os.path.join(WORK_DIR, "augmented_positive_v2.xlsx")
OUT_CSV  = os.path.join(WORK_DIR, "relabeled_v7_aug_v2_normalized.csv")

PREFIX_RE = re.compile(r"^\d+\.\s*")

def load_aug_v2(xlsx_path: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, header=None)
    if raw.shape[1] < 1:
        raise ValueError(f"{xlsx_path}: no columns")
    raw = raw.iloc[:, :1].copy()
    raw.columns = ["text_light_clean"]
    raw = raw.dropna(how="all").reset_index(drop=True)

    raw["text_light_clean"] = raw["text_light_clean"].astype(str).str.strip()

    n_with_prefix = raw["text_light_clean"].str.match(PREFIX_RE).sum()
    raw["text_light_clean"] = raw["text_light_clean"].str.replace(
        PREFIX_RE, "", regex=True
    )
    print(f"  stripped 'N. ' prefix from {int(n_with_prefix)} rows")

    before = len(raw)
    raw = raw[raw["text_light_clean"].str.len() > 0].reset_index(drop=True)
    if len(raw) < before:
        print(f"  ⚠  dropped {before - len(raw)} empty-text rows")

    raw["label"]           = "POSITIVE"
    raw["text_normalized"] = raw["text_light_clean"].apply(normalize_mongolian)
    raw["source"]          = "augmented"
    raw["split"]           = "train"
    raw["id"]              = [f"aug_pos_v2_{i+1}" for i in range(len(raw))]
    return raw

def main():
    print(f"Loading v7 base: {BASE_CSV}")
    base = pd.read_csv(BASE_CSV, encoding="utf-8-sig")
    print(f"  base rows={len(base)}  columns={base.columns.tolist()}")
    print("✅ [Step 1.1 — v7 base loaded]")

    print(f"\nLoading POSITIVE augmented v2: {POS_XLSX}")
    aug = load_aug_v2(POS_XLSX)
    print(f"  POSITIVE aug rows: {len(aug)}")
    print("✅ [Step 1.2 — augmented file loaded + normalized]")

    base_cols = base.columns.tolist()
    for col in base_cols:
        if col in aug.columns:
            continue
        base_dtype = base[col].dtype
        if pd.api.types.is_numeric_dtype(base_dtype):
            aug[col] = 0
        else:
            aug[col] = ""

    aug = aug[base_cols].copy()

    drop_cols = [c for c in base.columns if str(c).startswith("Unnamed:")]
    if drop_cols:
        print(f"Dropping {len(drop_cols)} 'Unnamed' columns: {drop_cols}")
        base = base.drop(columns=drop_cols)
        aug  = aug.drop(columns=drop_cols, errors="ignore")

    print("\n── Train-split class distribution BEFORE merge ──")
    classes = ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]
    before = (
        base[base["split"] == "train"]["label"].value_counts()
        .reindex(classes, fill_value=0)
    )
    print(before.to_string())
    print(f"  total train: {int(before.sum())}")

    merged = pd.concat([base, aug], ignore_index=True)
    print(f"\nMerged rows: {len(merged)} (base={len(base)} + aug={len(aug)})")
    print("✅ [Step 1.3 — merged dataframe assembled]")

    print("\n── Train-split class distribution AFTER merge ───")
    after = (
        merged[merged["split"] == "train"]["label"].value_counts()
        .reindex(classes, fill_value=0)
    )
    print(after.to_string())
    print(f"  total train: {int(after.sum())}")

    print("\n── Δ summary ────────────────────────────────")
    print(f"| {'class':<14s} | {'before':>7s} | {'after':>7s} | {'delta':>7s} |")
    print(f"|{'-'*16}|{'-'*9}|{'-'*9}|{'-'*9}|")
    for cls in classes:
        b = int(before[cls]); a = int(after[cls])
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
    print("\n✅ STEP 1 (aug_v2) complete.")

if __name__ == "__main__":
    main()
