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
POS_XLSX = os.path.join(WORK_DIR, "Augmented positive.xlsx")
OUT_CSV  = os.path.join(WORK_DIR, "augmented_pos700_dataset.csv")

def load_pos_aug(xlsx_path: str, n: int) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, header=None)
    if raw.shape[1] < 2:
        raise ValueError(f"{xlsx_path}: expected ≥2 columns, got {raw.shape[1]}")
    raw = raw.iloc[:, :2].copy()
    raw.columns = ["text_light_clean", "label"]
    raw = raw.dropna(how="all").reset_index(drop=True)

    raw["label"] = raw["label"].astype(str).str.strip().str.upper()
    raw["text_light_clean"] = raw["text_light_clean"].astype(str).str.strip()

    before = len(raw)
    raw = raw[raw["text_light_clean"].str.len() > 0].reset_index(drop=True)
    if len(raw) < before:
        print(f"⚠  dropped {before - len(raw)} empty-text rows from {xlsx_path}")

    bad = raw[raw["label"] != "POSITIVE"]
    if len(bad):
        print(f"ERROR: {len(bad)} rows in {xlsx_path} have label != POSITIVE")
        print(f"  offending labels: {bad['label'].value_counts().to_dict()}")
        raise SystemExit(1)

    if len(raw) < n:
        print(f"ERROR: only {len(raw)} valid POSITIVE rows in {xlsx_path}, "
              f"need {n}")
        raise SystemExit(1)

    raw = raw.iloc[:n].copy().reset_index(drop=True)
    print(f"  took first {n} of {len(raw) if len(raw)>=n else len(raw)} "
          f"valid POSITIVE rows (file has more)")

    raw["text_normalized"] = raw["text_light_clean"].apply(normalize_mongolian)
    raw["source"] = "augmented"
    raw["split"]  = "train"
    raw["id"]     = [f"aug_pos_{i+1}" for i in range(len(raw))]
    return raw

def main():
    print(f"Loading base CSV: {BASE_CSV}")
    base = pd.read_csv(BASE_CSV, encoding="utf-8-sig")
    unnamed = [c for c in base.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        base = base.drop(columns=unnamed)
    print(f"  base rows={len(base)}  columns={base.columns.tolist()}")
    print("✅ [Step 1.1 — base loaded]")

    print(f"Loading POSITIVE augmented: {POS_XLSX} (take first {AUG_N})")
    aug_pos = load_pos_aug(POS_XLSX, AUG_N)
    assert len(aug_pos) == AUG_N
    print(f"  aug rows: {len(aug_pos)} (all verified POSITIVE)")
    print("✅ [Step 1.2 — POSITIVE augmented loaded + normalized]")

    base_cols = base.columns.tolist()
    for col in base_cols:
        if col in aug_pos.columns:
            continue
        if pd.api.types.is_numeric_dtype(base[col].dtype):
            aug_pos[col] = 0
        else:
            aug_pos[col] = ""
    aug_pos = aug_pos[base_cols].copy()

    print("\n── Train-split class distribution BEFORE merge ──")
    before_counts = (
        base[base["split"] == "train"]["label"].value_counts().reindex(
            ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"], fill_value=0
        )
    )
    print(before_counts.to_string())
    print(f"  total train: {int(before_counts.sum())}")

    merged = pd.concat([base, aug_pos], ignore_index=True)
    print(f"\nMerged rows: {len(merged)} (base={len(base)} + aug={len(aug_pos)})")
    print("✅ [Step 1.3 — merged]")

    print("\n── Train-split class distribution AFTER merge ───")
    after_counts = (
        merged[merged["split"] == "train"]["label"].value_counts().reindex(
            ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"], fill_value=0
        )
    )
    print(after_counts.to_string())
    print(f"  total train: {int(after_counts.sum())}")

    pos_before = int(before_counts["POSITIVE"])
    pos_after  = int(after_counts["POSITIVE"])
    assert pos_after == pos_before + AUG_N, \
        f"POSITIVE delta mismatch: {pos_before}+{AUG_N} != {pos_after}"
    print(f"\n✓ POSITIVE delta verified: {pos_before} + {AUG_N} = {pos_after}")

    print("\n── Δ summary ────────────────────────────────")
    print(f"| {'class':<14s} | {'before':>7s} | {'after':>7s} | {'delta':>7s} |")
    print(f"|{'-'*16}|{'-'*9}|{'-'*9}|{'-'*9}|")
    for cls in ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]:
        b = int(before_counts[cls]); a = int(after_counts[cls])
        print(f"| {cls:<14s} | {b:>7d} | {a:>7d} | {a-b:+7d} |")

    for sp in ("val", "test"):
        b_sz = int((base["split"] == sp).sum())
        a_sz = int((merged["split"] == sp).sum())
        assert b_sz == a_sz, f"split {sp} size changed"
    print("\n✓ val / test split sizes unchanged")

    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ [Step 1.4 — saved to {OUT_CSV}]")
    print(f"  final shape: {merged.shape}")
    print("\n✅ STEP 1 complete.")

if __name__ == "__main__":
    main()
