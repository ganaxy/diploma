import os
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
OUT_CSV  = os.path.join(WORK_DIR, "relabeled_v7_preproc_normalized.csv")

def main():
    print(f"Loading v7 base: {BASE_CSV}")
    base = pd.read_csv(BASE_CSV, encoding="utf-8-sig")
    print(f"  base rows={len(base)}  cols={base.columns.tolist()}")

    print("\nRe-deriving text_normalized with normalize_mongolian_v2 ...")
    old = base["text_normalized"].astype(str).copy()
    base["text_normalized"] = base["text_light_clean"].astype(str).apply(
        normalize_mongolian_v2
    )
    n_changed = int((base["text_normalized"] != old).sum())
    print(f"  rows whose text_normalized changed: {n_changed}/{len(base)}")

    drop_cols = [c for c in base.columns if str(c).startswith("Unnamed:")]
    if drop_cols:
        print(f"Dropping 'Unnamed' columns: {drop_cols}")
        base = base.drop(columns=drop_cols)

    classes = ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]
    print("\n── Train-split class distribution (unchanged from v7) ──")
    train_counts = (
        base[base["split"] == "train"]["label"].value_counts()
        .reindex(classes, fill_value=0)
    )
    print(train_counts.to_string())
    print(f"  total train: {int(train_counts.sum())}")

    def has_token(s, tok): return tok in str(s)
    for tok in ("[EMOJI]", "[URL]", "[NUM]", "[HTML]"):
        n_train = int(base[base["split"]=="train"]["text_normalized"].apply(
            lambda s: has_token(s, tok)).sum())
        n_test  = int(base[base["split"]=="test"]["text_normalized"].apply(
            lambda s: has_token(s, tok)).sum())
        print(f"  rows containing {tok:8s}: train={n_train:>5d}  test={n_test:>5d}")

    base.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ saved to {OUT_CSV}")
    print(f"  final shape: {base.shape}")

if __name__ == "__main__":
    main()
