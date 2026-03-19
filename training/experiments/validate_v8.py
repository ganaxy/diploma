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

NEW_CSV = os.path.join(WORK_DIR, "10k_fully_labeled_relabeled_v7.csv")
PREV_CSV = os.path.join(WORK_DIR, "relabeled_v7_normalized.csv")
OUT_CSV = os.path.join(WORK_DIR, "relabeled_v8_normalized.csv")

REQUIRED_COLS = ["id", "text_light_clean", "label", "source", "split"]
VALID_LABELS  = {"POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"}
VALID_SPLITS  = {"train", "val", "test"}
LABEL_ORDER   = ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]

def fail(msg: str) -> None:
    print(f"ERROR: {msg}")
    raise SystemExit(1)

if not os.path.exists(NEW_CSV):
    fail(f"{NEW_CSV} not found")
df = pd.read_csv(NEW_CSV, encoding="utf-8-sig")
unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
if unnamed:
    df = df.drop(columns=unnamed)
    print(f"  dropped {len(unnamed)} Unnamed artifacts: {unnamed}")
print(f"Loaded {NEW_CSV}: rows={len(df)} cols={df.columns.tolist()}")
print("✅ [Step 1.1 — new CSV loaded]")

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    fail(f"missing required columns: {missing}")
print(f"✅ [Step 1.2 — required columns present: {REQUIRED_COLS}]")

df["label"] = df["label"].astype(str).str.strip().str.upper()
bad_labels = df[~df["label"].isin(VALID_LABELS)]
if len(bad_labels):
    print(f"ERROR: {len(bad_labels)} rows have invalid label values")
    print(bad_labels["label"].value_counts().to_dict())
    raise SystemExit(1)
print(f"✅ [Step 1.3 — all labels ∈ {sorted(VALID_LABELS)}]")

df["split"] = df["split"].astype(str).str.strip().str.lower()
bad_splits = df[~df["split"].isin(VALID_SPLITS)]
if len(bad_splits):
    print(f"ERROR: {len(bad_splits)} rows have invalid split values")
    print(bad_splits["split"].value_counts().to_dict())
    raise SystemExit(1)
print(f"✅ [Step 1.4 — all splits ∈ {sorted(VALID_SPLITS)}]")

df["text_light_clean"] = df["text_light_clean"].astype(str)
empty_txt = int((df["text_light_clean"].str.strip().str.len() == 0).sum())
if empty_txt:
    print(f"WARN: {empty_txt} rows have empty text_light_clean")
print(f"✅ [Step 1.5 — text_light_clean dtype OK "
      f"(empty rows: {empty_txt})]")

print("\n── Per-split class distribution (v8) ──────────")
header = f"  {'split':<6s}" + "".join(f" {cls:>13s}" for cls in LABEL_ORDER) + f" {'total':>8s}"
print(header)
for sp in ("train", "val", "test"):
    sub = df[df["split"] == sp]
    counts = sub["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    line = f"  {sp:<6s}" + "".join(f" {int(counts[cls]):>13d}" for cls in LABEL_ORDER) + f" {len(sub):>8d}"
    print(line)
total_counts = df["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
print(f"  {'ALL':<6s}" + "".join(f" {int(total_counts[cls]):>13d}" for cls in LABEL_ORDER) + f" {len(df):>8d}")

print("\n── Label changes vs relabeled_v7_normalized.csv (previous round) ──")
if not os.path.exists(PREV_CSV):
    print(f"  WARN: {PREV_CSV} not found — skipping comparison")
else:
    prev = pd.read_csv(PREV_CSV, encoding="utf-8-sig")
    prev_unnamed = [c for c in prev.columns if str(c).startswith("Unnamed:")]
    if prev_unnamed:
        prev = prev.drop(columns=prev_unnamed)
    prev["label"] = prev["label"].astype(str).str.strip().str.upper()
    prev["id"] = prev["id"].astype(str)
    df["id"] = df["id"].astype(str)

    merged = df[["id", "label", "split"]].merge(
        prev[["id", "label"]].rename(columns={"label": "old_label"}),
        on="id", how="left", validate="one_to_one"
    )
    missing_ids = int(merged["old_label"].isna().sum())
    if missing_ids:
        print(f"  WARN: {missing_ids} ids in new file not found in previous")

    cmp = merged.dropna(subset=["old_label"]).copy()
    cmp["changed"] = cmp["label"] != cmp["old_label"]
    n_changed = int(cmp["changed"].sum())
    n_total = len(cmp)
    print(f"  total comparable : {n_total}")
    print(f"  unchanged        : {n_total - n_changed}  "
          f"({100.0*(n_total-n_changed)/n_total:.2f}%)")
    print(f"  changed          : {n_changed}  "
          f"({100.0*n_changed/n_total:.2f}%)")

    if n_changed:
        pair = cmp[cmp["changed"]].copy()
        pair["pair"] = pair["old_label"] + " → " + pair["label"]
        pair_cnt = pair["pair"].value_counts()
        print("\n  ── Change breakdown (prev → new, desc) ──")
        print(f"    {'pair':<32s} {'count':>6s} {'% of changed':>13s}")
        for p, c in pair_cnt.items():
            print(f"    {p:<32s} {c:>6d} {100.0*c/n_changed:>12.2f}%")

        print("\n  ── Net label delta per class ───────")
        old_counts = cmp["old_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
        new_counts = cmp["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
        print(f"    {'class':<14s} {'prev':>7s} {'new':>7s} {'delta':>7s}")
        for cls in LABEL_ORDER:
            o = int(old_counts[cls]); n = int(new_counts[cls])
            print(f"    {cls:<14s} {o:>7d} {n:>7d} {n-o:>+7d}")

        print("\n  ── Changes concentrated in which splits ──")
        split_cnt = pair["split"].value_counts().reindex(
            ["train", "val", "test"], fill_value=0)
        for sp in ("train", "val", "test"):
            c = int(split_cnt[sp])
            denom = int((df["split"] == sp).sum())
            rate = 100.0 * c / denom if denom else 0.0
            print(f"    {sp:<6s}: {c:>5d}  ({rate:5.2f}% of that split)")
print("✅ [Step 1.6 — label comparison done]")

print("\nApplying normalize_mongolian() to text_light_clean ...")
df["text_normalized"] = df["text_light_clean"].apply(normalize_mongolian)
changed_norm = int((df["text_normalized"] != df["text_light_clean"]).sum())
print(f"  rows where normalization changed text: {changed_norm} / {len(df)}")
print("✅ [Step 1.7 — text_normalized created]")

df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ [Step 1.8 — saved to {OUT_CSV}]")
print(f"  final shape: {df.shape}")
print(f"  columns: {df.columns.tolist()}")

print("\n✅ STEP 1 complete.  STOP — waiting for confirmation to run STEP 2.")
