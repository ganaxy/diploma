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

IN_CSV      = os.path.join(WORK_DIR, "relabeled_v9_normalized.csv")
STAGE1_CSV  = os.path.join(WORK_DIR, "stage1_dataset.csv")
STAGE2_CSV  = os.path.join(WORK_DIR, "stage2_dataset.csv")

LABEL_TO_STAGE1 = {
    "POSITIVE":     "POSITIVE",
    "NEUTRAL":      "NEUTRAL",
    "CONSTRUCTIVE": "NEGATIVE",
    "TOXIC":        "NEGATIVE",
}
STAGE1_ORDER = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
STAGE2_ORDER = ["CONSTRUCTIVE", "TOXIC"]
STAGE1_MAP   = {"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2}
STAGE2_MAP   = {"CONSTRUCTIVE": 0, "TOXIC": 1}

VALID_LABELS = set(LABEL_TO_STAGE1.keys())
VALID_SPLITS = {"train", "val", "test"}

def fail(msg: str):
    print(f"ERROR: {msg}")
    raise SystemExit(1)

if not os.path.exists(IN_CSV):
    fail(f"{IN_CSV} not found")
df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
if unnamed:
    df = df.drop(columns=unnamed)
print(f"Loaded {IN_CSV}: rows={len(df)} cols={df.columns.tolist()}")

for col in ("id", "text_normalized", "label", "split", "source"):
    if col not in df.columns:
        fail(f"required column missing: {col}")

df["label"] = df["label"].astype(str).str.strip().str.upper()
df["split"] = df["split"].astype(str).str.strip().str.lower()

bad_lbl = df[~df["label"].isin(VALID_LABELS)]
if len(bad_lbl):
    fail(f"{len(bad_lbl)} rows have invalid label")

bad_sp = df[~df["split"].isin(VALID_SPLITS)]
if len(bad_sp):
    fail(f"{len(bad_sp)} rows have invalid split")
print("✅ [Step 1.1 — v9 CSV loaded and validated]")

df["label_stage1"]    = df["label"].map(LABEL_TO_STAGE1)
df["label_stage1_id"] = df["label_stage1"].map(STAGE1_MAP)

print("\n── Stage 1 per-split class distribution (POSITIVE/NEUTRAL/NEGATIVE) ──")
header = f"  {'split':<6s}" + "".join(f" {c:>12s}" for c in STAGE1_ORDER) + f" {'total':>8s}"
print(header)
for sp in ("train", "val", "test"):
    sub = df[df["split"] == sp]
    cnt = sub["label_stage1"].value_counts().reindex(STAGE1_ORDER, fill_value=0)
    line = f"  {sp:<6s}" + "".join(f" {int(cnt[c]):>12d}" for c in STAGE1_ORDER) + f" {len(sub):>8d}"
    print(line)
tot = df["label_stage1"].value_counts().reindex(STAGE1_ORDER, fill_value=0)
print(f"  {'ALL':<6s}" + "".join(f" {int(tot[c]):>12d}" for c in STAGE1_ORDER) + f" {len(df):>8d}")
print("✅ [Step 1.2 — Stage 1 labels derived]")

stage2 = df[df["label"].isin(STAGE2_ORDER)].copy().reset_index(drop=True)
stage2["label_stage2"]    = stage2["label"]
stage2["label_stage2_id"] = stage2["label_stage2"].map(STAGE2_MAP)

print("\n── Stage 2 per-split class distribution (CONSTRUCTIVE/TOXIC) ──")
header = f"  {'split':<6s}" + "".join(f" {c:>14s}" for c in STAGE2_ORDER) + f" {'total':>8s}"
print(header)
for sp in ("train", "val", "test"):
    sub = stage2[stage2["split"] == sp]
    cnt = sub["label_stage2"].value_counts().reindex(STAGE2_ORDER, fill_value=0)
    line = f"  {sp:<6s}" + "".join(f" {int(cnt[c]):>14d}" for c in STAGE2_ORDER) + f" {len(sub):>8d}"
    print(line)
tot2 = stage2["label_stage2"].value_counts().reindex(STAGE2_ORDER, fill_value=0)
print(f"  {'ALL':<6s}" + "".join(f" {int(tot2[c]):>14d}" for c in STAGE2_ORDER) + f" {len(stage2):>8d}")
print("✅ [Step 1.3 — Stage 2 subset derived]")

s2_train = int((stage2["split"] == "train").sum())
s2_val   = int((stage2["split"] == "val").sum())
s2_test  = int((stage2["split"] == "test").sum())
print(f"\nStage 2 subset size: total={len(stage2)} "
      f"train={s2_train} val={s2_val} test={s2_test}")
print(f"  (that's {100.0*len(stage2)/len(df):.1f}% of the full v9 dataset)")
print("✅ [Step 1.4 — Stage 2 subset size reported]")

def leak_check(frame: pd.DataFrame, tag: str) -> None:
    ids_by_split = {
        sp: set(frame.loc[frame["split"] == sp, "id"].astype(str).tolist())
        for sp in ("train", "val", "test")
    }
    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val",   "test"),
    ]
    any_leak = False
    for a, b in pairs:
        inter = ids_by_split[a] & ids_by_split[b]
        if inter:
            print(f"  LEAK [{tag}] {a}∩{b}: {len(inter)} ids overlap")
            any_leak = True
        else:
            print(f"  OK   [{tag}] {a}∩{b}: 0 ids overlap")
    if any_leak:
        fail(f"{tag}: split leakage detected")

print("\n── Leakage check (Stage 1 — all 10k rows) ─────")
leak_check(df, "stage1")
print("\n── Leakage check (Stage 2 — CON+TOX subset) ───")
leak_check(stage2, "stage2")
print("✅ [Step 1.5 — no split leakage in either subset]")

keep_cols = [c for c in
             ["id", "text_light_clean", "text_normalized",
              "label", "split", "source",
              "label_stage1", "label_stage1_id",
              "label_stage2", "label_stage2_id"]
             if c in df.columns or c in stage2.columns]

df_out = df[[c for c in keep_cols if c in df.columns]].copy()
df_out.to_csv(STAGE1_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ [Step 1.6 — saved Stage 1 full dataset → {STAGE1_CSV}] "
      f"shape={df_out.shape}")

s2_out = stage2[[c for c in keep_cols if c in stage2.columns]].copy()
s2_out.to_csv(STAGE2_CSV, index=False, encoding="utf-8-sig")
print(f"✅ [Step 1.7 — saved Stage 2 filtered subset → {STAGE2_CSV}] "
      f"shape={s2_out.shape}")

print("\n✅ STEP 1 (2-stage) complete.  "
      "STOP — waiting for confirmation to run STEP 2 (Stage 1 training).")
