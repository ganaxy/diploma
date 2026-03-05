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

BASE_CSV = os.path.join(WORK_DIR, "10k_fully_labeled_relabeled.csv")
MIS_CSV  = os.path.join(WORK_DIR, "mislabeled_rows_v6.csv")
OUT_CSV  = os.path.join(WORK_DIR, "10k_fully_labeled_relabeled_v6.csv")

VALID_LABELS = {"POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"}
LABEL_ORDER  = ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]

def fail(msg: str) -> None:
    print(f"ERROR: {msg}")
    raise SystemExit(1)

if not os.path.exists(BASE_CSV):
    fail(f"{BASE_CSV} not found")
if not os.path.exists(MIS_CSV):
    fail(f"{MIS_CSV} not found")

base = pd.read_csv(BASE_CSV, encoding="utf-8-sig")
unnamed = [c for c in base.columns if str(c).startswith("Unnamed:")]
if unnamed:
    base = base.drop(columns=unnamed)
print(f"Base (v6): {len(base)} rows × {len(base.columns)} cols")

mis = pd.read_csv(MIS_CSV, encoding="utf-8-sig")
print(f"mislabeled_rows_v6: {len(mis)} rows × {len(mis.columns)} cols")
print(f"  columns: {mis.columns.tolist()}")

if RELABEL_COL not in mis.columns:
    fail(f"{MIS_CSV} missing column {RELABEL_COL!r}")
if "id" not in mis.columns:
    fail(f"{MIS_CSV} missing 'id' column")

base["id"] = base["id"].astype(str)
mis["id"]  = mis["id"].astype(str)

mis[RELABEL_COL] = mis[RELABEL_COL].astype(str).str.strip().str.upper()

blank_mask = (mis[RELABEL_COL] == "") | (mis[RELABEL_COL].str.lower() == "nan")
n_blank = int(blank_mask.sum())
if n_blank:
    print(f"WARN: {n_blank} rows in {RELABEL_COL} are blank / NaN "
          f"(those rows will keep their original v6 label)")

nonblank = mis[~blank_mask]
invalid = nonblank[~nonblank[RELABEL_COL].isin(VALID_LABELS)]
if len(invalid):
    print(f"ERROR: {len(invalid)} rows have invalid {RELABEL_COL} values")
    print("  value counts:", invalid[RELABEL_COL].value_counts().to_dict())
    sample = invalid[["id", "true_label", "predicted_label",
                      RELABEL_COL]].head(10)
    print(sample.to_string(index=False))
    raise SystemExit(1)

print(f"✅ {len(nonblank)} rows have valid new labels "
      f"(of {len(mis)} total rows in mislabeled file)")

base_ids = set(base["id"])
missing_ids = [i for i in mis["id"] if i not in base_ids]
if missing_ids:
    fail(f"{len(missing_ids)} ids in mislabeled_rows_v6 not found in base "
         f"(examples: {missing_ids[:5]})")

if base["id"].duplicated().any():
    fail("duplicate ids in base CSV")
if mis["id"].duplicated().any():
    dup = mis[mis["id"].duplicated(keep=False)]
    print(f"ERROR: {len(dup)} duplicate ids in mislabeled_rows_v6:")
    print(dup[["id", RELABEL_COL]].head(10).to_string(index=False))
    raise SystemExit(1)

update_map = dict(zip(nonblank["id"], nonblank[RELABEL_COL]))
print(f"\nApplying {len(update_map)} label updates ...")

new_base = base.copy()
old_labels = new_base["label"].astype(str).str.strip().str.upper().copy()
new_base["label"] = old_labels.copy()

mask_update = new_base["id"].isin(update_map.keys())
new_base.loc[mask_update, "label"] = new_base.loc[mask_update, "id"].map(update_map)

actual_changed = int((new_base["label"] != old_labels).sum())
print(f"  rows touched (in update_map): {int(mask_update.sum())}")
print(f"  rows where label actually changed: {actual_changed}")
unchanged_relabels = int(mask_update.sum()) - actual_changed
if unchanged_relabels:
    print(f"  (of the {int(mask_update.sum())} touched, {unchanged_relabels} "
          f"kept their existing v6 label — relabaled_labels matched)")

if actual_changed:
    diff = new_base.loc[new_base["label"] != old_labels, ["id"]].copy()
    diff["old"] = old_labels.loc[diff.index].values
    diff["new"] = new_base.loc[diff.index, "label"].values
    diff["pair"] = diff["old"] + " → " + diff["new"]
    pair_cnt = diff["pair"].value_counts()
    print("\n── Change breakdown (old v6 → new v7, desc) ──")
    print(f"  {'pair':<32s} {'count':>6s} {'% of changes':>13s}")
    for p, c in pair_cnt.items():
        print(f"  {p:<32s} {c:>6d} {100.0*c/actual_changed:>12.2f}%")

    old_counts = old_labels.value_counts().reindex(LABEL_ORDER, fill_value=0)
    new_counts = new_base["label"].value_counts().reindex(LABEL_ORDER,
                                                          fill_value=0)
    print("\n── Net label delta per class ───────")
    print(f"  {'class':<14s} {'v6':>7s} {'v7':>7s} {'delta':>7s}")
    for cls in LABEL_ORDER:
        o = int(old_counts[cls]); n = int(new_counts[cls])
        print(f"  {cls:<14s} {o:>7d} {n:>7d} {n-o:>+7d}")

    print("\n── Changes by split ─────────────────")
    split_map = new_base.set_index("id")["split"]
    diff["split"] = diff["id"].map(split_map)
    split_cnt = diff["split"].value_counts().reindex(
        ["train", "val", "test"], fill_value=0)
    for sp in ("train", "val", "test"):
        print(f"  {sp:<6s}: {int(split_cnt[sp]):>5d}")

bad_labels = new_base[~new_base["label"].isin(VALID_LABELS)]
if len(bad_labels):
    fail(f"{len(bad_labels)} rows have invalid label values after merge")

print("\n── Per-split class distribution (v7) ──────────")
header = f"  {'split':<6s}" + "".join(f" {cls:>13s}" for cls in LABEL_ORDER) + f" {'total':>8s}"
print(header)
for sp in ("train", "val", "test"):
    sub = new_base[new_base["split"] == sp]
    c = sub["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    line = f"  {sp:<6s}" + "".join(f" {int(c[cls]):>13d}" for cls in LABEL_ORDER) + f" {len(sub):>8d}"
    print(line)
total_c = new_base["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
print(f"  {'ALL':<6s}" + "".join(f" {int(total_c[cls]):>13d}" for cls in LABEL_ORDER) + f" {len(new_base):>8d}")

new_base.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ Saved {len(new_base)} rows to {OUT_CSV}")
print(f"  columns: {new_base.columns.tolist()}")
