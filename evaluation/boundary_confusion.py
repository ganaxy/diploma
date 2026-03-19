import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch_directml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

CKPT_DIR = os.path.join(WORK_DIR, "best_mnbert_model")
NORMALIZED_CSV = os.path.join(WORK_DIR, "10k_fully_labeled_normalized.csv")
OUT_CSV = os.path.join(WORK_DIR, "boundary_confusion.csv")

LABEL_MAP = {"POSITIVE": 0, "NEUTRAL": 1, "CONSTRUCTIVE": 2, "TOXIC": 3}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]

MAX_LENGTH = 256
BATCH_SIZE = 32

try:
    device = torch_directml.device()
    _ = torch.zeros(1, device=device)
    print(f"DirectML device initialized: {device}")
except Exception as e:
    print(f"ERROR: DirectML failed to initialize — {e}")
    print("Stopping. Do NOT fall back to CPU.")
    raise SystemExit(1)

if not os.path.isdir(CKPT_DIR):
    print(f"ERROR: checkpoint dir not found: {CKPT_DIR}")
    raise SystemExit(1)

print(f"Loading tokenizer + model from {CKPT_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
config = AutoConfig.from_pretrained(CKPT_DIR)
if getattr(config, "num_labels", None) != 4:
    config.num_labels = 4
model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR, config=config)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
model.eval()
print(f"✅ [Step 1 — checkpoint loaded, moved to DirectML]")

if not os.path.exists(NORMALIZED_CSV):
    print(f"ERROR: {NORMALIZED_CSV} not found. Run task1_normalize.py first.")
    raise SystemExit(1)

df = pd.read_csv(NORMALIZED_CSV, encoding="utf-8-sig")
required = {"id", "text_light_clean", "text_normalized", "label", "split", "source"}
missing = required - set(df.columns)
if missing:
    print(f"ERROR: missing columns in normalized CSV: {missing}")
    raise SystemExit(1)

df["input_text"] = df.apply(
    lambda r: f"[{r['source']}] {str(r['text_normalized'] or '')}", axis=1
)
df["label_id"] = df["label"].map(LABEL_MAP)

print(f"Rows for inference: {len(df)}  "
      f"(train={int((df['split']=='train').sum())}, "
      f"val={int((df['split']=='val').sum())}, "
      f"test={int((df['split']=='test').sum())})")
print(f"✅ [Step 2 — data loaded with text_normalized]")

class InferDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }

print(f"Tokenizing {len(df)} rows at max_length={MAX_LENGTH} ...")
enc = tokenizer(
    df["input_text"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
ds = InferDataset(enc["input_ids"], enc["attention_mask"])
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"✅ [Step 3 — tokenized]")

all_preds = []
print(f"Running inference on {len(df)} rows, batch_size={BATCH_SIZE} ...")
with torch.no_grad():
    for bi, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.append(preds)
        if (bi + 1) % 25 == 0 or (bi + 1) == len(loader):
            done = min((bi + 1) * BATCH_SIZE, len(df))
            print(f"  batch {bi+1}/{len(loader)}  ({done}/{len(df)} rows)")
all_preds = np.concatenate(all_preds, axis=0).astype(int)
assert len(all_preds) == len(df)
df["pred_id"] = all_preds
df["predicted_label"] = df["pred_id"].map(ID2LABEL)
df["true_label"] = df["label"]
print(f"✅ [Step 4 — inference complete]")

CONFUSION_PAIRS = [
    ("NEUTRAL",      "CONSTRUCTIVE", "NEUTRAL→CONSTRUCTIVE"),
    ("CONSTRUCTIVE", "NEUTRAL",      "CONSTRUCTIVE→NEUTRAL"),
    ("TOXIC",        "CONSTRUCTIVE", "TOXIC→CONSTRUCTIVE"),
    ("TOXIC",        "NEUTRAL",      "TOXIC→NEUTRAL"),
    ("CONSTRUCTIVE", "TOXIC",        "CONSTRUCTIVE→TOXIC"),
    ("NEUTRAL",      "TOXIC",        "NEUTRAL→TOXIC"),
]

def _pair_name(row):
    for true, pred, name in CONFUSION_PAIRS:
        if row["true_label"] == true and row["predicted_label"] == pred:
            return name
    return None

df["confusion_pair"] = df.apply(_pair_name, axis=1)
confused = df[df["confusion_pair"].notna()].copy()

print("\n" + "═" * 60)
print(" BOUNDARY CONFUSION ANALYSIS — ./best_mnbert_model/")
print("═" * 60)

overall_acc = (df["pred_id"] == df["label_id"]).mean()
print(f"\nOverall accuracy across ALL splits: {overall_acc:.4f}")
print(f"Total confused rows (boundary pairs only): {len(confused)} / {len(df)}")

print("\n── Confusion pair counts ─────────────────────")
for true, pred, name in CONFUSION_PAIRS:
    n = int((confused["confusion_pair"] == name).sum())
    print(f"  {name:<28s} {n}")

print("\n── Breakdown by pair × split ─────────────────")
pivot = (
    confused.groupby(["confusion_pair", "split"]).size().unstack(fill_value=0)
)
pair_order = [name for _, _, name in CONFUSION_PAIRS if name in pivot.index]
pivot = pivot.loc[pair_order]
for sp in ("train", "val", "test"):
    if sp not in pivot.columns:
        pivot[sp] = 0
pivot = pivot[["train", "val", "test"]]
pivot["total"] = pivot.sum(axis=1)
print(pivot.to_string())

print("\n── Top 5 sources among confused rows ─────────")
top_src = confused["source"].value_counts().head(5)
for src, cnt in top_src.items():
    total_for_src = int((df["source"] == src).sum())
    rate = 100.0 * cnt / total_for_src if total_for_src else 0.0
    print(f"  {src:<30s} {cnt:>5d}   ({rate:.1f}% of {total_for_src} rows from this source)")

print("\n── 20 random confused examples ──────────────────")
sample = confused.sample(n=min(20, len(confused)), random_state=42)
for row in sample.itertuples(index=False):
    txt = str(row.text_light_clean).replace("\n", " ")
    if len(txt) > 80:
        txt = txt[:77] + "..."
    print(f"  id={row.id:<6d} {row.true_label:<12s}→{row.predicted_label:<12s}  "
          f"{txt}")

out_cols = ["id", "text_light_clean", "true_label", "predicted_label",
            "source", "split"]
out_df = confused[out_cols + ["confusion_pair"]].copy()
out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ [Step 5 — saved {len(out_df)} confused rows to {OUT_CSV}]")
print("\n✅ TASK 3 complete.")
