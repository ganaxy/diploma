import os
import sys
import numpy as np
import pandas as pd
import torch
import torch_directml
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

HF_NAME  = "tugstugi/bert-base-mongolian-cased"
CKPT_DIR = os.path.join(WORK_DIR, "best_mnbert_v7_model")
BASE_CSV = os.path.join(WORK_DIR, "relabeled_v7_normalized.csv")
OUT_CSV  = os.path.join(WORK_DIR, "mislabeled_rows_v7.csv")

LABEL_MAP   = {"POSITIVE": 0, "NEUTRAL": 1, "CONSTRUCTIVE": 2, "TOXIC": 3}
ID2LABEL    = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]
MAX_LENGTH  = 256
BATCH       = 32
CLASSIFIER_DROPOUT = 0.15

try:
    device = torch_directml.device()
    _ = torch.zeros(1, device=device)
    print(f"DirectML device initialized: {device}")
except Exception as e:
    print(f"ERROR: DirectML failed to initialize — {e}")
    raise SystemExit(1)
print("✅ [Step 3.1 — device ready]")

if not os.path.exists(BASE_CSV):
    print(f"ERROR: {BASE_CSV} not found"); raise SystemExit(1)
df = pd.read_csv(BASE_CSV, encoding="utf-8-sig")
unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
if unnamed:
    df = df.drop(columns=unnamed)
df["input_text"] = df.apply(
    lambda r: f"[{r['source']}] {str(r['text_normalized'] or '')}", axis=1
)
df["label_id"] = df["label"].map(LABEL_MAP)
print(f"Loaded v7 base: {len(df)} rows "
      f"(train={int((df.split=='train').sum())}, "
      f"val={int((df.split=='val').sum())}, "
      f"test={int((df.split=='test').sum())})")
print("✅ [Step 3.2 — data loaded]")

if not os.path.isdir(CKPT_DIR):
    print(f"ERROR: {CKPT_DIR} not found"); raise SystemExit(1)
tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)

config = AutoConfig.from_pretrained(HF_NAME, num_labels=4)
config.classifier_dropout = CLASSIFIER_DROPOUT
model = AutoModelForSequenceClassification.from_pretrained(HF_NAME, config=config)
model.resize_token_embeddings(len(tokenizer))
state = torch.load(os.path.join(CKPT_DIR, "pytorch_model.bin"),
                   map_location="cpu")
model.load_state_dict(state)
model.to(device); model.eval()
print(f"✅ [Step 3.3 — v7 checkpoint loaded]")

class InferDS(Dataset):
    def __init__(self, ids, mask):
        self.ids, self.mask = ids, mask
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.mask[i]}

print(f"Tokenizing {len(df)} rows at max_length={MAX_LENGTH} ...")
enc = tokenizer(df["input_text"].tolist(), padding="max_length",
                truncation=True, max_length=MAX_LENGTH,
                return_tensors="pt")
ds = InferDS(enc["input_ids"], enc["attention_mask"])
loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

all_preds = []
print(f"Running inference (batch_size={BATCH}) ...")
with torch.no_grad():
    for bi, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.append(preds)
        if (bi + 1) % 50 == 0 or (bi + 1) == len(loader):
            done = min((bi + 1) * BATCH, len(df))
            print(f"  batch {bi+1}/{len(loader)}  ({done}/{len(df)})")
all_preds = np.concatenate(all_preds).astype(int)
assert len(all_preds) == len(df)
df["pred_id"] = all_preds
df["predicted_label"] = df["pred_id"].map(ID2LABEL)
df["true_label"] = df["label"]
print("✅ [Step 3.4 — inference complete]")

overall_acc = (df["pred_id"] == df["label_id"]).mean()
print(f"\nOverall accuracy on v7 base (all splits): {overall_acc:.4f}")

mis = df[df["pred_id"] != df["label_id"]].copy()
n_mis = len(mis)
n_total = len(df)
print(f"Total mislabeled: {n_mis} / {n_total}  "
      f"({100.0*n_mis/n_total:.2f}%)")

out_cols = ["id", "text_light_clean", "true_label",
            "predicted_label", "split", "source"]
mis[out_cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"✅ [Step 3.5 — saved mislabeled rows → {OUT_CSV}]")

print("\n" + "═"*64)
print(" STEP 3 — MISLABELED ROWS REPORT (v7 checkpoint)")
print("═"*64)

print(f"\nTotal mislabeled     : {n_mis}")
print(f"Total rows evaluated : {n_total}")
print(f"Mislabel rate        : {100.0*n_mis/n_total:.2f}%")

print("\n── Breakdown by split ─────────────────────────")
for sp in ("train", "val", "test"):
    sp_total = int((df["split"] == sp).sum())
    sp_mis = int(((df["split"] == sp) & (df["pred_id"] != df["label_id"])).sum())
    rate = 100.0 * sp_mis / sp_total if sp_total else 0.0
    print(f"  {sp:<6s}: {sp_mis:>5d} / {sp_total:>5d}   "
          f"({rate:5.2f}% mislabel rate)")

mis["pair"] = mis["true_label"] + "→" + mis["predicted_label"]
pair_counts = mis["pair"].value_counts()
print("\n── Confusion pair counts (true → predicted, desc) ──")
print(f"  {'pair':<28s} {'count':>6s}   {'% of mis':>8s}")
for pair, cnt in pair_counts.items():
    pct = 100.0 * cnt / n_mis
    print(f"  {pair:<28s} {cnt:>6d}   {pct:>7.2f}%")

print("\n── Top 3 most confused pairs (5 examples each) ──")
top3 = pair_counts.head(3)
for pair, cnt in top3.items():
    print(f"\n  [{pair}]  n={cnt}")
    subset = mis[mis["pair"] == pair].sample(
        n=min(5, cnt), random_state=42
    )
    for r in subset.itertuples(index=False):
        txt = str(r.text_light_clean).replace("\n", " ")
        if len(txt) > 100:
            txt = txt[:97] + "..."
        print(f"    id={str(r.id):<6s} {r.true_label:<12s}→"
              f"{r.predicted_label:<12s}  {txt}")

print("\n── Per-class miss rate ───────────────────────")
print(f"  {'class':<14s} {'mislabeled':>10s} {'total':>7s}   {'miss rate':>9s}")
for cls in LABEL_NAMES:
    cls_total = int((df["label"] == cls).sum())
    cls_mis = int(((df["label"] == cls) & (df["pred_id"] != df["label_id"])).sum())
    rate = 100.0 * cls_mis / cls_total if cls_total else 0.0
    print(f"  {cls:<14s} {cls_mis:>10d} {cls_total:>7d}   {rate:>8.2f}%")

print("\n✅ STEP 3 complete.")
