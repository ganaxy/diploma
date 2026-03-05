import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import torch
import torch_directml
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
import warnings
warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

HF_NAME     = "tugstugi/bert-base-mongolian-cased"
MERGED_CSV  = os.path.join(WORK_DIR, "augmented_full_dataset.csv")
SAVE_DIR    = os.path.join(WORK_DIR, "best_mnbert_v3_model")
PRED_CSV    = os.path.join(WORK_DIR, "test_predictions_v3.csv")

LABEL_MAP  = {"POSITIVE": 0, "NEUTRAL": 1, "CONSTRUCTIVE": 2, "TOXIC": 3}
ID2LABEL   = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES= ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]

LR              = 3e-5
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.06
GRAD_CLIP       = 1.0
CLASSIFIER_DROPOUT = 0.15
MAX_EPOCHS      = 15
PATIENCE        = 4
VAL_TEST_BATCH  = 32
SEED            = 42

V1_METRICS = {
    "accuracy": 0.7656, "macro_f1": 0.7235, "weighted_f1": 0.7644,
    "POSITIVE": 0.6358, "NEUTRAL": 0.7295,
    "CONSTRUCTIVE": 0.8242, "TOXIC": 0.7045,
}
V2_METRICS = {
    "accuracy": 0.7567, "macro_f1": 0.7217, "weighted_f1": 0.7580,
    "POSITIVE": 0.6421, "NEUTRAL": 0.7275,
    "CONSTRUCTIVE": 0.8102, "TOXIC": 0.7069,
}

torch.manual_seed(SEED)
np.random.seed(SEED)

if os.path.isdir(SAVE_DIR) and os.listdir(SAVE_DIR):
    print(f"ERROR: {SAVE_DIR} already exists and is non-empty. "
          f"Stop and ask before overwriting.")
    raise SystemExit(1)
os.makedirs(SAVE_DIR, exist_ok=True)

try:
    device = torch_directml.device()
    _ = torch.zeros(1, device=device)
    print(f"DirectML device initialized: {device}")
except Exception as e:
    print(f"ERROR: DirectML failed to initialize — {e}")
    print("Stopping. Do NOT fall back to CPU.")
    raise SystemExit(1)

if not os.path.exists(MERGED_CSV):
    print(f"ERROR: {MERGED_CSV} not found. Run step1_merge_augmented.py first.")
    raise SystemExit(1)

df = pd.read_csv(MERGED_CSV, encoding="utf-8-sig")
required = {"id", "text_light_clean", "text_normalized", "label",
            "split", "source"}
missing = required - set(df.columns)
if missing:
    print(f"ERROR: missing columns: {missing}")
    raise SystemExit(1)

df["input_text"] = df.apply(
    lambda r: f"[{r['source']}] {str(r['text_normalized'] or '')}", axis=1
)
df["label_id"] = df["label"].map(LABEL_MAP)

split_counts = df["split"].value_counts().to_dict()
print(f"Rows: total={len(df)} | splits={split_counts}")

aug_mask = df["source"].astype(str) == "augmented"
print(f"Augmented rows (all in train): {int(aug_mask.sum())}")
assert int(((df["split"] != "train") & aug_mask).sum()) == 0, \
    "augmented rows leaked into val/test"
print(f"✅ [Step 2.1 — merged data loaded, train={int((df.split=='train').sum())} "
      f"val={int((df.split=='val').sum())} test={int((df.split=='test').sum())}]")

tokenizer = AutoTokenizer.from_pretrained(HF_NAME)
print(f"✅ [Step 2.2 — tokenizer loaded]")

print("Computing p95 token length on train split (augmented included)...")
train_texts = df[df["split"] == "train"]["input_text"].tolist()
tok_lens = []
B = 256
for i in range(0, len(train_texts), B):
    enc = tokenizer(train_texts[i:i+B], add_special_tokens=True,
                    truncation=False, padding=False)
    tok_lens.extend(len(x) for x in enc["input_ids"])
tok_lens = np.array(tok_lens, dtype=np.int64)
p50 = int(np.percentile(tok_lens, 50))
p95 = int(np.percentile(tok_lens, 95))
p99 = int(np.percentile(tok_lens, 99))
p_max = int(tok_lens.max())
print(f"  token lengths: p50={p50} p95={p95} p99={p99} max={p_max}")

if p95 <= 256:
    MAX_LENGTH = 256
    BATCH_SIZE = 16
else:
    MAX_LENGTH = 512
    BATCH_SIZE = 8
print(f"  → chosen max_length={MAX_LENGTH}, batch_size={BATCH_SIZE}")
print(f"✅ [Step 2.3 — max_length/batch_size decided]")

def encode_split(texts, labels):
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=MAX_LENGTH, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long)

class TextDataset(Dataset):
    def __init__(self, ids, mask, y):
        self.ids, self.mask, self.y = ids, mask, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i],
                "attention_mask": self.mask[i],
                "labels": self.y[i]}

print("Tokenizing splits ...")
tr_ids, tr_mask, tr_y = encode_split(
    df[df.split == "train"]["input_text"].tolist(),
    df[df.split == "train"]["label_id"].tolist(),
)
va_ids, va_mask, va_y = encode_split(
    df[df.split == "val"]["input_text"].tolist(),
    df[df.split == "val"]["label_id"].tolist(),
)
te_ids, te_mask, te_y = encode_split(
    df[df.split == "test"]["input_text"].tolist(),
    df[df.split == "test"]["label_id"].tolist(),
)
train_ds = TextDataset(tr_ids, tr_mask, tr_y)
val_ds   = TextDataset(va_ids, va_mask, va_y)
test_ds  = TextDataset(te_ids, te_mask, te_y)
print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
print(f"✅ [Step 2.4 — tokenized]")

train_labels_np = tr_y.numpy()
counts = np.bincount(train_labels_np, minlength=4).astype(np.float64)
print(f"Augmented-train label counts: "
      f"{dict(zip(LABEL_NAMES, counts.astype(int).tolist()))}")
total = counts.sum()
class_weights = (total / (len(counts) * counts))
print(f"Class weights (CE): "
      f"{dict(zip(LABEL_NAMES, [round(w,3) for w in class_weights.tolist()]))}")
sample_weights = np.array([class_weights[y] for y in train_labels_np],
                          dtype=np.float64)
print(f"✅ [Step 2.5 — class weights + sampler weights recomputed]")

def fresh_model():
    config = AutoConfig.from_pretrained(HF_NAME, num_labels=4)
    config.classifier_dropout = CLASSIFIER_DROPOUT
    m = AutoModelForSequenceClassification.from_pretrained(HF_NAME, config=config)
    m.resize_token_embeddings(len(tokenizer))
    return m

def is_oom(err: BaseException) -> bool:
    s = str(err).lower()
    keywords = ["out of memory", "oom", "allocation", "alloc failed",
                "cuda error: out", "directml", "hipout"]
    return any(k in s for k in keywords)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses, y_true, y_pred = [], [], []
    ce = torch.nn.CrossEntropyLoss()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        losses.append(ce(logits, labels).item())
        y_true.append(labels.cpu().numpy())
        y_pred.append(torch.argmax(logits, dim=-1).cpu().numpy())
    return {
        "loss":     float(np.mean(losses)),
        "macro_f1": f1_score(np.concatenate(y_true), np.concatenate(y_pred), average="macro"),
        "accuracy": accuracy_score(np.concatenate(y_true), np.concatenate(y_pred)),
        "y_true":   np.concatenate(y_true),
        "y_pred":   np.concatenate(y_pred),
    }

def do_full_training(batch_size):
    print(f"\nTraining with batch_size={batch_size}, max_length={MAX_LENGTH}")
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=VAL_TEST_BATCH,
                            shuffle=False, num_workers=0)

    model = fresh_model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps = len(train_loader) * MAX_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    ce_weighted = torch.nn.CrossEntropyLoss(weight=cw_tensor)

    print(f"Steps/epoch={len(train_loader)} total={total_steps} "
          f"warmup={warmup_steps}")

    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_losses = []
        tr_true, tr_pred = [], []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attn).logits
            loss = ce_weighted(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            tr_losses.append(loss.item())
            tr_true.append(labels.detach().cpu().numpy())
            tr_pred.append(torch.argmax(logits, dim=-1).detach().cpu().numpy())

        tr_loss = float(np.mean(tr_losses))
        tr_f1 = f1_score(np.concatenate(tr_true),
                          np.concatenate(tr_pred), average="macro")

        val = evaluate(model, val_loader)
        val_loss = val["loss"]
        val_f1 = val["macro_f1"]

        new_best = val_f1 > best_val_f1
        tag = " [NEW BEST]" if new_best else ""
        print(f"Epoch {epoch:02d} | "
              f"train_loss={tr_loss:.4f} train_macro_f1={tr_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}{tag}")

        if new_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            cpu_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            torch.save(cpu_state_dict,
                       os.path.join(SAVE_DIR, "pytorch_model.bin"))
            model.config.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  >> saved best checkpoint to {SAVE_DIR}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stop: patience {PATIENCE} hit at epoch {epoch}. "
                      f"Best epoch was {best_epoch} (val_macro_f1={best_val_f1:.4f}).")
                break

    return best_val_f1, best_epoch

current_batch = BATCH_SIZE
try:
    best_val_f1, best_epoch = do_full_training(current_batch)
except Exception as e:
    if is_oom(e):
        halved = max(1, current_batch // 2)
        print(f"\n⚠️  OOM at batch_size={current_batch}. Retrying at {halved}.")
        gc.collect()
        current_batch = halved
        try:
            best_val_f1, best_epoch = do_full_training(current_batch)
        except Exception as e2:
            if is_oom(e2):
                print(f"\nERROR: OOM persists at batch_size={current_batch}. Stopping.")
                raise SystemExit(1)
            raise
    else:
        raise
print(f"\nTraining complete. best_epoch={best_epoch} best_val_macro_f1={best_val_f1:.4f}")
print(f"✅ [Step 2.6 — training done]")

print("\nLoading best checkpoint for test evaluation ...")
config = AutoConfig.from_pretrained(HF_NAME, num_labels=4)
config.classifier_dropout = CLASSIFIER_DROPOUT
model = AutoModelForSequenceClassification.from_pretrained(HF_NAME, config=config)
model.resize_token_embeddings(len(tokenizer))
state = torch.load(os.path.join(SAVE_DIR, "pytorch_model.bin"),
                   map_location="cpu")
model.load_state_dict(state)
model.to(device)
model.eval()

test_loader = DataLoader(test_ds, batch_size=VAL_TEST_BATCH,
                         shuffle=False, num_workers=0)
test = evaluate(model, test_loader)
y_true = test["y_true"]
y_pred = test["y_pred"]

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")
per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])

print("\n" + "═" * 64)
print(" STEP 2 — Test-set results (MN-BERT v3, augmented training)")
print("═" * 64)
print(f"test_loss       : {test['loss']:.4f}")
print(f"Accuracy        : {acc:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")
print(f"Weighted F1     : {weighted_f1:.4f}")

print("\nclassification_report:")
print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3],
                            target_names=LABEL_NAMES, digits=4))

print("Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
print("            " + " ".join(f"{n:>13s}" for n in LABEL_NAMES))
for i, row in enumerate(cm):
    cells = " ".join(f"{v:>13d}" for v in row)
    print(f"  {LABEL_NAMES[i]:<10s}{cells}")

def pct(prev, new):
    return f"{(new - prev) * 100:+.2f}%"

new_metrics = {
    "accuracy":     acc,
    "macro_f1":     macro_f1,
    "weighted_f1":  weighted_f1,
    "POSITIVE":     per_class_f1[0],
    "NEUTRAL":      per_class_f1[1],
    "CONSTRUCTIVE": per_class_f1[2],
    "TOXIC":        per_class_f1[3],
}

print("\n── Comparison: v1 baseline | v2 normalized | v3 augmented ──")
print(f"| {'Metric':<15s} | {'v1':>7s} | {'v2':>7s} | {'v3':>7s} | {'Δ v2→v3':>8s} |")
print(f"|{'-'*17}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*10}|")
for key, label in [
    ("accuracy",     "Accuracy"),
    ("macro_f1",     "Macro F1"),
    ("weighted_f1",  "Weighted F1"),
    ("POSITIVE",     "POSITIVE F1"),
    ("NEUTRAL",      "NEUTRAL F1"),
    ("CONSTRUCTIVE", "CONSTRUCTIVE F1"),
    ("TOXIC",        "TOXIC F1"),
]:
    v1 = V1_METRICS[key]
    v2 = V2_METRICS[key]
    v3 = new_metrics[key]
    print(f"| {label:<15s} | {v1:>7.4f} | {v2:>7.4f} | {v3:>7.4f} | {pct(v2, v3):>8s} |")

test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
assert len(test_df) == len(y_pred), f"test row mismatch {len(test_df)} vs {len(y_pred)}"
out = pd.DataFrame({
    "id":              test_df["id"].values,
    "text_light_clean":test_df["text_light_clean"].values,
    "true_label":      [ID2LABEL[int(v)] for v in y_true],
    "predicted_label": [ID2LABEL[int(v)] for v in y_pred],
})
out.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ [Step 2.7 — saved {len(out)} test predictions to {PRED_CSV}]")
print("\n✅ STEP 2 complete.")
