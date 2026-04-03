import os
import sys
import gc
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

HF_NAME    = "tugstugi/bert-base-mongolian-cased"
DATA_CSV   = os.path.join(WORK_DIR, "stage2_dataset.csv")
SAVE_DIR   = os.path.join(WORK_DIR, "best_stage2_model")

STAGE2_MAP   = {"CONSTRUCTIVE": 0, "TOXIC": 1}
STAGE2_ID2L  = {v: k for k, v in STAGE2_MAP.items()}
LABEL_NAMES  = ["CONSTRUCTIVE", "TOXIC"]
NUM_LABELS   = 2

LR                 = 2e-5
WEIGHT_DECAY       = 0.01
WARMUP_RATIO       = 0.06
GRAD_CLIP          = 1.0
CLASSIFIER_DROPOUT = 0.15
MAX_EPOCHS         = 20
PATIENCE           = 5
VAL_TEST_BATCH     = 32
SEED               = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if os.path.isdir(SAVE_DIR) and os.listdir(SAVE_DIR):
    print(f"ERROR: {SAVE_DIR} already exists and non-empty. Stop.")
    raise SystemExit(1)
os.makedirs(SAVE_DIR, exist_ok=True)

try:
    device = torch_directml.device()
    _ = torch.zeros(1, device=device)
    print(f"DirectML device initialized: {device}")
except Exception as e:
    print(f"ERROR: DirectML failed to initialize — {e}")
    raise SystemExit(1)

if not os.path.exists(DATA_CSV):
    print(f"ERROR: {DATA_CSV} not found. Run step1_prepare_2stage.py first.")
    raise SystemExit(1)

df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
required = {"id", "text_normalized", "label_stage2", "split", "source"}
missing = required - set(df.columns)
if missing:
    print(f"ERROR: missing cols in stage2_dataset.csv: {missing}")
    raise SystemExit(1)

df["input_text"] = df.apply(
    lambda r: f"[{r['source']}] {str(r['text_normalized'] or '')}", axis=1
)
df["label_id"] = df["label_stage2"].map(STAGE2_MAP)

print(f"Stage 2 rows: total={len(df)} "
      f"train={int((df.split=='train').sum())} "
      f"val={int((df.split=='val').sum())} "
      f"test={int((df.split=='test').sum())}")
print(f"Hyperparameters: lr={LR}, patience={PATIENCE}, num_labels={NUM_LABELS}")
print("✅ [Step 3.1 — Stage 2 data loaded]")

tokenizer = AutoTokenizer.from_pretrained(HF_NAME)
print("✅ [Step 3.2 — tokenizer loaded]")

print("Computing p95 token length on Stage 2 train split ...")
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
print(f"  token lengths: p50={p50} p95={p95} p99={p99} max={int(tok_lens.max())}")
if p95 <= 256:
    MAX_LENGTH = 256; BATCH_SIZE = 16
else:
    MAX_LENGTH = 512; BATCH_SIZE = 8
print(f"  → chosen max_length={MAX_LENGTH}, batch_size={BATCH_SIZE}")
print("✅ [Step 3.3 — max_length/batch_size decided]")

def encode(texts, labels):
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=MAX_LENGTH, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long)

class DS(Dataset):
    def __init__(self, ids, mask, y):
        self.ids, self.mask, self.y = ids, mask, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.mask[i], "labels": self.y[i]}

print("Tokenizing Stage 2 splits ...")
tr_ids, tr_mask, tr_y = encode(df[df.split=="train"]["input_text"].tolist(), df[df.split=="train"]["label_id"].tolist())
va_ids, va_mask, va_y = encode(df[df.split=="val"]["input_text"].tolist(),   df[df.split=="val"]["label_id"].tolist())
te_ids, te_mask, te_y = encode(df[df.split=="test"]["input_text"].tolist(),  df[df.split=="test"]["label_id"].tolist())
train_ds = DS(tr_ids, tr_mask, tr_y)
val_ds   = DS(va_ids, va_mask, va_y)
test_ds  = DS(te_ids, te_mask, te_y)
print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
print("✅ [Step 3.4 — tokenized]")

train_labels_np = tr_y.numpy()
counts = np.bincount(train_labels_np, minlength=NUM_LABELS).astype(np.float64)
print(f"Stage 2 train label counts: "
      f"{dict(zip(LABEL_NAMES, counts.astype(int).tolist()))}")
total = counts.sum()
class_weights = (total / (NUM_LABELS * counts))
print(f"Class weights (CE): "
      f"{dict(zip(LABEL_NAMES, [round(w,3) for w in class_weights.tolist()]))}")
sample_weights = np.array([class_weights[y] for y in train_labels_np], dtype=np.float64)
print("✅ [Step 3.5 — class weights computed]")

def fresh_model():
    config = AutoConfig.from_pretrained(HF_NAME, num_labels=NUM_LABELS)
    config.classifier_dropout = CLASSIFIER_DROPOUT
    m = AutoModelForSequenceClassification.from_pretrained(HF_NAME, config=config)
    m.resize_token_embeddings(len(tokenizer))
    return m

def is_oom(err: BaseException) -> bool:
    s = str(err).lower()
    return any(k in s for k in [
        "out of memory","oom","allocation","alloc failed",
        "cuda error: out","directml","hipout"
    ])

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
    yt = np.concatenate(y_true); yp = np.concatenate(y_pred)
    return {"loss": float(np.mean(losses)),
            "macro_f1": f1_score(yt, yp, average="macro"),
            "accuracy": accuracy_score(yt, yp),
            "y_true": yt, "y_pred": yp}

def do_full_training(batch_size):
    print(f"\nTraining Stage 2 with batch_size={batch_size}, "
          f"max_length={MAX_LENGTH}, lr={LR}, patience={PATIENCE}")
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=VAL_TEST_BATCH, shuffle=False, num_workers=0)

    model = fresh_model().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * MAX_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    ce_weighted = torch.nn.CrossEntropyLoss(weight=cw_tensor)

    print(f"Steps/epoch={len(train_loader)} total={total_steps} warmup={warmup_steps}")

    best_val_f1 = -1.0; best_epoch = -1; patience_counter = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_losses = []; tr_true, tr_pred = [], []
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
        tr_f1 = f1_score(np.concatenate(tr_true), np.concatenate(tr_pred), average="macro")

        val = evaluate(model, val_loader)
        val_loss = val["loss"]; val_f1 = val["macro_f1"]

        new_best = val_f1 > best_val_f1
        tag = " [NEW BEST]" if new_best else ""
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} train_macro_f1={tr_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}{tag}")

        if new_best:
            best_val_f1 = val_f1; best_epoch = epoch; patience_counter = 0
            cpu_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(cpu_state_dict, os.path.join(SAVE_DIR, "pytorch_model.bin"))
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
                print(f"\nERROR: OOM persists at batch_size={current_batch}.")
                raise SystemExit(1)
            raise
    else:
        raise
print(f"\nStage 2 training complete. best_epoch={best_epoch} "
      f"best_val_macro_f1={best_val_f1:.4f}")
print("✅ [Step 3.6 — Stage 2 training done]")

print("\nLoading best Stage 2 checkpoint for test evaluation ...")
config = AutoConfig.from_pretrained(HF_NAME, num_labels=NUM_LABELS)
config.classifier_dropout = CLASSIFIER_DROPOUT
model = AutoModelForSequenceClassification.from_pretrained(HF_NAME, config=config)
model.resize_token_embeddings(len(tokenizer))
state = torch.load(os.path.join(SAVE_DIR, "pytorch_model.bin"), map_location="cpu")
model.load_state_dict(state)
model.to(device); model.eval()

test_loader = DataLoader(test_ds, batch_size=VAL_TEST_BATCH, shuffle=False, num_workers=0)
test = evaluate(model, test_loader)
y_true = test["y_true"]; y_pred = test["y_pred"]

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")

print("\n" + "═"*64)
print(" STEP 3 — Stage 2 test-set results (CONSTRUCTIVE vs TOXIC, subset only)")
print("═"*64)
print(f"test_loss       : {test['loss']:.4f}")
print(f"Accuracy        : {acc:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")
print(f"Weighted F1     : {weighted_f1:.4f}")

print("\nclassification_report:")
print(classification_report(y_true, y_pred, labels=list(range(NUM_LABELS)),
                            target_names=LABEL_NAMES, digits=4))

print("Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))
print("                " + " ".join(f"{n:>13s}" for n in LABEL_NAMES))
for i, row in enumerate(cm):
    cells = " ".join(f"{v:>13d}" for v in row)
    print(f"  {LABEL_NAMES[i]:<13s} {cells}")

print("\n✅ STEP 3 (Stage 2) complete. "
      "STOP — waiting for confirmation to run STEP 4 (full pipeline + A vs B compare).")
