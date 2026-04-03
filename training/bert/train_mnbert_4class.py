import os
import json
import pandas as pd
import numpy as np
import torch
import torch_directml
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

try:
    device = torch_directml.device()
    _ = torch.zeros(1, device=device)
    print(f"DirectML device initialized: {device}")
except Exception as e:
    print(f"ERROR: DirectML failed to initialize — {e}")
    print("Stopping. Do NOT fall back to CPU.")
    raise SystemExit(1)

CSV_PATH = os.path.join(WORK_DIR, "10k_fully_labeled.csv")
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

required_cols = {"id", "text_light_clean", "label", "split"}
missing = required_cols - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

assert set(df["split"].unique()) == {"train", "val", "test"}, \
    f"Unexpected split values: {df['split'].unique()}"

valid_labels = {"POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"}
invalid_mask = ~df["label"].isin(valid_labels)
n_invalid = invalid_mask.sum()
if n_invalid > 0:
    print(f"WARNING: Dropping {n_invalid} rows with invalid labels (annotation artifacts)")
    df = df[~invalid_mask].reset_index(drop=True)

assert set(df["label"].unique()) == valid_labels, \
    f"Unexpected label values after filtering: {df['label'].unique()}"

print(f"Dataset: {len(df)} rows  |  splits: {df['split'].value_counts().to_dict()}")
print("✅ Step 1 — CSV loaded, all columns verified")

LABEL_MAP = {"POSITIVE": 0, "NEUTRAL": 1, "CONSTRUCTIVE": 2, "TOXIC": 3}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

with open(os.path.join(WORK_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(LABEL_MAP, f, ensure_ascii=False)

df["label_id"] = df["label"].map(LABEL_MAP)
print("✅ Step 2 — label_map.json saved  (POSITIVE=0, NEUTRAL=1, CONSTRUCTIVE=2, TOXIC=3)")

MODEL_NAME = "tugstugi/bert-base-mongolian-cased"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"ERROR: Failed to load tokenizer from '{MODEL_NAME}': {e}")
    print("Stopping — do NOT substitute a different model.")
    raise SystemExit(1)

texts = df.apply(lambda r: f"[{r['source']}] {str(r['text_light_clean'] or '')}", axis=1).tolist()
MAX_LEN = 256
encodings = tokenizer(texts, max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt")
print(f"✅ Step 3 — Tokenized {len(texts)} texts  (max_length={MAX_LEN}, source prepended)")

class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels, indices):
        self.input_ids = input_ids[indices]
        self.attention_mask = attention_mask[indices]
        self.labels = labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

all_labels = torch.tensor(df["label_id"].values, dtype=torch.long)

train_idx = torch.tensor(df.index[df["split"] == "train"].tolist())
val_idx   = torch.tensor(df.index[df["split"] == "val"].tolist())
test_idx  = torch.tensor(df.index[df["split"] == "test"].tolist())

train_ds = TextDataset(encodings["input_ids"], encodings["attention_mask"], all_labels, train_idx)
val_ds   = TextDataset(encodings["input_ids"], encodings["attention_mask"], all_labels, val_idx)
test_ds  = TextDataset(encodings["input_ids"], encodings["attention_mask"], all_labels, test_idx)

train_label_ids = df.loc[df["split"] == "train", "label_id"].values
train_class_counts = np.bincount(train_label_ids, minlength=4).astype(float)
sample_weights = torch.tensor([1.0 / train_class_counts[l] for l in train_label_ids], dtype=torch.float64)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Step 4 — DataLoaders ready  (train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, batch={BATCH_SIZE})")
print(f"  Balanced sampling ON — per-class weight: { {ID2LABEL[i]: round(1.0/train_class_counts[i], 6) for i in range(4)} }")

train_labels = df.loc[df["split"] == "train", "label_id"].values
class_counts = np.bincount(train_labels, minlength=4).astype(float)
class_weights = (1.0 / class_counts) * class_counts.sum() / len(class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print(f"Class counts (train):  { {ID2LABEL[i]: int(class_counts[i]) for i in range(4)} }")
print(f"Class weights:         { {ID2LABEL[i]: round(class_weights[i], 4) for i in range(4)} }")
print("✅ Step 5 — Class-weighted CrossEntropyLoss ready (weights on DirectML device)")

try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4,
        hidden_dropout_prob=0.15,
        attention_probs_dropout_prob=0.15,
    )
except Exception as e:
    print(f"ERROR: Failed to load model from '{MODEL_NAME}': {e}")
    print("Stopping — do NOT substitute a different model.")
    raise SystemExit(1)

model.resize_token_embeddings(len(tokenizer))
print(f"  Dropout: hidden=0.15, attention=0.15")
model.to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

MAX_EPOCHS = 15
PATIENCE = 4
WARMUP_RATIO = 0.1
total_steps = len(train_loader) * MAX_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

SAVE_DIR = os.path.join(WORK_DIR, "best_mnbert_model")

best_val_f1 = 0.0
patience_counter = 0

print(f"\nTraining config: epochs={MAX_EPOCHS}, lr=2e-5, warmup_steps={warmup_steps}, total_steps={total_steps}")
print(f"Early stopping: patience={PATIENCE} on val macro F1\n")

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    train_preds, train_true = [], []

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        train_preds.extend(preds)
        train_true.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch} | batch {batch_idx+1}/{len(train_loader)} | loss={loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    train_f1 = f1_score(train_true, train_preds, average="macro")

    model.eval()
    val_preds, val_true = [], []
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()

            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(val_true, val_preds, average="macro")
    val_acc = accuracy_score(val_true, val_preds)

    print(f"Epoch {epoch}/{MAX_EPOCHS}  |  train_loss={avg_loss:.4f}  train_macroF1={train_f1:.4f}"
          f"  |  val_loss={avg_val_loss:.4f}  val_macroF1={val_f1:.4f}  val_acc={val_acc:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        os.makedirs(SAVE_DIR, exist_ok=True)
        cpu_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(cpu_state_dict, os.path.join(SAVE_DIR, "pytorch_model.bin"))
        model.config.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"  >> New best val macro F1: {val_f1:.4f} — checkpoint saved to {SAVE_DIR}")
    else:
        patience_counter += 1
        print(f"  >> No improvement ({patience_counter}/{PATIENCE})")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE} exhausted)")
        break

print(f"\n✅ Step 6 — Training complete. Best val macro F1: {best_val_f1:.4f}")

print("\nLoading best checkpoint for test evaluation...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
model.resize_token_embeddings(len(tokenizer))
state_dict = torch.load(os.path.join(SAVE_DIR, "pytorch_model.bin"), map_location="cpu")
model.load_state_dict(state_dict)
model.to(device)
model.eval()

test_preds, test_true = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        test_preds.extend(preds)
        test_true.extend(labels.numpy())

test_preds = np.array(test_preds)
test_true = np.array(test_true)
print("✅ Step 7 — Test inference complete")

label_names = ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]

macro_f1   = f1_score(test_true, test_preds, average="macro")
weighted_f1 = f1_score(test_true, test_preds, average="weighted")
accuracy   = accuracy_score(test_true, test_preds)

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Accuracy:    {accuracy:.4f}")
print("\n--- Classification Report ---")
print(classification_report(test_true, test_preds, target_names=label_names, digits=4))

print("--- Confusion Matrix ---")
cm = confusion_matrix(test_true, test_preds)
header = "                " + "  ".join(f"{n:>13s}" for n in label_names)
print(f"{'Predicted →':>16s}")
print(header)
for i, row in enumerate(cm):
    row_str = "  ".join(f"{v:>13d}" for v in row)
    print(f"{label_names[i]:>16s}  {row_str}")

test_df = df.loc[df["split"] == "test"].copy()
test_df = test_df.reset_index(drop=True)
test_df["predicted_label"] = [ID2LABEL[p] for p in test_preds]
test_df["true_label"] = test_df["label"]

out_cols = ["id", "text_light_clean", "true_label", "predicted_label"]
test_df[out_cols].to_csv(os.path.join(WORK_DIR, "test_predictions.csv"), index=False, encoding="utf-8-sig")

print(f"\n✅ Step 8 — Evaluation complete. Files saved:")
print(f"   ./best_mnbert_model/      (best checkpoint)")
print(f"   ./label_map.json          (label mapping)")
print(f"   ./test_predictions.csv    (test predictions)")
print("\nDone.")
