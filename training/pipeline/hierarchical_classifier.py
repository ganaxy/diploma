import os
import sys
import time
import copy
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch_directml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed as hf_set_seed,
)
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

LEARNING_RATE = 2e-5
EPOCHS = 50
BATCH_SIZE = 16
MAX_LENGTH = 128
SEED = 42

DATA_PATH = "sampled_fully_labeled_new_version_v1_stage2.csv"
STAGE1_OUTPUT_DIR = "./stage1_model"
STAGE2_OUTPUT_DIR = "./stage2_model"
CONFUSION_MATRIX_PATH = "./confusion_matrix_hierarchical.png"

STAGE1_LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
STAGE2_LABELS = ["TOXIC", "CONSTRUCTIVE"]
FINAL_4CLASS_LABELS = ["POSITIVE", "NEUTRAL", "TOXIC", "CONSTRUCTIVE"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hf_set_seed(seed)

set_seed(SEED)

try:
    DEVICE = torch_directml.device()
    GPU_NAME = torch_directml.device_name(0)
    print(f"[INFO] Using AMD GPU via DirectML: {GPU_NAME}")
except Exception:
    DEVICE = torch.device("cpu")
    print("[INFO] DirectML not available — using CPU.")

print("\n" + "=" * 60)
print("DATA PREPARATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded {len(df)} rows from {DATA_PATH}")
print(f"[INFO] Columns: {list(df.columns)}")

print("\n[INFO] Value counts for 'split' column:")
print(df["split"].value_counts())

df["split_norm"] = df["split"].str.strip().str.lower()
df["split_norm"] = df["split_norm"].replace({"val": "validation"})

train_df = df[df["split_norm"] == "train"].copy()
val_df = df[df["split_norm"] == "validation"].copy()
test_df = df[df["split_norm"] == "test"].copy()

print(f"\n[INFO] Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

stage1_label2id = {lab: i for i, lab in enumerate(STAGE1_LABELS)}
stage1_id2label = {i: lab for lab, i in stage1_label2id.items()}

for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"  Stage1 {split_name} distribution: {split_df['label_stage1'].value_counts().to_dict()}")

stage2_label2id = {lab: i for i, lab in enumerate(STAGE2_LABELS)}
stage2_id2label = {i: lab for lab, i in stage2_label2id.items()}

train_s2 = train_df[train_df["label_stage2"].notna() & (train_df["label_stage2"] != "")].copy()
val_s2 = val_df[val_df["label_stage2"].notna() & (val_df["label_stage2"] != "")].copy()
test_s2 = test_df[test_df["label_stage2"].notna() & (test_df["label_stage2"] != "")].copy()

print(f"\n[INFO] Stage 2 subsets — Train: {len(train_s2)}  |  Val: {len(val_s2)}  |  Test: {len(test_s2)}")
for split_name, split_df in [("train", train_s2), ("val", val_s2), ("test", test_s2)]:
    print(f"  Stage2 {split_name} distribution: {split_df['label_stage2'].value_counts().to_dict()}")

print("\n[INFO] Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"[INFO] Tokenizer loaded: {MODEL_NAME}")
except Exception as e:
    raise RuntimeError(
        f"[ERROR] Cannot download tokenizer for '{MODEL_NAME}'. "
        f"Check your internet connection or model name. Details: {e}"
    )

class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def build_datasets(train_df, val_df, test_df, text_col, label_col, label2id, tokenizer, max_length):
    def encode(split_df):
        texts = split_df[text_col].astype(str).tolist()
        labels = [label2id[lab] for lab in split_df[label_col]]
        return TextClassificationDataset(texts, labels, tokenizer, max_length)
    return encode(train_df), encode(val_df), encode(test_df)

print("\n[INFO] Tokenising Stage 1 datasets...")
s1_train_ds, s1_val_ds, s1_test_ds = build_datasets(
    train_df, val_df, test_df,
    text_col="text_ml_clean", label_col="label_stage1",
    label2id=stage1_label2id, tokenizer=tokenizer, max_length=MAX_LENGTH,
)

print("[INFO] Tokenising Stage 2 datasets...")
s2_train_ds, s2_val_ds, s2_test_ds = build_datasets(
    train_s2, val_s2, test_s2,
    text_col="text_ml_clean", label_col="label_stage2",
    label2id=stage2_label2id, tokenizer=tokenizer, max_length=MAX_LENGTH,
)

def compute_weights(labels, label_order):
    weights = compute_class_weight("balanced", classes=np.array(label_order), y=labels)
    return torch.tensor(weights, dtype=torch.float32)

s1_weights = compute_weights(train_df["label_stage1"].tolist(), STAGE1_LABELS)
print(f"\n[INFO] Stage 1 class weights: {dict(zip(STAGE1_LABELS, s1_weights.tolist()))}")

s2_weights = compute_weights(train_s2["label_stage2"].tolist(), STAGE2_LABELS)
print(f"[INFO] Stage 2 class weights: {dict(zip(STAGE2_LABELS, s2_weights.tolist()))}")

def evaluate_model(model, dataloader, device, class_weights=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = total_loss / max(n_batches, 1)
    acc = accuracy_score(all_labels, all_preds)
    macro_p = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_r = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }
    return metrics, all_preds, all_labels

def train_model(model, train_ds, val_ds, device, class_weights, label_list, stage_name,
                epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model.to(device)
    weights_on_device = class_weights.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights_on_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )

    best_macro_f1 = -1.0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        step_count = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step_count += 1

            if (batch_idx + 1) % 20 == 0:
                avg = running_loss / step_count
                print(f"  [{stage_name}] Epoch {epoch}/{epochs} | Step {batch_idx+1}/{len(train_loader)} | Train Loss: {avg:.4f}")

        epoch_time = time.time() - epoch_start
        train_avg_loss = running_loss / max(step_count, 1)

        val_metrics, _, _ = evaluate_model(model, val_loader, device, class_weights)
        print(
            f"  [{stage_name}] Epoch {epoch}/{epochs} done in {epoch_time:.0f}s | "
            f"Train Loss: {train_avg_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro P: {val_metrics['macro_precision']:.4f} | "
            f"Val Macro R: {val_metrics['macro_recall']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  [{stage_name}] >>> New best model! Macro F1 = {best_macro_f1:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[INFO] {stage_name}: Restored best model (Macro F1 = {best_macro_f1:.4f})")

    return model

print("\n" + "=" * 60)
print("STAGE 1 TRAINING — 3-class (POSITIVE / NEUTRAL / NEGATIVE)")
print("=" * 60)

try:
    s1_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(STAGE1_LABELS),
        id2label=stage1_id2label,
        label2id=stage1_label2id,
    )
    s1_model.resize_token_embeddings(len(tokenizer))
except Exception as e:
    raise RuntimeError(
        f"[ERROR] Cannot download model '{MODEL_NAME}'. "
        f"Check your internet connection or model name. Details: {e}"
    )

print("[INFO] Starting Stage 1 training...")
s1_model = train_model(
    s1_model, s1_train_ds, s1_val_ds, DEVICE,
    s1_weights, STAGE1_LABELS, stage_name="Stage1",
)
print("[INFO] Stage 1 training complete.")

print("\n" + "=" * 60)
print("STAGE 2 TRAINING — 2-class (TOXIC / CONSTRUCTIVE)")
print("=" * 60)

try:
    s2_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(STAGE2_LABELS),
        id2label=stage2_id2label,
        label2id=stage2_label2id,
    )
    s2_model.resize_token_embeddings(len(tokenizer))
except Exception as e:
    raise RuntimeError(
        f"[ERROR] Cannot download model '{MODEL_NAME}'. Details: {e}"
    )

print("[INFO] Starting Stage 2 training...")
s2_model = train_model(
    s2_model, s2_train_ds, s2_val_ds, DEVICE,
    s2_weights, STAGE2_LABELS, stage_name="Stage2",
)
print("[INFO] Stage 2 training complete.")

print("\n" + "=" * 60)
print("2-STAGE INFERENCE PIPELINE")
print("=" * 60)

def hierarchical_predict(texts, s1_model, s2_model, tokenizer, max_length, device):
    s1_model.eval()
    s2_model.eval()

    enc = tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=max_length, return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        s1_logits = s1_model(**enc).logits
    s1_preds = torch.argmax(s1_logits, dim=-1).cpu().numpy()

    final_labels = [None] * len(s1_preds)
    neg_indices = []

    for i, pred_id in enumerate(s1_preds):
        s1_label = stage1_id2label[pred_id]
        if s1_label in ("POSITIVE", "NEUTRAL"):
            final_labels[i] = s1_label
        else:  # NEGATIVE → route to Stage 2 (TOXIC vs CONSTRUCTIVE)
            neg_indices.append(i)

    if neg_indices:
        neg_texts = [texts[i] for i in neg_indices]
        enc2 = tokenizer(
            neg_texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        enc2 = {k: v.to(device) for k, v in enc2.items()}

        with torch.no_grad():
            s2_logits = s2_model(**enc2).logits
        s2_preds = torch.argmax(s2_logits, dim=-1).cpu().numpy()

        for j, idx in enumerate(neg_indices):
            final_labels[idx] = stage2_id2label[s2_preds[j]]

    return final_labels

def batched_hierarchical_predict(texts, s1_model, s2_model, tokenizer, max_length, device, batch_size=64):
    all_preds = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        preds = hierarchical_predict(batch_texts, s1_model, s2_model, tokenizer, max_length, device)
        all_preds.extend(preds)
        if (start // batch_size) % 5 == 0:
            print(f"  [inference] processed {min(start + batch_size, len(texts))}/{len(texts)}")
    return all_preds

print("\n" + "=" * 60)
print("STAGE 1 STANDALONE EVALUATION (3-class)")
print("=" * 60)

s1_test_loader = DataLoader(s1_test_ds, batch_size=BATCH_SIZE, shuffle=False)
s1_test_metrics, s1_test_preds, s1_test_true = evaluate_model(s1_model, s1_test_loader, DEVICE, s1_weights)

print(classification_report(
    s1_test_true, s1_test_preds,
    target_names=STAGE1_LABELS, digits=4, zero_division=0,
))

print("\n" + "=" * 60)
print("STAGE 2 STANDALONE EVALUATION (2-class)")
print("=" * 60)

s2_test_loader = DataLoader(s2_test_ds, batch_size=BATCH_SIZE, shuffle=False)
s2_test_metrics, s2_test_preds, s2_test_true = evaluate_model(s2_model, s2_test_loader, DEVICE, s2_weights)

print(classification_report(
    s2_test_true, s2_test_preds,
    target_names=STAGE2_LABELS, digits=4, zero_division=0,
))

print("\n" + "=" * 60)
print("FULL HIERARCHICAL EVALUATION (4-class)")
print("=" * 60)

test_texts = test_df["text_ml_clean"].astype(str).tolist()
test_true_4class = test_df["label4_class"].tolist()

s1_model.to(DEVICE)
s2_model.to(DEVICE)

print("[INFO] Running 2-stage inference on test set...")
test_pred_4class = batched_hierarchical_predict(
    test_texts, s1_model, s2_model, tokenizer, MAX_LENGTH, DEVICE, batch_size=BATCH_SIZE,
)

print("\n[RESULTS] 4-class classification report:")
print(classification_report(
    test_true_4class, test_pred_4class,
    labels=FINAL_4CLASS_LABELS, target_names=FINAL_4CLASS_LABELS,
    digits=4, zero_division=0,
))

overall_acc = accuracy_score(test_true_4class, test_pred_4class)
macro_f1 = f1_score(test_true_4class, test_pred_4class, labels=FINAL_4CLASS_LABELS, average="macro", zero_division=0)
weighted_f1 = f1_score(test_true_4class, test_pred_4class, labels=FINAL_4CLASS_LABELS, average="weighted", zero_division=0)

print(f"Overall Accuracy : {overall_acc:.4f}")
print(f"Macro F1         : {macro_f1:.4f}")
print(f"Weighted F1      : {weighted_f1:.4f}")

cm = confusion_matrix(test_true_4class, test_pred_4class, labels=FINAL_4CLASS_LABELS)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=FINAL_4CLASS_LABELS, yticklabels=FINAL_4CLASS_LABELS,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Hierarchical 2-Stage — 4-Class Confusion Matrix")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
plt.close()
print(f"\n[INFO] Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

print("\n" + "=" * 60)
print("MODEL SAVING")
print("=" * 60)

s1_model.to("cpu")
s2_model.to("cpu")

os.makedirs(STAGE1_OUTPUT_DIR, exist_ok=True)
os.makedirs(STAGE2_OUTPUT_DIR, exist_ok=True)

s1_model.save_pretrained(STAGE1_OUTPUT_DIR)
tokenizer.save_pretrained(STAGE1_OUTPUT_DIR)
print(f"[INFO] Stage 1 model + tokenizer saved to {STAGE1_OUTPUT_DIR}/")

s2_model.save_pretrained(STAGE2_OUTPUT_DIR)
tokenizer.save_pretrained(STAGE2_OUTPUT_DIR)
print(f"[INFO] Stage 2 model + tokenizer saved to {STAGE2_OUTPUT_DIR}/")

print("\n" + "=" * 60)
print("ALL DONE — Hierarchical 2-stage pipeline complete.")
print("=" * 60)
