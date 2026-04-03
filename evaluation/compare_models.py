import gc
import os
import time
import copy
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch_directml
import matplotlib
matplotlib.use("Agg")
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

MODEL_NAME = "tugstugi/bert-base-mongolian-cased"
LEARNING_RATE = 2e-5
EPOCHS = 15
BATCH_SIZE = 8
MAX_LENGTH = 128
SEED = 42

FLAT_DATA_PATH = "sampled_fully_labeled_new_version_v1.csv"
HIER_DATA_PATH = "sampled_fully_labeled_new_version_v1_stage2.csv"

OUTPUT_DIR = "./comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FLAT_4CLASS_LABELS = ["POSITIVE", "NEUTRAL", "TOXIC", "CONSTRUCTIVE"]
STAGE1_LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
STAGE2_LABELS = ["TOXIC", "CONSTRUCTIVE"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hf_set_seed(seed)

try:
    DEVICE = torch_directml.device()
    GPU_NAME = torch_directml.device_name(0)
    print(f"[INFO] Using AMD GPU via DirectML: {GPU_NAME}")
except Exception:
    DEVICE = torch.device("cpu")
    print("[INFO] DirectML not available — using CPU.")

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
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

def compute_weights(labels, label_order):
    weights = compute_class_weight("balanced", classes=np.array(label_order), y=labels)
    return torch.tensor(weights, dtype=torch.float32)

def evaluate_model(model, dataloader, device, class_weights=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
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
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"loss": avg_loss, "accuracy": acc, "macro_f1": macro_f1}, all_preds, all_labels

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
    epoch_log = []

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

        epoch_time = time.time() - epoch_start
        train_avg_loss = running_loss / max(step_count, 1)

        val_metrics, _, _ = evaluate_model(model, val_loader, device, class_weights)
        status = ""
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            if best_model_state is not None:
                del best_model_state
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            gc.collect()
            status = "* BEST *"

        epoch_log.append({
            "epoch": epoch, "train_loss": train_avg_loss,
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"], "time_sec": epoch_time,
        })

        print(
            f"  [{stage_name}] Epoch {epoch:>3}/{epochs} | "
            f"Train Loss: {train_avg_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f} | "
            f"{epoch_time:.0f}s {status}"
        )

    if best_model_state is not None:

        device_state = {k: v.to(device) for k, v in best_model_state.items()}
        model.load_state_dict(device_state)
        del device_state
        print(f"  [{stage_name}] Restored best model (Macro F1 = {best_macro_f1:.4f})")

    return model, epoch_log

def hierarchical_predict(texts, s1_model, s2_model, tokenizer, max_length, device,
                         stage1_id2label, stage2_id2label):
    s1_model.eval()
    s2_model.eval()
    enc = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        s1_logits = s1_model(**enc).logits
    s1_preds = torch.argmax(s1_logits, dim=-1).cpu().numpy()

    final_labels = []
    neg_indices = []
    for i, pred_id in enumerate(s1_preds):
        s1_label = stage1_id2label[pred_id]
        if s1_label in ("POSITIVE", "NEUTRAL"):
            final_labels.append(s1_label)
        else:
            final_labels.append(None)
            neg_indices.append(i)

    if neg_indices:
        neg_texts = [texts[i] for i in neg_indices]
        enc2 = tokenizer(neg_texts, truncation=True, padding="max_length",
                         max_length=max_length, return_tensors="pt")
        enc2 = {k: v.to(device) for k, v in enc2.items()}
        with torch.no_grad():
            s2_logits = s2_model(**enc2).logits
        s2_preds = torch.argmax(s2_logits, dim=-1).cpu().numpy()
        for j, idx in enumerate(neg_indices):
            final_labels[idx] = stage2_id2label[s2_preds[j]]

    return final_labels

def batched_hierarchical_predict(texts, s1_model, s2_model, tokenizer, max_length, device,
                                 stage1_id2label, stage2_id2label, batch_size=64):
    all_preds = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        preds = hierarchical_predict(batch_texts, s1_model, s2_model, tokenizer,
                                     max_length, device, stage1_id2label, stage2_id2label)
        all_preds.extend(preds)
    return all_preds

def main():
    set_seed(SEED)

    print("\n[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\n" + "=" * 70)
    print("  PART A: FLAT 4-CLASS MN-BERT MODEL")
    print("=" * 70)

    set_seed(SEED)

    flat_df = pd.read_csv(FLAT_DATA_PATH)
    print(f"[INFO] Loaded {len(flat_df)} rows from {FLAT_DATA_PATH}")

    flat_df["split_norm"] = flat_df["split"].str.strip().str.lower().replace({"val": "validation"})
    flat_train = flat_df[flat_df["split_norm"] == "train"].copy()
    flat_val = flat_df[flat_df["split_norm"] == "validation"].copy()
    flat_test = flat_df[flat_df["split_norm"] == "test"].copy()
    print(f"[INFO] Flat splits — Train: {len(flat_train)} | Val: {len(flat_val)} | Test: {len(flat_test)}")

    flat_label2id = {lab: i for i, lab in enumerate(FLAT_4CLASS_LABELS)}
    flat_id2label = {i: lab for lab, i in flat_label2id.items()}

    print("[INFO] Label distribution:")
    for split_name, sdf in [("train", flat_train), ("val", flat_val), ("test", flat_test)]:
        print(f"  {split_name}: {sdf['label'].value_counts().to_dict()}")

    print("[INFO] Tokenising flat datasets...")
    flat_train_ds, flat_val_ds, flat_test_ds = build_datasets(
        flat_train, flat_val, flat_test,
        text_col="text_light_clean", label_col="label",
        label2id=flat_label2id, tokenizer=tokenizer, max_length=MAX_LENGTH,
    )

    flat_weights = compute_weights(flat_train["label"].tolist(), FLAT_4CLASS_LABELS)
    print(f"[INFO] Flat class weights: {dict(zip(FLAT_4CLASS_LABELS, flat_weights.tolist()))}")

    flat_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(FLAT_4CLASS_LABELS),
        id2label=flat_id2label, label2id=flat_label2id,
    )
    flat_model.resize_token_embeddings(len(tokenizer))

    print("[INFO] Starting FLAT model training...")
    flat_model, flat_epoch_log = train_model(
        flat_model, flat_train_ds, flat_val_ds, DEVICE,
        flat_weights, FLAT_4CLASS_LABELS, stage_name="FLAT-4class",
    )
    print("[INFO] FLAT model training complete.\n")

    flat_test_loader = DataLoader(flat_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    flat_test_metrics, flat_test_preds, flat_test_true = evaluate_model(
        flat_model, flat_test_loader, DEVICE, flat_weights
    )
    flat_test_pred_labels = [flat_id2label[p] for p in flat_test_preds]
    flat_test_true_labels = [flat_id2label[t] for t in flat_test_true]

    print("\n" + "=" * 70)
    print("  FLAT MODEL — TEST RESULTS (4-class)")
    print("=" * 70)
    print(classification_report(
        flat_test_true_labels, flat_test_pred_labels,
        labels=FLAT_4CLASS_LABELS, target_names=FLAT_4CLASS_LABELS,
        digits=4, zero_division=0,
    ))

    flat_acc = accuracy_score(flat_test_true_labels, flat_test_pred_labels)
    flat_macro_f1 = f1_score(flat_test_true_labels, flat_test_pred_labels,
                             labels=FLAT_4CLASS_LABELS, average="macro", zero_division=0)
    flat_weighted_f1 = f1_score(flat_test_true_labels, flat_test_pred_labels,
                                labels=FLAT_4CLASS_LABELS, average="weighted", zero_division=0)
    flat_macro_p = precision_score(flat_test_true_labels, flat_test_pred_labels,
                                   labels=FLAT_4CLASS_LABELS, average="macro", zero_division=0)
    flat_macro_r = recall_score(flat_test_true_labels, flat_test_pred_labels,
                                labels=FLAT_4CLASS_LABELS, average="macro", zero_division=0)

    del flat_model, flat_test_loader
    gc.collect()

    print("\n" + "=" * 70)
    print("  PART B: TWO-STAGE HIERARCHICAL MN-BERT MODEL")
    print("=" * 70)

    set_seed(SEED)

    hier_df = pd.read_csv(HIER_DATA_PATH)
    print(f"[INFO] Loaded {len(hier_df)} rows from {HIER_DATA_PATH}")

    hier_df["split_norm"] = hier_df["split"].str.strip().str.lower().replace({"val": "validation"})
    hier_train = hier_df[hier_df["split_norm"] == "train"].copy()
    hier_val = hier_df[hier_df["split_norm"] == "validation"].copy()
    hier_test = hier_df[hier_df["split_norm"] == "test"].copy()
    print(f"[INFO] Hier splits — Train: {len(hier_train)} | Val: {len(hier_val)} | Test: {len(hier_test)}")

    stage1_label2id = {lab: i for i, lab in enumerate(STAGE1_LABELS)}
    stage1_id2label = {i: lab for lab, i in stage1_label2id.items()}

    stage2_label2id = {lab: i for i, lab in enumerate(STAGE2_LABELS)}
    stage2_id2label = {i: lab for lab, i in stage2_label2id.items()}

    train_s2 = hier_train[hier_train["label_stage2"].notna() & (hier_train["label_stage2"] != "")].copy()
    val_s2 = hier_val[hier_val["label_stage2"].notna() & (hier_val["label_stage2"] != "")].copy()
    test_s2 = hier_test[hier_test["label_stage2"].notna() & (hier_test["label_stage2"] != "")].copy()

    print(f"[INFO] Stage 1 — all rows: Train={len(hier_train)}, Val={len(hier_val)}, Test={len(hier_test)}")
    print(f"[INFO] Stage 2 — NEGATIVE rows: Train={len(train_s2)}, Val={len(val_s2)}, Test={len(test_s2)}")

    for split_name, sdf in [("train", hier_train), ("val", hier_val), ("test", hier_test)]:
        print(f"  Stage1 {split_name}: {sdf['label_stage1'].value_counts().to_dict()}")
    for split_name, sdf in [("train", train_s2), ("val", val_s2), ("test", test_s2)]:
        print(f"  Stage2 {split_name}: {sdf['label_stage2'].value_counts().to_dict()}")

    print("[INFO] Tokenising Stage 1 datasets...")
    s1_train_ds, s1_val_ds, s1_test_ds = build_datasets(
        hier_train, hier_val, hier_test,
        text_col="text_light_clean", label_col="label_stage1",
        label2id=stage1_label2id, tokenizer=tokenizer, max_length=MAX_LENGTH,
    )

    print("[INFO] Tokenising Stage 2 datasets...")
    s2_train_ds, s2_val_ds, s2_test_ds = build_datasets(
        train_s2, val_s2, test_s2,
        text_col="text_light_clean", label_col="label_stage2",
        label2id=stage2_label2id, tokenizer=tokenizer, max_length=MAX_LENGTH,
    )

    s1_weights = compute_weights(hier_train["label_stage1"].tolist(), STAGE1_LABELS)
    print(f"[INFO] Stage 1 class weights: {dict(zip(STAGE1_LABELS, s1_weights.tolist()))}")

    s1_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(STAGE1_LABELS),
        id2label=stage1_id2label, label2id=stage1_label2id,
    )
    s1_model.resize_token_embeddings(len(tokenizer))

    print("[INFO] Starting Stage 1 training...")
    s1_model, s1_epoch_log = train_model(
        s1_model, s1_train_ds, s1_val_ds, DEVICE,
        s1_weights, STAGE1_LABELS, stage_name="Stage1-3class",
    )
    print("[INFO] Stage 1 training complete.\n")

    gc.collect()

    s2_weights = compute_weights(train_s2["label_stage2"].tolist(), STAGE2_LABELS)
    print(f"[INFO] Stage 2 class weights: {dict(zip(STAGE2_LABELS, s2_weights.tolist()))}")

    s2_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(STAGE2_LABELS),
        id2label=stage2_id2label, label2id=stage2_label2id,
    )
    s2_model.resize_token_embeddings(len(tokenizer))

    print("[INFO] Starting Stage 2 training...")
    s2_model, s2_epoch_log = train_model(
        s2_model, s2_train_ds, s2_val_ds, DEVICE,
        s2_weights, STAGE2_LABELS, stage_name="Stage2-binary",
    )
    print("[INFO] Stage 2 training complete.\n")

    print("\n" + "-" * 50)
    print("  Stage 1 standalone test (3-class)")
    print("-" * 50)
    s1_test_loader = DataLoader(s1_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    s1_test_metrics, s1_test_preds, s1_test_true = evaluate_model(
        s1_model, s1_test_loader, DEVICE, s1_weights
    )
    print(classification_report(
        s1_test_true, s1_test_preds,
        target_names=STAGE1_LABELS, digits=4, zero_division=0,
    ))

    print("-" * 50)
    print("  Stage 2 standalone test (binary)")
    print("-" * 50)
    s2_test_loader = DataLoader(s2_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    s2_test_metrics, s2_test_preds, s2_test_true = evaluate_model(
        s2_model, s2_test_loader, DEVICE, s2_weights
    )
    print(classification_report(
        s2_test_true, s2_test_preds,
        target_names=STAGE2_LABELS, digits=4, zero_division=0,
    ))

    print("\n" + "=" * 70)
    print("  TWO-STAGE MODEL — TEST RESULTS (4-class)")
    print("=" * 70)

    test_texts = hier_test["text_light_clean"].astype(str).tolist()
    test_true_4class = hier_test["label4_class"].tolist()

    print("[INFO] Running 2-stage inference on test set...")
    hier_pred_4class = batched_hierarchical_predict(
        test_texts, s1_model, s2_model, tokenizer, MAX_LENGTH, DEVICE,
        stage1_id2label, stage2_id2label, batch_size=BATCH_SIZE,
    )

    print(classification_report(
        test_true_4class, hier_pred_4class,
        labels=FLAT_4CLASS_LABELS, target_names=FLAT_4CLASS_LABELS,
        digits=4, zero_division=0,
    ))

    hier_acc = accuracy_score(test_true_4class, hier_pred_4class)
    hier_macro_f1 = f1_score(test_true_4class, hier_pred_4class,
                             labels=FLAT_4CLASS_LABELS, average="macro", zero_division=0)
    hier_weighted_f1 = f1_score(test_true_4class, hier_pred_4class,
                                labels=FLAT_4CLASS_LABELS, average="weighted", zero_division=0)
    hier_macro_p = precision_score(test_true_4class, hier_pred_4class,
                                   labels=FLAT_4CLASS_LABELS, average="macro", zero_division=0)
    hier_macro_r = recall_score(test_true_4class, hier_pred_4class,
                                labels=FLAT_4CLASS_LABELS, average="macro", zero_division=0)

    print("\n" + "=" * 70)
    print("  FINAL COMPARISON: FLAT vs TWO-STAGE")
    print("=" * 70)

    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"],
        "Flat 4-Class": [
            f"{flat_acc:.4f}", f"{flat_macro_p:.4f}", f"{flat_macro_r:.4f}",
            f"{flat_macro_f1:.4f}", f"{flat_weighted_f1:.4f}",
        ],
        "Two-Stage Hierarchical": [
            f"{hier_acc:.4f}", f"{hier_macro_p:.4f}", f"{hier_macro_r:.4f}",
            f"{hier_macro_f1:.4f}", f"{hier_weighted_f1:.4f}",
        ],
    })

    flat_vals = [flat_acc, flat_macro_p, flat_macro_r, flat_macro_f1, flat_weighted_f1]
    hier_vals = [hier_acc, hier_macro_p, hier_macro_r, hier_macro_f1, hier_weighted_f1]
    winners = []
    for fv, hv in zip(flat_vals, hier_vals):
        if fv > hv + 0.0001:
            winners.append("Flat")
        elif hv > fv + 0.0001:
            winners.append("Two-Stage")
        else:
            winners.append("Tie")
    comparison["Winner"] = winners

    print(comparison.to_string(index=False))
    comparison.to_csv(os.path.join(OUTPUT_DIR, "comparison_summary.csv"), index=False)

    print("\n" + "-" * 70)
    print("  PER-CLASS F1 COMPARISON")
    print("-" * 70)

    flat_report = classification_report(
        flat_test_true_labels, flat_test_pred_labels,
        labels=FLAT_4CLASS_LABELS, output_dict=True, zero_division=0,
    )
    hier_report = classification_report(
        test_true_4class, hier_pred_4class,
        labels=FLAT_4CLASS_LABELS, output_dict=True, zero_division=0,
    )

    per_class_rows = []
    for label in FLAT_4CLASS_LABELS:
        flat_f1_c = flat_report[label]["f1-score"]
        hier_f1_c = hier_report[label]["f1-score"]
        diff = hier_f1_c - flat_f1_c
        w = "Two-Stage" if diff > 0.0001 else ("Flat" if diff < -0.0001 else "Tie")
        per_class_rows.append({
            "Class": label,
            "Flat F1": f"{flat_f1_c:.4f}",
            "Two-Stage F1": f"{hier_f1_c:.4f}",
            "Diff (2S - Flat)": f"{diff:+.4f}",
            "Winner": w,
        })

    per_class_df = pd.DataFrame(per_class_rows)
    print(per_class_df.to_string(index=False))
    per_class_df.to_csv(os.path.join(OUTPUT_DIR, "per_class_f1_comparison.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cm_flat = confusion_matrix(flat_test_true_labels, flat_test_pred_labels, labels=FLAT_4CLASS_LABELS)
    sns.heatmap(cm_flat, annot=True, fmt="d", cmap="Blues",
                xticklabels=FLAT_4CLASS_LABELS, yticklabels=FLAT_4CLASS_LABELS, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Flat 4-Class (Macro F1={flat_macro_f1:.4f})")

    cm_hier = confusion_matrix(test_true_4class, hier_pred_4class, labels=FLAT_4CLASS_LABELS)
    sns.heatmap(cm_hier, annot=True, fmt="d", cmap="Oranges",
                xticklabels=FLAT_4CLASS_LABELS, yticklabels=FLAT_4CLASS_LABELS, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"Two-Stage Hierarchical (Macro F1={hier_macro_f1:.4f})")

    plt.suptitle("Flat vs Two-Stage MN-BERT — Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices_comparison.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n[INFO] Confusion matrices saved to {cm_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    flat_epochs = [e["epoch"] for e in flat_epoch_log]
    flat_val_f1s = [e["val_macro_f1"] for e in flat_epoch_log]
    flat_train_losses = [e["train_loss"] for e in flat_epoch_log]

    s1_epochs = [e["epoch"] for e in s1_epoch_log]
    s1_val_f1s = [e["val_macro_f1"] for e in s1_epoch_log]
    s1_train_losses = [e["train_loss"] for e in s1_epoch_log]

    s2_epochs_list = [e["epoch"] for e in s2_epoch_log]
    s2_val_f1s = [e["val_macro_f1"] for e in s2_epoch_log]

    axes[0].plot(flat_epochs, flat_val_f1s, label="Flat 4-class Val F1", linewidth=2)
    axes[0].plot(s1_epochs, s1_val_f1s, label="Stage1 3-class Val F1", linewidth=2, linestyle="--")
    axes[0].plot(s2_epochs_list, s2_val_f1s, label="Stage2 binary Val F1", linewidth=2, linestyle=":")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Validation Macro F1 over Epochs")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(flat_epochs, flat_train_losses, label="Flat 4-class", linewidth=2)
    axes[1].plot(s1_epochs, s1_train_losses, label="Stage1 3-class", linewidth=2, linestyle="--")
    s2_train_losses = [e["train_loss"] for e in s2_epoch_log]
    axes[1].plot(s2_epochs_list, s2_train_losses, label="Stage2 binary", linewidth=2, linestyle=":")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train Loss")
    axes[1].set_title("Training Loss over Epochs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training Curves Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    curves_path = os.path.join(OUTPUT_DIR, "training_curves_comparison.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved to {curves_path}")

    pd.DataFrame(flat_epoch_log).to_csv(os.path.join(OUTPUT_DIR, "flat_epoch_log.csv"), index=False)
    pd.DataFrame(s1_epoch_log).to_csv(os.path.join(OUTPUT_DIR, "stage1_epoch_log.csv"), index=False)
    pd.DataFrame(s2_epoch_log).to_csv(os.path.join(OUTPUT_DIR, "stage2_epoch_log.csv"), index=False)

    print("\n" + "-" * 70)
    print("  MISLABELED ROWS")
    print("-" * 70)

    flat_test_reset = flat_test.reset_index(drop=True)
    flat_mislabeled = []
    for i in range(len(flat_test_true_labels)):
        if flat_test_true_labels[i] != flat_test_pred_labels[i]:
            row = flat_test_reset.iloc[i]
            flat_mislabeled.append({
                "id": row.get("id", ""),
                "text_raw": row.get("text_raw", ""),
                "text_light_clean": row.get("text_light_clean", ""),
                "true_label": flat_test_true_labels[i],
                "predicted_label": flat_test_pred_labels[i],
                "source": row.get("source", ""),
            })

    flat_mislabeled_df = pd.DataFrame(flat_mislabeled)
    flat_mis_path = os.path.join(OUTPUT_DIR, "mislabeled_flat_model.csv")
    flat_mislabeled_df.to_csv(flat_mis_path, index=False, encoding="utf-8-sig")
    print(f"  Flat model: {len(flat_mislabeled)}/{len(flat_test_true_labels)} mislabeled ({len(flat_mislabeled)/len(flat_test_true_labels)*100:.1f}%)")
    print(f"  Saved: {flat_mis_path}")

    hier_test_reset = hier_test.reset_index(drop=True)
    hier_mislabeled = []
    for i in range(len(test_true_4class)):
        if test_true_4class[i] != hier_pred_4class[i]:
            row = hier_test_reset.iloc[i]
            hier_mislabeled.append({
                "id": row.get("id", ""),
                "text_raw": row.get("text_raw", ""),
                "text_light_clean": row.get("text_light_clean", ""),
                "true_label": test_true_4class[i],
                "predicted_label": hier_pred_4class[i],
                "source": row.get("source", ""),
            })

    hier_mislabeled_df = pd.DataFrame(hier_mislabeled)
    hier_mis_path = os.path.join(OUTPUT_DIR, "mislabeled_twostage_model.csv")
    hier_mislabeled_df.to_csv(hier_mis_path, index=False, encoding="utf-8-sig")
    print(f"  Two-stage model: {len(hier_mislabeled)}/{len(test_true_4class)} mislabeled ({len(hier_mislabeled)/len(test_true_4class)*100:.1f}%)")
    print(f"  Saved: {hier_mis_path}")

    from collections import Counter
    print("\n  Flat model — top confusion pairs:")
    flat_confusion = Counter((r["true_label"], r["predicted_label"]) for r in flat_mislabeled)
    for (true, pred), count in sorted(flat_confusion.items(), key=lambda x: -x[1])[:10]:
        print(f"    {true:>15} -> {pred:<15} {count}")

    print("\n  Two-stage model — top confusion pairs:")
    hier_confusion = Counter((r["true_label"], r["predicted_label"]) for r in hier_mislabeled)
    for (true, pred), count in sorted(hier_confusion.items(), key=lambda x: -x[1])[:10]:
        print(f"    {true:>15} -> {pred:<15} {count}")

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    if flat_macro_f1 > hier_macro_f1 + 0.001:
        print(f"  >>> FLAT 4-CLASS model is currently better (Macro F1: {flat_macro_f1:.4f} vs {hier_macro_f1:.4f})")
    elif hier_macro_f1 > flat_macro_f1 + 0.001:
        print(f"  >>> TWO-STAGE HIERARCHICAL model is currently better (Macro F1: {hier_macro_f1:.4f} vs {flat_macro_f1:.4f})")
    else:
        print(f"  >>> Models are roughly equivalent (Flat F1: {flat_macro_f1:.4f}, Two-Stage F1: {hier_macro_f1:.4f})")

    print(f"\n  Flat:      Acc={flat_acc:.4f}  Macro-F1={flat_macro_f1:.4f}  Weighted-F1={flat_weighted_f1:.4f}")
    print(f"  Two-Stage: Acc={hier_acc:.4f}  Macro-F1={hier_macro_f1:.4f}  Weighted-F1={hier_weighted_f1:.4f}")
    print(f"\n  All results saved to: {OUTPUT_DIR}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
