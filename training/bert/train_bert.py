import os
import re
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score

from src.config import (
    LABEL_NAMES, RANDOM_SEED, MODELS_DIR, REPORTS_DIR,
    BERT_MODEL_NAME, BERT_TEXT_COLUMN, BERT_MAX_LENGTH,
    BERT_BATCH_SIZE, BERT_EPOCHS, BERT_LR, BERT_WARMUP_RATIO,
)
from src.data_prep import load_and_prepare
from src.evaluate import full_evaluation

LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}

def get_device():
    if torch.cuda.is_available():
        print("[bert] Using CUDA GPU")
        return torch.device("cuda")

    try:
        import torch_directml
        dml_device = torch_directml.device()
        print("[bert] Using DirectML (AMD GPU)")
        return dml_device
    except ImportError:
        pass

    print("[bert] WARNING: No GPU detected — training will be slow.")
    print("[bert] Install torch-directml for AMD GPU support: pip install torch-directml")
    return torch.device("cpu")

def tokenize_texts(texts, tokenizer):
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=BERT_MAX_LENGTH,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]

def make_dataloader(input_ids, attention_mask, labels, batch_size, shuffle=False):
    dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits.cpu(), dim=-1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "preds": all_preds,
        "labels": all_labels,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }

def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(dataloader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % 50 == 0:
            avg = total_loss / num_batches
            print(f"    Step {step + 1}/{len(dataloader)}, Loss: {avg:.4f}")

    return total_loss / max(num_batches, 1)

def main():
    print("=" * 60)
    print("  Mongolian BERT Fine-tuning")
    print("=" * 60)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = get_device()

    print(f"\n[bert] Using text column: {BERT_TEXT_COLUMN}")
    splits = load_and_prepare()

    def _clean(series):
        return series.fillna("").astype(str).str.strip().apply(lambda t: re.sub(r"\s+", " ", t)).values

    X_train = _clean(splits["train"]["df"][BERT_TEXT_COLUMN])
    y_train = splits["train"]["y"]
    X_val = _clean(splits["val"]["df"][BERT_TEXT_COLUMN])
    y_val = splits["val"]["y"]
    X_test = _clean(splits["test"]["df"][BERT_TEXT_COLUMN])
    y_test = splits["test"]["y"]

    unique_labels = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=unique_labels, y=y_train)
    label_weight_map = dict(zip(unique_labels, weights))
    class_weights = [label_weight_map[name] for name in LABEL_NAMES]
    print(f"[bert] Class weights: {dict(zip(LABEL_NAMES, [f'{w:.3f}' for w in class_weights]))}")

    print(f"[bert] Loading tokenizer: {BERT_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    print("[bert] Tokenizing ...")
    train_ids, train_mask = tokenize_texts(X_train, tokenizer)
    val_ids, val_mask = tokenize_texts(X_val, tokenizer)
    test_ids, test_mask = tokenize_texts(X_test, tokenizer)

    train_labels = torch.tensor([LABEL2ID[l] for l in y_train], dtype=torch.long)
    val_labels = torch.tensor([LABEL2ID[l] for l in y_val], dtype=torch.long)
    test_labels = torch.tensor([LABEL2ID[l] for l in y_test], dtype=torch.long)

    train_loader = make_dataloader(train_ids, train_mask, train_labels, BERT_BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(val_ids, val_mask, val_labels, BERT_BATCH_SIZE * 2)
    test_loader = make_dataloader(test_ids, test_mask, test_labels, BERT_BATCH_SIZE * 2)

    print(f"[bert] Loading model: {BERT_MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print(f"[bert] Model moved to {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)

    total_steps = len(train_loader) * BERT_EPOCHS
    warmup_steps = int(BERT_WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[bert] Steps/epoch: {len(train_loader)}, total: {total_steps}, warmup: {warmup_steps}")

    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor)

    best_f1 = 0.0
    patience = 3
    patience_counter = 0
    best_model_dir = os.path.join(MODELS_DIR, "bert_mn_cased_best")
    os.makedirs(best_model_dir, exist_ok=True)

    print("\n[bert] Starting training ...")
    for epoch in range(BERT_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{BERT_EPOCHS} ---")

        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

        val_metrics = evaluate_model(model, val_loader, device)
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val Macro-F1: {val_metrics['f1_macro']:.4f}")
        print(f"  Val Weighted-F1: {val_metrics['f1_weighted']:.4f}")

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience_counter = 0
            cpu_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model_for_save = AutoModelForSequenceClassification.from_pretrained(
                BERT_MODEL_NAME, num_labels=len(LABEL_NAMES), id2label=ID2LABEL, label2id=LABEL2ID,
            )
            model_for_save.resize_token_embeddings(len(tokenizer))
            model_for_save.load_state_dict(cpu_state)
            model_for_save.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            del model_for_save, cpu_state
            print(f"  >> New best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch + 1}")
                break

    print("\n[bert] Training complete.")

    print(f"[bert] Loading best model from {best_model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    model.to(device)

    print("\n" + "=" * 60)
    print("  BERT VALIDATION EVALUATION")
    print("=" * 60)
    val_result_raw = evaluate_model(model, val_loader, device)
    val_pred_labels = np.array([ID2LABEL[p] for p in val_result_raw["preds"]])
    val_result = full_evaluation(X_val, y_val, val_pred_labels, "BERT_mn_cased", "val")

    print("\n" + "=" * 60)
    print("  BERT TEST EVALUATION")
    print("=" * 60)
    test_result_raw = evaluate_model(model, test_loader, device)
    test_pred_labels = np.array([ID2LABEL[p] for p in test_result_raw["preds"]])
    test_result = full_evaluation(X_test, y_test, test_pred_labels, "BERT_mn_cased", "test")

    summary = f"""
{'='*60}
  BERT EXPERIMENT COMPLETE
{'='*60}
  Model:        {BERT_MODEL_NAME}
  Text column:  {BERT_TEXT_COLUMN}
  Device:       {device}
  Val Macro-F1:  {val_result['f1_macro']:.4f}
  Test Macro-F1: {test_result['f1_macro']:.4f}
  Test Accuracy: {test_result['accuracy']:.4f}
  Saved model:   {best_model_dir}
{'='*60}
"""
    try:
        print(summary)
    except UnicodeEncodeError:
        print(summary.encode("ascii", errors="replace").decode("ascii"))

if __name__ == "__main__":
    main()
