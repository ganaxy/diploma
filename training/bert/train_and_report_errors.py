import os, re, numpy as np, torch, pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.config import (
    LABEL_NAMES, RANDOM_SEED, MODELS_DIR, REPORTS_DIR,
    BERT_MODEL_NAME, BERT_TEXT_COLUMN, BERT_MAX_LENGTH,
    BERT_BATCH_SIZE, BERT_EPOCHS, BERT_LR, BERT_WARMUP_RATIO,
)
from src.data_prep import load_and_prepare

LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
PATIENCE = 3

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")

def tokenize(texts, tokenizer):
    enc = tokenizer(list(texts), padding="max_length", truncation=True,
                    max_length=BERT_MAX_LENGTH, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]

def evaluate(model, loader, device):
    model.eval()
    preds, logits_all = [], []
    with torch.no_grad():
        for ids, mask, labs in loader:
            ids, mask = ids.to(device), mask.to(device)
            out = model(input_ids=ids, attention_mask=mask)
            logits_cpu = out.logits.cpu()
            preds.extend(torch.argmax(logits_cpu, dim=-1).numpy())
            logits_all.extend(logits_cpu.numpy())
    return np.array(preds), np.array(logits_all)

def main():
    print("=" * 60)
    print("  BERT Training (Old Data, 10 epochs) + Mislabel Report")
    print("=" * 60)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = get_device()
    print(f"Device: {device}")
    print(f"Epochs: {BERT_EPOCHS}, Patience: {PATIENCE}")

    splits = load_and_prepare()

    def _clean(series):
        return series.fillna("").astype(str).str.strip().apply(
            lambda t: re.sub(r"\s+", " ", t)).values

    X_train = _clean(splits["train"]["df"][BERT_TEXT_COLUMN])
    y_train = splits["train"]["y"]
    X_val = _clean(splits["val"]["df"][BERT_TEXT_COLUMN])
    y_val = splits["val"]["y"]
    X_test = _clean(splits["test"]["df"][BERT_TEXT_COLUMN])
    y_test = splits["test"]["y"]

    val_df = splits["val"]["df"].reset_index(drop=True)
    test_df = splits["test"]["df"].reset_index(drop=True)

    unique = np.unique(y_train)
    w = compute_class_weight("balanced", classes=unique, y=y_train)
    wmap = dict(zip(unique, w))
    class_weights = [wmap[n] for n in LABEL_NAMES]
    print(f"Class weights: {dict(zip(LABEL_NAMES, [f'{cw:.3f}' for cw in class_weights]))}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    print("Tokenizing ...")
    train_ids, train_mask = tokenize(X_train, tokenizer)
    val_ids, val_mask = tokenize(X_val, tokenizer)
    test_ids, test_mask = tokenize(X_test, tokenizer)

    train_labels = torch.tensor([LABEL2ID[l] for l in y_train], dtype=torch.long)
    val_labels = torch.tensor([LABEL2ID[l] for l in y_val], dtype=torch.long)
    test_labels = torch.tensor([LABEL2ID[l] for l in y_test], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_ids, train_mask, train_labels),
                              batch_size=BERT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_ids, val_mask, val_labels),
                            batch_size=BERT_BATCH_SIZE * 2)
    test_loader = DataLoader(TensorDataset(test_ids, test_mask, test_labels),
                             batch_size=BERT_BATCH_SIZE * 2)

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)
    total_steps = len(train_loader) * BERT_EPOCHS
    warmup_steps = int(BERT_WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )

    best_f1 = 0.0
    patience_counter = 0
    best_state = None

    print(f"\nTraining ({len(train_loader)} steps/epoch) ...")
    for epoch in range(BERT_EPOCHS):
        model.train()
        total_loss = 0
        for step, (ids, mask, labs) in enumerate(train_loader):
            ids, mask, labs = ids.to(device), mask.to(device), labs.to(device)
            out = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        val_preds, _ = evaluate(model, val_loader, device)
        val_pred_labels = [ID2LABEL[p] for p in val_preds]
        val_f1 = f1_score(y_val, val_pred_labels, average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, val_pred_labels)

        improved = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = " *BEST*"
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} — early stop")
                break

        print(f"  Epoch {epoch+1}/{BERT_EPOCHS}: loss={avg_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}{improved}")

    print(f"\nBest val Macro-F1: {best_f1:.4f}")
    model.load_state_dict(best_state)
    model.to(device)

    print("\n" + "=" * 60)
    print("  GENERATING MISLABELED ROWS REPORT")
    print("=" * 60)

    mislabeled_rows = []

    val_preds, val_logits = evaluate(model, val_loader, device)
    val_pred_labels = np.array([ID2LABEL[p] for p in val_preds])
    val_acc = accuracy_score(y_val, val_pred_labels)
    val_f1m = f1_score(y_val, val_pred_labels, average="macro", zero_division=0)
    print(f"\nVal Accuracy: {val_acc:.4f}, Val Macro-F1: {val_f1m:.4f}")
    print(classification_report(y_val, val_pred_labels, labels=LABEL_NAMES, zero_division=0))

    for i in range(len(y_val)):
        if y_val[i] != val_pred_labels[i]:
            row = val_df.iloc[i]
            confidence_scores = {LABEL_NAMES[j]: float(val_logits[i][j]) for j in range(len(LABEL_NAMES))}
            softmax = np.exp(val_logits[i]) / np.exp(val_logits[i]).sum()
            mislabeled_rows.append({
                "split": "val",
                "id": row.get("id", ""),
                "text_raw": row.get("text_raw", ""),
                "text_light_clean": row.get("text_light_clean", ""),
                "true_label": y_val[i],
                "predicted_label": val_pred_labels[i],
                "confidence": f"{softmax[val_preds[i]]:.4f}",
                "source": row.get("source", ""),
                **{f"score_{name}": f"{softmax[j]:.4f}" for j, name in enumerate(LABEL_NAMES)},
            })

    test_preds, test_logits = evaluate(model, test_loader, device)
    test_pred_labels = np.array([ID2LABEL[p] for p in test_preds])
    test_acc = accuracy_score(y_test, test_pred_labels)
    test_f1m = f1_score(y_test, test_pred_labels, average="macro", zero_division=0)
    test_f1w = f1_score(y_test, test_pred_labels, average="weighted", zero_division=0)
    print(f"\nTest Accuracy: {test_acc:.4f}, Test Macro-F1: {test_f1m:.4f}")
    print(classification_report(y_test, test_pred_labels, labels=LABEL_NAMES, zero_division=0))

    for i in range(len(y_test)):
        if y_test[i] != test_pred_labels[i]:
            row = test_df.iloc[i]
            softmax = np.exp(test_logits[i]) / np.exp(test_logits[i]).sum()
            mislabeled_rows.append({
                "split": "test",
                "id": row.get("id", ""),
                "text_raw": row.get("text_raw", ""),
                "text_light_clean": row.get("text_light_clean", ""),
                "true_label": y_test[i],
                "predicted_label": test_pred_labels[i],
                "confidence": f"{softmax[test_preds[i]]:.4f}",
                "source": row.get("source", ""),
                **{f"score_{name}": f"{softmax[j]:.4f}" for j, name in enumerate(LABEL_NAMES)},
            })

    os.makedirs(REPORTS_DIR, exist_ok=True)
    mislabeled_df = pd.DataFrame(mislabeled_rows)
    report_path = os.path.join(REPORTS_DIR, "mislabeled_rows.csv")
    mislabeled_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    val_wrong = len([r for r in mislabeled_rows if r["split"] == "val"])
    test_wrong = len([r for r in mislabeled_rows if r["split"] == "test"])

    print(f"\n{'='*60}")
    print(f"  MISLABELED ROWS REPORT")
    print(f"{'='*60}")
    print(f"  Val:  {val_wrong}/{len(y_val)} mislabeled ({val_wrong/len(y_val)*100:.1f}%)")
    print(f"  Test: {test_wrong}/{len(y_test)} mislabeled ({test_wrong/len(y_test)*100:.1f}%)")
    print(f"  Total: {len(mislabeled_rows)} mislabeled rows")
    print(f"\n  Breakdown by true_label -> predicted_label:")

    from collections import Counter
    confusion = Counter((r["true_label"], r["predicted_label"]) for r in mislabeled_rows)
    for (true, pred), count in sorted(confusion.items(), key=lambda x: -x[1]):
        print(f"    {true} -> {pred}: {count}")

    print(f"\n  Saved: {report_path}")
    print(f"\n{'='*60}")
    print(f"  FINAL SCORES")
    print(f"{'='*60}")
    print(f"  Test Accuracy:    {test_acc:.4f}")
    print(f"  Test Macro-F1:    {test_f1m:.4f}")
    print(f"  Test Weighted-F1: {test_f1w:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
