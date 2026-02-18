import os, re, time, numpy as np, torch, pandas as pd
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
    preds, logits_all, labels_all = [], [], []
    with torch.no_grad():
        for ids, mask, labs in loader:
            ids, mask = ids.to(device), mask.to(device)
            out = model(input_ids=ids, attention_mask=mask)
            logits_cpu = out.logits.cpu()
            preds.extend(torch.argmax(logits_cpu, dim=-1).numpy())
            logits_all.extend(logits_cpu.numpy())
            labels_all.extend(labs.numpy())
    return np.array(preds), np.array(logits_all), np.array(labels_all)

def main():
    print("=" * 70)
    print("  BERT Training — new_version_v1 data — 100 epochs (no early stopping)")
    print("=" * 70)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = get_device()
    print(f"  Device: {device}")
    print(f"  Epochs: {BERT_EPOCHS}, No early stopping, LR: {BERT_LR}")

    splits = load_and_prepare()

    def _clean(s):
        return s.fillna("").astype(str).str.strip().apply(
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
    print(f"  Class weights: {dict(zip(LABEL_NAMES, [f'{cw:.3f}' for cw in class_weights]))}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    print("  Tokenizing ...")
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

    print(f"  Steps/epoch: {len(train_loader)}, Total: {total_steps}, Warmup: {warmup_steps}")

    best_f1 = 0.0
    best_state = None
    best_epoch = 0
    epoch_log = []

    print(f"\n{'='*70}")
    print(f"  {'Epoch':>5} | {'Loss':>8} | {'Val Acc':>8} | {'Val F1-M':>8} | {'Val F1-W':>8} | {'Time':>6} | Status")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+--------")

    for epoch in range(BERT_EPOCHS):
        t0 = time.time()

        model.train()
        total_loss = 0
        for ids, mask, labs in train_loader:
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

        val_preds, _, _ = evaluate(model, val_loader, device)
        val_pred_labels = [ID2LABEL[p] for p in val_preds]
        val_f1 = f1_score(y_val, val_pred_labels, average="macro", zero_division=0)
        val_f1w = f1_score(y_val, val_pred_labels, average="weighted", zero_division=0)
        val_acc = accuracy_score(y_val, val_pred_labels)

        elapsed = time.time() - t0
        status = ""

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            status = "* BEST *"
        else:
            status = f"  +{epoch + 1 - best_epoch}"

        epoch_log.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_f1_macro": round(val_f1, 4),
            "val_f1_weighted": round(val_f1w, 4),
            "time_sec": round(elapsed, 1),
            "status": status,
        })

        print(f"  {epoch+1:>5} | {avg_loss:>8.4f} | {val_acc:>8.4f} | {val_f1:>8.4f} | {val_f1w:>8.4f} | {elapsed:>5.0f}s | {status}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    epoch_df = pd.DataFrame(epoch_log)
    epoch_report_path = os.path.join(REPORTS_DIR, "epoch_report_v1.csv")

    print(f"\n{'='*70}")
    print(f"  Best model from epoch {best_epoch} (Val Macro-F1 = {best_f1:.4f})")
    print(f"{'='*70}")

    model.load_state_dict(best_state)
    model.to(device)

    val_preds, val_logits, _ = evaluate(model, val_loader, device)
    val_pred_labels = np.array([ID2LABEL[p] for p in val_preds])
    val_acc = accuracy_score(y_val, val_pred_labels)
    val_f1m = f1_score(y_val, val_pred_labels, average="macro", zero_division=0)
    print(f"\n  VAL:  Accuracy={val_acc:.4f}  Macro-F1={val_f1m:.4f}")
    print(classification_report(y_val, val_pred_labels, labels=LABEL_NAMES, zero_division=0))

    test_preds, test_logits, _ = evaluate(model, test_loader, device)
    test_pred_labels = np.array([ID2LABEL[p] for p in test_preds])
    test_acc = accuracy_score(y_test, test_pred_labels)
    test_f1m = f1_score(y_test, test_pred_labels, average="macro", zero_division=0)
    test_f1w = f1_score(y_test, test_pred_labels, average="weighted", zero_division=0)
    print(f"  TEST: Accuracy={test_acc:.4f}  Macro-F1={test_f1m:.4f}")
    print(classification_report(y_test, test_pred_labels, labels=LABEL_NAMES, zero_division=0))

    epoch_df["test_accuracy"] = np.nan
    epoch_df["test_f1_macro"] = np.nan
    epoch_df["test_f1_weighted"] = np.nan
    best_idx = epoch_df.index[epoch_df["epoch"] == best_epoch][0]
    epoch_df.loc[best_idx, "test_accuracy"] = round(test_acc, 4)
    epoch_df.loc[best_idx, "test_f1_macro"] = round(test_f1m, 4)
    epoch_df.loc[best_idx, "test_f1_weighted"] = round(test_f1w, 4)
    epoch_df.to_csv(epoch_report_path, index=False)
    print(f"\n  Epoch report saved (with test results): {epoch_report_path}")

    mislabeled_rows = []

    for split_name, preds, logits, y_true, df_orig in [
        ("val", val_preds, val_logits, y_val, val_df),
        ("test", test_preds, test_logits, y_test, test_df),
    ]:
        pred_labels = [ID2LABEL[p] for p in preds]
        for i in range(len(y_true)):
            if y_true[i] != pred_labels[i]:
                row = df_orig.iloc[i]
                softmax = np.exp(logits[i]) / np.exp(logits[i]).sum()
                mislabeled_rows.append({
                    "split": split_name,
                    "id": row.get("id", ""),
                    "text_raw": row.get("text_raw", ""),
                    "text_light_clean": row.get("text_light_clean", ""),
                    "true_label": y_true[i],
                    "predicted_label": pred_labels[i],
                    "confidence": f"{softmax[preds[i]]:.4f}",
                    "source": row.get("source", ""),
                    **{f"score_{name}": f"{softmax[j]:.4f}" for j, name in enumerate(LABEL_NAMES)},
                })

    mislabeled_df = pd.DataFrame(mislabeled_rows)
    mislabeled_path = os.path.join(REPORTS_DIR, "mislabeled_rows_v1.csv")
    mislabeled_df.to_csv(mislabeled_path, index=False, encoding="utf-8-sig")

    val_wrong = sum(1 for r in mislabeled_rows if r["split"] == "val")
    test_wrong = sum(1 for r in mislabeled_rows if r["split"] == "test")

    print(f"\n{'='*70}")
    print(f"  MISLABELED ROWS")
    print(f"{'='*70}")
    print(f"  Val:  {val_wrong}/{len(y_val)} wrong ({val_wrong/len(y_val)*100:.1f}%)")
    print(f"  Test: {test_wrong}/{len(y_test)} wrong ({test_wrong/len(y_test)*100:.1f}%)")
    print(f"  Total: {len(mislabeled_rows)} mislabeled rows")
    print(f"  Saved: {mislabeled_path}")

    from collections import Counter
    confusion = Counter((r["true_label"], r["predicted_label"]) for r in mislabeled_rows)
    print(f"\n  Top confusion pairs:")
    for (true, pred), count in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
        print(f"    {true:>15} -> {pred:<15} {count}")

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Data:             sampled_fully_labeled_new_version_v1.csv")
    print(f"  Device:           {device}")
    print(f"  Best epoch:       {best_epoch}/{len(epoch_log)}")
    print(f"  Val Macro-F1:     {val_f1m:.4f}")
    print(f"  Test Accuracy:    {test_acc:.4f}")
    print(f"  Test Macro-F1:    {test_f1m:.4f}")
    print(f"  Test Weighted-F1: {test_f1w:.4f}")
    print(f"  Epoch report:     {epoch_report_path}")
    print(f"  Mislabeled rows:  {mislabeled_path}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
