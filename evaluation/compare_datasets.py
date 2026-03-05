import os, re, copy, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.config import (
    LABEL_NAMES, RANDOM_SEED, MODELS_DIR,
    BERT_MODEL_NAME, BERT_TEXT_COLUMN, BERT_MAX_LENGTH,
    BERT_BATCH_SIZE, BERT_EPOCHS, BERT_LR, BERT_WARMUP_RATIO,
    PROJECT_ROOT,
)

LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}

OLD_CSV = os.path.join(PROJECT_ROOT, "data", "sampled_labeled_fully_new.csv")
NEW_CSV = os.path.join(PROJECT_ROOT, "data", "sampled_labeled_fully_newly.csv")

PATIENCE = 3

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")

def load_splits(csv_path):
    import pandas as pd
    from src.config import LABEL_COLUMN, SPLIT_COLUMN

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    for col in list(df.columns):
        if col.strip() == "":
            df = df.drop(columns=[col])

    df = df.dropna(subset=[BERT_TEXT_COLUMN])
    df = df[df[BERT_TEXT_COLUMN].str.strip() != ""]
    df[BERT_TEXT_COLUMN] = df[BERT_TEXT_COLUMN].astype(str).str.strip().apply(lambda t: re.sub(r"\s+", " ", t))
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.upper()
    df = df[df[LABEL_COLUMN].isin(LABEL_NAMES)]
    df[SPLIT_COLUMN] = df[SPLIT_COLUMN].astype(str).str.strip().str.lower()

    splits = {}
    for s in ["train", "val", "test"]:
        sdf = df[df[SPLIT_COLUMN] == s]
        splits[s] = {
            "X": sdf[BERT_TEXT_COLUMN].values,
            "y": sdf[LABEL_COLUMN].values,
        }
    return splits

def tokenize(texts, tokenizer):
    enc = tokenizer(list(texts), padding="max_length", truncation=True,
                    max_length=BERT_MAX_LENGTH, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for ids, mask, labs in loader:
            ids, mask, labs = ids.to(device), mask.to(device), labs.to(device)
            out = model(input_ids=ids, attention_mask=mask)
            p = torch.argmax(out.logits.cpu(), dim=-1)
            preds.extend(p.numpy())
            labels.extend(labs.cpu().numpy())
    return np.array(preds), np.array(labels)

def run_experiment(csv_path, label, device, tokenizer):
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {label}")
    print(f"  Data: {os.path.basename(csv_path)}")
    print(f"  Epochs: {BERT_EPOCHS}, Patience: {PATIENCE}, LR: {BERT_LR}")
    print(f"{'='*70}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    splits = load_splits(csv_path)
    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]
    X_test, y_test = splits["test"]["X"], splits["test"]["y"]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    for lbl in LABEL_NAMES:
        print(f"    {lbl}: train={sum(y_train==lbl)} val={sum(y_val==lbl)} test={sum(y_test==lbl)}")

    unique = np.unique(y_train)
    w = compute_class_weight("balanced", classes=unique, y=y_train)
    wmap = dict(zip(unique, w))
    class_weights = [wmap[n] for n in LABEL_NAMES]

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
    epoch_log = []

    print(f"\n  Training ({len(train_loader)} steps/epoch, {total_steps} total) ...")
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

        val_preds, val_labs = evaluate(model, val_loader, device)
        val_pred_labels = [ID2LABEL[p] for p in val_preds]
        val_f1 = f1_score(y_val, val_pred_labels, average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, val_pred_labels)

        epoch_log.append({"epoch": epoch + 1, "loss": avg_loss, "val_acc": val_acc, "val_f1": val_f1})
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

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}{improved}")

    model.load_state_dict(best_state)
    model.to(device)
    test_preds, test_labs = evaluate(model, test_loader, device)
    test_pred_labels = [ID2LABEL[p] for p in test_preds]

    test_acc = accuracy_score(y_test, test_pred_labels)
    test_f1m = f1_score(y_test, test_pred_labels, average="macro", zero_division=0)
    test_f1w = f1_score(y_test, test_pred_labels, average="weighted", zero_division=0)

    print(f"\n  --- {label} TEST RESULTS ---")
    print(classification_report(y_test, test_pred_labels, labels=LABEL_NAMES, zero_division=0))

    return {
        "label": label,
        "csv": os.path.basename(csv_path),
        "best_val_f1": best_f1,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1m,
        "test_f1_weighted": test_f1w,
        "epochs_trained": len(epoch_log),
        "epoch_log": epoch_log,
    }

def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Epochs: {BERT_EPOCHS}, Patience: {PATIENCE}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    old_result = run_experiment(OLD_CSV, "OLD_LABELS", device, tokenizer)
    new_result = run_experiment(NEW_CSV, "NEW_LABELS", device, tokenizer)

    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON — OLD vs NEW LABELS")
    print(f"{'='*70}")
    print(f"  {'Metric':<22} {'OLD_LABELS':>12} {'NEW_LABELS':>12} {'Diff':>10}")
    print(f"  {'-'*56}")

    for key, name in [
        ("best_val_f1", "Val Macro-F1 (best)"),
        ("test_accuracy", "Test Accuracy"),
        ("test_f1_macro", "Test Macro-F1"),
        ("test_f1_weighted", "Test Weighted-F1"),
    ]:
        old_v = old_result[key]
        new_v = new_result[key]
        diff = new_v - old_v
        sign = "+" if diff >= 0 else ""
        print(f"  {name:<22} {old_v:>12.4f} {new_v:>12.4f} {sign}{diff:>9.4f}")

    print(f"  {'-'*56}")
    print(f"  {'Epochs trained':<22} {old_result['epochs_trained']:>12} {new_result['epochs_trained']:>12}")
    print(f"{'='*70}")

    print(f"\n  EPOCH-BY-EPOCH VAL F1")
    print(f"  {'Epoch':<8} {'OLD':>10} {'NEW':>10}")
    max_ep = max(len(old_result["epoch_log"]), len(new_result["epoch_log"]))
    for i in range(max_ep):
        old_f1 = old_result["epoch_log"][i]["val_f1"] if i < len(old_result["epoch_log"]) else ""
        new_f1 = new_result["epoch_log"][i]["val_f1"] if i < len(new_result["epoch_log"]) else ""
        old_s = f"{old_f1:.4f}" if isinstance(old_f1, float) else "—"
        new_s = f"{new_f1:.4f}" if isinstance(new_f1, float) else "—"
        print(f"  {i+1:<8} {old_s:>10} {new_s:>10}")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
