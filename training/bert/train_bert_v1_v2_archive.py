import gc
import os
import re
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.config import (
    PROJECT_ROOT, LABEL_NAMES, RANDOM_SEED, MODELS_DIR, REPORTS_DIR,
    BERT_MODEL_NAME, BERT_TEXT_COLUMN, BERT_MAX_LENGTH,
    BERT_BATCH_SIZE, BERT_LR, BERT_WARMUP_RATIO,
)
from src.data_prep import load_and_prepare
from src.train_bert import (
    get_device, make_dataloader, LABEL2ID, ID2LABEL,
)

EPOCHS = 8
PATIENCE = 3

V2_CSV = os.path.join(PROJECT_ROOT, "..", "sampled_fully_labeled_old_v2.csv")
V1_CSV = os.path.join(PROJECT_ROOT, "..", "sampled_fully_labeled_old.csv")

def tokenize(texts, tokenizer):
    enc = tokenizer(
        list(texts), padding="max_length", truncation=True,
        max_length=BERT_MAX_LENGTH, return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]

def evaluate_with_logits(model, loader, device):
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

def train_experiment(csv_path, label, device, tokenizer):
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {label}")
    print(f"  Data: {os.path.basename(csv_path)}")
    print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}, LR: {BERT_LR}")
    print(f"{'='*70}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    splits = load_and_prepare(csv_path)

    def _clean(series):
        return (
            series.fillna("").astype(str).str.strip()
            .apply(lambda t: re.sub(r"\s+", " ", t)).values
        )

    X_train = _clean(splits["train"]["df"][BERT_TEXT_COLUMN])
    y_train = splits["train"]["y"]
    X_val = _clean(splits["val"]["df"][BERT_TEXT_COLUMN])
    y_val = splits["val"]["y"]
    X_test = _clean(splits["test"]["df"][BERT_TEXT_COLUMN])
    y_test = splits["test"]["y"]

    val_df = splits["val"]["df"].reset_index(drop=True)
    test_df = splits["test"]["df"].reset_index(drop=True)

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    for lbl in LABEL_NAMES:
        print(f"    {lbl}: train={sum(y_train==lbl)} val={sum(y_val==lbl)} test={sum(y_test==lbl)}")

    unique = np.unique(y_train)
    w = compute_class_weight("balanced", classes=unique, y=y_train)
    wmap = dict(zip(unique, w))
    class_weights = [wmap[n] for n in LABEL_NAMES]
    print(f"  Class weights: {dict(zip(LABEL_NAMES, [f'{cw:.3f}' for cw in class_weights]))}")

    print(f"  Tokenizing (max_length={BERT_MAX_LENGTH}) ...")
    train_ids, train_mask = tokenize(X_train, tokenizer)
    val_ids, val_mask = tokenize(X_val, tokenizer)
    test_ids, test_mask = tokenize(X_test, tokenizer)

    train_labels = torch.tensor([LABEL2ID[l] for l in y_train], dtype=torch.long)
    val_labels = torch.tensor([LABEL2ID[l] for l in y_val], dtype=torch.long)
    test_labels = torch.tensor([LABEL2ID[l] for l in y_test], dtype=torch.long)

    train_loader = make_dataloader(train_ids, train_mask, train_labels, BERT_BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(val_ids, val_mask, val_labels, BERT_BATCH_SIZE * 2)
    test_loader = make_dataloader(test_ids, test_mask, test_labels, BERT_BATCH_SIZE * 2)

    print(f"  Loading model: {BERT_MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(BERT_WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )
    print(f"  Steps/epoch: {len(train_loader)}, total: {total_steps}, warmup: {warmup_steps}")

    best_f1 = 0.0
    patience_counter = 0
    best_state = None
    epoch_log = []

    print(f"\n  Training ...")
    start_time = time.time()

    for epoch in range(EPOCHS):
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

        val_preds, _ = evaluate_with_logits(model, val_loader, device)
        val_pred_labels = [ID2LABEL[p] for p in val_preds]
        val_f1 = f1_score(y_val, val_pred_labels, average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, val_pred_labels)

        epoch_log.append({
            "epoch": epoch + 1, "loss": avg_loss,
            "val_acc": val_acc, "val_f1": val_f1,
        })

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

        print(f"  Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}{improved}")

    train_time = time.time() - start_time
    print(f"\n  Training done in {train_time:.0f}s, best val F1: {best_f1:.4f}")

    model.load_state_dict(best_state)
    model.to(device)

    val_preds, val_logits = evaluate_with_logits(model, val_loader, device)
    val_pred_labels = np.array([ID2LABEL[p] for p in val_preds])
    val_acc = accuracy_score(y_val, val_pred_labels)
    val_f1m = f1_score(y_val, val_pred_labels, average="macro", zero_division=0)
    val_f1w = f1_score(y_val, val_pred_labels, average="weighted", zero_division=0)

    print(f"\n  --- {label} VAL RESULTS ---")
    print(f"  Val Accuracy: {val_acc:.4f}, Macro-F1: {val_f1m:.4f}")
    print(classification_report(y_val, val_pred_labels, labels=LABEL_NAMES, zero_division=0))

    test_preds, test_logits = evaluate_with_logits(model, test_loader, device)
    test_pred_labels = np.array([ID2LABEL[p] for p in test_preds])
    test_acc = accuracy_score(y_test, test_pred_labels)
    test_f1m = f1_score(y_test, test_pred_labels, average="macro", zero_division=0)
    test_f1w = f1_score(y_test, test_pred_labels, average="weighted", zero_division=0)

    print(f"\n  --- {label} TEST RESULTS ---")
    print(f"  Test Accuracy: {test_acc:.4f}, Macro-F1: {test_f1m:.4f}")
    print(classification_report(y_test, test_pred_labels, labels=LABEL_NAMES, zero_division=0))

    val_report_dict = classification_report(
        y_val, val_pred_labels, labels=LABEL_NAMES,
        output_dict=True, zero_division=0,
    )
    test_report_dict = classification_report(
        y_test, test_pred_labels, labels=LABEL_NAMES,
        output_dict=True, zero_division=0,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label": label,
        "csv": os.path.basename(csv_path),
        "best_val_f1": best_f1,
        "train_time_s": train_time,
        "epochs_trained": len(epoch_log),
        "epoch_log": epoch_log,
        "val_accuracy": val_acc,
        "val_f1_macro": val_f1m,
        "val_f1_weighted": val_f1w,
        "val_report_dict": val_report_dict,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1m,
        "test_f1_weighted": test_f1w,
        "test_report_dict": test_report_dict,
        "y_val": y_val,
        "y_test": y_test,
        "val_preds": val_preds,
        "test_preds": test_preds,
        "val_logits": val_logits,
        "test_logits": test_logits,
        "val_pred_labels": val_pred_labels,
        "test_pred_labels": test_pred_labels,
        "val_df": val_df,
        "test_df": test_df,
    }

def generate_mislabeled_report(result):
    mislabeled_rows = []

    y_val = result["y_val"]
    val_pred_labels = result["val_pred_labels"]
    val_preds = result["val_preds"]
    val_logits = result["val_logits"]
    val_df = result["val_df"]

    for i in range(len(y_val)):
        if y_val[i] != val_pred_labels[i]:
            row = val_df.iloc[i]
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

    y_test = result["y_test"]
    test_pred_labels = result["test_pred_labels"]
    test_preds = result["test_preds"]
    test_logits = result["test_logits"]
    test_df = result["test_df"]

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

    return mislabeled_rows

def generate_comparison_report(v1_result, v2_result, mislabeled_rows):
    md = []
    md.append("# BERT Old Data V1 vs V2 — Comparison Report\n")
    md.append(f"**Model:** `{BERT_MODEL_NAME}`  ")
    md.append(f"**V1 Data:** `{v1_result['csv']}`  ")
    md.append(f"**V2 Data:** `{v2_result['csv']}`  ")
    md.append(f"**Max Epochs:** {EPOCHS}  ")
    md.append(f"**Early Stopping Patience:** {PATIENCE}  ")
    md.append(f"**Seed:** {RANDOM_SEED}\n")
    md.append("---\n")

    md.append("## Training Configuration\n")
    md.append("| Setting | Value |")
    md.append("|---------|-------|")
    md.append(f"| Model | `{BERT_MODEL_NAME}` |")
    md.append(f"| Text column | `{BERT_TEXT_COLUMN}` |")
    md.append(f"| Max epochs | {EPOCHS} |")
    md.append(f"| Learning rate | {BERT_LR} |")
    md.append(f"| Max length | {BERT_MAX_LENGTH} |")
    md.append(f"| Batch size | {BERT_BATCH_SIZE} |")
    md.append(f"| Warmup ratio | {BERT_WARMUP_RATIO} |")
    md.append(f"| Early stopping patience | {PATIENCE} |")
    md.append("")

    md.append("## Training Summary\n")
    md.append("| Property | V1 (Old Data) | V2 (Old Data V2) |")
    md.append("|----------|---------------|-------------------|")
    md.append(f"| CSV file | `{v1_result['csv']}` | `{v2_result['csv']}` |")
    md.append(f"| Epochs trained | {v1_result['epochs_trained']} | {v2_result['epochs_trained']} |")
    md.append(f"| Training time | {v1_result['train_time_s']:.0f}s | {v2_result['train_time_s']:.0f}s |")
    md.append(f"| Best val Macro-F1 | {v1_result['best_val_f1']:.4f} | {v2_result['best_val_f1']:.4f} |")
    md.append("")

    md.append("---\n")
    md.append("## Validation Results\n")
    md.append("| Metric | V1 | V2 | Delta |")
    md.append("|--------|----|----|-------|")
    for key, label in [
        ("val_accuracy", "Accuracy"),
        ("val_f1_macro", "Macro-F1"),
        ("val_f1_weighted", "Weighted-F1"),
    ]:
        v1_v = v1_result[key]
        v2_v = v2_result[key]
        delta = v2_v - v1_v
        sign = "+" if delta >= 0 else ""
        best_v1 = "**" if v1_v > v2_v else ""
        best_v2 = "**" if v2_v > v1_v else ""
        md.append(
            f"| {label} | {best_v1}{v1_v:.4f}{best_v1} | "
            f"{best_v2}{v2_v:.4f}{best_v2} | {sign}{delta:.4f} |"
        )
    md.append("")

    md.append("## Test Results\n")
    md.append("| Metric | V1 | V2 | Delta |")
    md.append("|--------|----|----|-------|")
    for key, label in [
        ("test_accuracy", "Accuracy"),
        ("test_f1_macro", "Macro-F1"),
        ("test_f1_weighted", "Weighted-F1"),
    ]:
        v1_v = v1_result[key]
        v2_v = v2_result[key]
        delta = v2_v - v1_v
        sign = "+" if delta >= 0 else ""
        best_v1 = "**" if v1_v > v2_v else ""
        best_v2 = "**" if v2_v > v1_v else ""
        md.append(
            f"| {label} | {best_v1}{v1_v:.4f}{best_v1} | "
            f"{best_v2}{v2_v:.4f}{best_v2} | {sign}{delta:.4f} |"
        )
    md.append("")

    md.append("## Per-Class Test F1\n")
    md.append("| Class | V1 | V2 | Delta |")
    md.append("|-------|----|----|-------|")
    for c in LABEL_NAMES:
        v1_f1 = v1_result["test_report_dict"].get(c, {}).get("f1-score", 0)
        v2_f1 = v2_result["test_report_dict"].get(c, {}).get("f1-score", 0)
        delta = v2_f1 - v1_f1
        sign = "+" if delta >= 0 else ""
        best_v1 = "**" if v1_f1 > v2_f1 else ""
        best_v2 = "**" if v2_f1 > v1_f1 else ""
        md.append(
            f"| {c} | {best_v1}{v1_f1:.4f}{best_v1} | "
            f"{best_v2}{v2_f1:.4f}{best_v2} | {sign}{delta:.4f} |"
        )
    md.append("")

    md.append("## Per-Class Val F1\n")
    md.append("| Class | V1 | V2 | Delta |")
    md.append("|-------|----|----|-------|")
    for c in LABEL_NAMES:
        v1_f1 = v1_result["val_report_dict"].get(c, {}).get("f1-score", 0)
        v2_f1 = v2_result["val_report_dict"].get(c, {}).get("f1-score", 0)
        delta = v2_f1 - v1_f1
        sign = "+" if delta >= 0 else ""
        best_v1 = "**" if v1_f1 > v2_f1 else ""
        best_v2 = "**" if v2_f1 > v1_f1 else ""
        md.append(
            f"| {c} | {best_v1}{v1_f1:.4f}{best_v1} | "
            f"{best_v2}{v2_f1:.4f}{best_v2} | {sign}{delta:.4f} |"
        )
    md.append("")

    md.append("---\n")
    md.append("## Training Progress\n")
    md.append("### Epoch-by-Epoch Val Macro-F1\n")
    md.append("| Epoch | V1 Loss | V1 Val F1 | V2 Loss | V2 Val F1 |")
    md.append("|-------|---------|-----------|---------|-----------|")
    max_ep = max(len(v1_result["epoch_log"]), len(v2_result["epoch_log"]))
    for i in range(max_ep):
        v1_loss = f"{v1_result['epoch_log'][i]['loss']:.4f}" if i < len(v1_result["epoch_log"]) else "—"
        v1_f1 = f"{v1_result['epoch_log'][i]['val_f1']:.4f}" if i < len(v1_result["epoch_log"]) else "—"
        v2_loss = f"{v2_result['epoch_log'][i]['loss']:.4f}" if i < len(v2_result["epoch_log"]) else "—"
        v2_f1 = f"{v2_result['epoch_log'][i]['val_f1']:.4f}" if i < len(v2_result["epoch_log"]) else "—"
        md.append(f"| {i+1} | {v1_loss} | {v1_f1} | {v2_loss} | {v2_f1} |")
    md.append("")

    md.append("---\n")
    md.append("## Mislabeled Rows Report (V2 Only)\n")
    val_wrong = len([r for r in mislabeled_rows if r["split"] == "val"])
    test_wrong = len([r for r in mislabeled_rows if r["split"] == "test"])
    total_val = len(v2_result["y_val"])
    total_test = len(v2_result["y_test"])
    md.append(f"- **Val:** {val_wrong}/{total_val} mislabeled ({val_wrong/total_val*100:.1f}%)")
    md.append(f"- **Test:** {test_wrong}/{total_test} mislabeled ({test_wrong/total_test*100:.1f}%)")
    md.append(f"- **Total:** {len(mislabeled_rows)} mislabeled rows")
    md.append(f"- **Saved to:** `outputs/reports/mislabeled_rows_v2.csv`\n")

    md.append("### Confusion Breakdown (V2)\n")
    md.append("| True Label | Predicted As | Count |")
    md.append("|------------|-------------|-------|")
    from collections import Counter
    confusion = Counter((r["true_label"], r["predicted_label"]) for r in mislabeled_rows)
    for (true, pred), count in sorted(confusion.items(), key=lambda x: -x[1]):
        md.append(f"| {true} | {pred} | {count} |")
    md.append("")

    md.append("---\n")

    best_test = "V2" if v2_result["test_f1_macro"] >= v1_result["test_f1_macro"] else "V1"
    best_val = "V2" if v2_result["val_f1_macro"] >= v1_result["val_f1_macro"] else "V1"
    md.append("## Summary\n")
    md.append(f"**Best on test set (Macro-F1):** {best_test}\n")
    md.append(f"**Best on val set (Macro-F1):** {best_val}\n")

    delta_test = v2_result["test_f1_macro"] - v1_result["test_f1_macro"]
    sign = "+" if delta_test >= 0 else ""
    md.append(
        f"**V2 vs V1 (test Macro-F1):** {sign}{delta_test:.4f} "
        f"({sign}{delta_test / max(v1_result['test_f1_macro'], 1e-9) * 100:.1f}% relative)\n"
    )

    md.append("---\n")
    md.append(f"*Model: `{BERT_MODEL_NAME}` | Epochs: {EPOCHS} | Seed: {RANDOM_SEED}*\n")

    return "\n".join(md)

def main():
    print("=" * 70)
    print("  BERT Old Data V1 vs V2 — Comparison + Mislabeled Report")
    print(f"  V1: {os.path.basename(V1_CSV)}")
    print(f"  V2: {os.path.basename(V2_CSV)}")
    print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    v2_result = train_experiment(V2_CSV, "OLD_V2", device, tokenizer)
    v1_result = train_experiment(V1_CSV, "OLD_V1", device, tokenizer)

    print("\n" + "=" * 70)
    print("  GENERATING MISLABELED ROWS REPORT (V2 ONLY)")
    print("=" * 70)

    mislabeled_rows = generate_mislabeled_report(v2_result)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    mislabeled_df = pd.DataFrame(mislabeled_rows)
    mislabeled_path = os.path.join(REPORTS_DIR, "mislabeled_rows_v2.csv")
    mislabeled_df.to_csv(mislabeled_path, index=False, encoding="utf-8-sig")

    val_wrong = len([r for r in mislabeled_rows if r["split"] == "val"])
    test_wrong = len([r for r in mislabeled_rows if r["split"] == "test"])
    print(f"  Val:  {val_wrong}/{len(v2_result['y_val'])} mislabeled ({val_wrong/len(v2_result['y_val'])*100:.1f}%)")
    print(f"  Test: {test_wrong}/{len(v2_result['y_test'])} mislabeled ({test_wrong/len(v2_result['y_test'])*100:.1f}%)")
    print(f"  Total: {len(mislabeled_rows)} mislabeled rows")
    print(f"  Saved: {mislabeled_path}")

    from collections import Counter
    confusion = Counter((r["true_label"], r["predicted_label"]) for r in mislabeled_rows)
    print(f"\n  Confusion breakdown:")
    for (true, pred), count in sorted(confusion.items(), key=lambda x: -x[1]):
        print(f"    {true} -> {pred}: {count}")

    print("\n" + "=" * 70)
    print("  GENERATING COMPARISON REPORT")
    print("=" * 70)

    report = generate_comparison_report(v1_result, v2_result, mislabeled_rows)
    report_path = os.path.join(REPORTS_DIR, "bert_old_v1_vs_v2_comparison.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\n" + report.encode("ascii", errors="replace").decode("ascii"))

    print("\n" + "=" * 70)
    print("  FINAL COMPARISON — V1 vs V2")
    print("=" * 70)
    print(f"  {'Metric':<22} {'V1':>12} {'V2':>12} {'Diff':>10}")
    print(f"  {'-'*56}")

    for key, name in [
        ("best_val_f1", "Val Macro-F1 (best)"),
        ("val_accuracy", "Val Accuracy"),
        ("test_accuracy", "Test Accuracy"),
        ("test_f1_macro", "Test Macro-F1"),
        ("test_f1_weighted", "Test Weighted-F1"),
    ]:
        v1_v = v1_result[key]
        v2_v = v2_result[key]
        diff = v2_v - v1_v
        sign = "+" if diff >= 0 else ""
        print(f"  {name:<22} {v1_v:>12.4f} {v2_v:>12.4f} {sign}{diff:>9.4f}")

    print(f"  {'-'*56}")
    print(f"  {'Epochs trained':<22} {v1_result['epochs_trained']:>12} {v2_result['epochs_trained']:>12}")
    print(f"  {'Training time':<22} {v1_result['train_time_s']:>11.0f}s {v2_result['train_time_s']:>11.0f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
