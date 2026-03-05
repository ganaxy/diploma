import gc
import os
import re
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    PROJECT_ROOT, LABEL_NAMES, RANDOM_SEED, MODELS_DIR, REPORTS_DIR,
    FIGURES_DIR, BERT_MODEL_NAME, BERT_TEXT_COLUMN, BERT_BATCH_SIZE,
)
from src.data_prep import load_and_prepare
from src.evaluate import full_evaluation
from src.train_bert import (
    get_device, make_dataloader, evaluate_model, train_one_epoch,
    LABEL2ID, ID2LABEL,
)

OLD_CSV = os.path.join(PROJECT_ROOT, "data", "sampled_fully_labeled.csv")
NEW_CSV = os.path.join(PROJECT_ROOT, "data", "sampled_labeled_fully_newly.csv")

EXPERIMENTS = [
    {
        "name": "BERT_old_data",
        "csv_path": OLD_CSV,
        "model_dir": "bert_old_data_best",
        "epochs": 10,
        "patience": 4,
        "lr": 2e-5,
        "max_length": 128,
        "batch_size": BERT_BATCH_SIZE,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "label_smoothing": 0.0,
    },
    {
        "name": "BERT_new_data",
        "csv_path": NEW_CSV,
        "model_dir": "bert_new_data_best",
        "epochs": 10,
        "patience": 4,
        "lr": 2e-5,
        "max_length": 128,
        "batch_size": BERT_BATCH_SIZE,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "label_smoothing": 0.0,
    },
    {
        "name": "BERT_new_finetuned",
        "csv_path": NEW_CSV,
        "model_dir": "bert_new_data_finetuned_best",
        "epochs": 10,
        "patience": 5,
        "lr": 3e-5,
        "max_length": 256,
        "batch_size": BERT_BATCH_SIZE,
        "warmup_ratio": 0.15,
        "weight_decay": 0.02,
        "label_smoothing": 0.1,
    },
]

def tokenize_texts(texts, tokenizer, max_length):
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]

def run_experiment(exp, device):
    name = exp["name"]
    csv_path = exp["csv_path"]
    epochs = exp["epochs"]
    patience = exp["patience"]
    lr = exp["lr"]
    max_length = exp["max_length"]
    batch_size = exp["batch_size"]
    warmup_ratio = exp["warmup_ratio"]
    weight_decay = exp["weight_decay"]
    label_smoothing = exp["label_smoothing"]
    model_dir = os.path.join(MODELS_DIR, exp["model_dir"])

    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  Data: {os.path.basename(csv_path)}")
    print(f"  Epochs: {epochs}, LR: {lr}, MaxLen: {max_length}")
    print(f"  Label smoothing: {label_smoothing}, Weight decay: {weight_decay}")
    print(f"{'='*60}")

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

    unique_labels = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=unique_labels, y=y_train)
    label_weight_map = dict(zip(unique_labels, weights))
    class_weights = [label_weight_map[n] for n in LABEL_NAMES]
    print(f"[{name}] Class weights: {dict(zip(LABEL_NAMES, [f'{w:.3f}' for w in class_weights]))}")

    print(f"[{name}] Loading tokenizer: {BERT_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    print(f"[{name}] Tokenizing (max_length={max_length}) ...")
    train_ids, train_mask = tokenize_texts(X_train, tokenizer, max_length)
    val_ids, val_mask = tokenize_texts(X_val, tokenizer, max_length)
    test_ids, test_mask = tokenize_texts(X_test, tokenizer, max_length)

    train_labels = torch.tensor([LABEL2ID[l] for l in y_train], dtype=torch.long)
    val_labels = torch.tensor([LABEL2ID[l] for l in y_val], dtype=torch.long)
    test_labels = torch.tensor([LABEL2ID[l] for l in y_test], dtype=torch.long)

    train_loader = make_dataloader(train_ids, train_mask, train_labels, batch_size, shuffle=True)
    val_loader = make_dataloader(val_ids, val_mask, val_labels, batch_size * 2)
    test_loader = make_dataloader(test_ids, test_mask, test_labels, batch_size * 2)

    print(f"[{name}] Loading model: {BERT_MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[{name}] Steps/epoch: {len(train_loader)}, total: {total_steps}, warmup: {warmup_steps}")

    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=label_smoothing)

    best_f1 = 0.0
    patience_counter = 0
    os.makedirs(model_dir, exist_ok=True)
    epoch_log = []

    print(f"\n[{name}] Starting training ...")
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\n--- {name} | Epoch {epoch + 1}/{epochs} ---")

        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

        val_metrics = evaluate_model(model, val_loader, device)
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val Macro-F1: {val_metrics['f1_macro']:.4f}")
        print(f"  Val Weighted-F1: {val_metrics['f1_weighted']:.4f}")

        epoch_log.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_weighted": val_metrics["f1_weighted"],
        })

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience_counter = 0
            cpu_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model_for_save = AutoModelForSequenceClassification.from_pretrained(
                BERT_MODEL_NAME, num_labels=len(LABEL_NAMES),
                id2label=ID2LABEL, label2id=LABEL2ID,
            )
            model_for_save.resize_token_embeddings(len(tokenizer))
            model_for_save.load_state_dict(cpu_state)
            model_for_save.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            del model_for_save, cpu_state
            print(f"  >> New best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    train_time = time.time() - start_time
    print(f"\n[{name}] Training done in {train_time:.0f}s")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[{name}] Loading best model from {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    print(f"\n[{name}] Evaluating on validation ...")
    val_raw = evaluate_model(model, val_loader, device)
    val_pred_labels = np.array([ID2LABEL[p] for p in val_raw["preds"]])
    val_result = full_evaluation(X_val, y_val, val_pred_labels, name, "val")

    print(f"\n[{name}] Evaluating on test ...")
    test_raw = evaluate_model(model, test_loader, device)
    test_pred_labels = np.array([ID2LABEL[p] for p in test_raw["preds"]])
    test_result = full_evaluation(X_test, y_test, test_pred_labels, name, "test")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": name,
        "csv": os.path.basename(csv_path),
        "epochs_trained": len(epoch_log),
        "best_val_f1": best_f1,
        "train_time_s": train_time,
        "epoch_log": epoch_log,
        "val": val_result,
        "test": test_result,
        "config": exp,
    }

def plot_comparison(results):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    names = [r["name"] for r in results]
    short_names = [n.replace("BERT_", "") for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("BERT Dataset Comparison — Test Set Metrics", fontsize=15, fontweight="bold")

    metrics = [
        ("accuracy", "Accuracy"),
        ("f1_macro", "Macro-F1"),
        ("f1_weighted", "Weighted-F1"),
    ]

    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for ax, (key, label) in zip(axes, metrics):
        vals = [r["test"][key] for r in results]
        bars = ax.bar(range(len(names)), vals, color=colors[:len(names)], alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short_names, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.15)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "bert_dataset_comparison_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(LABEL_NAMES))
    width = 0.25
    for i, r in enumerate(results):
        report = r["test"]["report_dict"]
        f1s = [report[c]["f1-score"] for c in LABEL_NAMES]
        bars = ax.bar(x + i * width, f1s, width, label=short_names[i],
                      color=colors[i], alpha=0.85)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(LABEL_NAMES, fontsize=10)
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class Test F1 — BERT Dataset Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "bert_dataset_comparison_per_class.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        log = r["epoch_log"]
        epochs_range = [e["epoch"] for e in log]
        f1s = [e["val_f1_macro"] for e in log]
        losses = [e["loss"] for e in log]
        ax.plot(epochs_range, f1s, "o-", color="#FF5722", label="Val Macro-F1")
        ax2 = ax.twinx()
        ax2.plot(epochs_range, losses, "s--", color="#2196F3", alpha=0.6, label="Train Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Macro-F1", color="#FF5722")
        ax2.set_ylabel("Train Loss", color="#2196F3")
        ax.set_title(r["name"].replace("BERT_", ""), fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "bert_dataset_comparison_training.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def generate_comparison_report(results):
    md = []
    md.append("# BERT Dataset Comparison Report")
    md.append("## Old Data vs New Data vs Fine-tuned\n")
    md.append("---\n")

    md.append("## Experiment Configurations\n")
    md.append("| Setting | Old Data | New Data | New Fine-tuned |")
    md.append("|---------|----------|----------|----------------|")
    keys = [
        ("Data file", "csv"),
        ("Epochs (max)", lambda r: str(r["config"]["epochs"])),
        ("Epochs trained", lambda r: str(r["epochs_trained"])),
        ("Learning rate", lambda r: str(r["config"]["lr"])),
        ("Max length", lambda r: str(r["config"]["max_length"])),
        ("Label smoothing", lambda r: str(r["config"]["label_smoothing"])),
        ("Weight decay", lambda r: str(r["config"]["weight_decay"])),
        ("Warmup ratio", lambda r: str(r["config"]["warmup_ratio"])),
        ("Patience", lambda r: str(r["config"]["patience"])),
        ("Training time", lambda r: f"{r['train_time_s']:.0f}s"),
    ]
    for label, key_or_fn in keys:
        vals = []
        for r in results:
            if callable(key_or_fn):
                vals.append(key_or_fn(r))
            else:
                vals.append(str(r[key_or_fn]))
        md.append(f"| {label} | {' | '.join(vals)} |")
    md.append("")

    md.append("## Validation Results\n")
    md.append("| Metric | " + " | ".join(r["name"] for r in results) + " |")
    md.append("|--------|" + "|".join("--------|" for _ in results))
    for metric, label in [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision (macro)"),
        ("recall_macro", "Recall (macro)"),
        ("f1_macro", "Macro-F1"),
        ("f1_weighted", "Weighted-F1"),
    ]:
        vals = [f"{r['val'][metric]:.4f}" for r in results]
        best_val = max(r["val"][metric] for r in results)
        vals = [f"**{v}**" if float(v) == best_val else v for v in vals]
        md.append(f"| {label} | {' | '.join(vals)} |")
    md.append("")

    md.append("## Test Results\n")
    md.append("| Metric | " + " | ".join(r["name"] for r in results) + " |")
    md.append("|--------|" + "|".join("--------|" for _ in results))
    for metric, label in [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision (macro)"),
        ("recall_macro", "Recall (macro)"),
        ("f1_macro", "Macro-F1"),
        ("f1_weighted", "Weighted-F1"),
    ]:
        vals = [f"{r['test'][metric]:.4f}" for r in results]
        best_val = max(r["test"][metric] for r in results)
        vals = [f"**{v}**" if float(v) == best_val else v for v in vals]
        md.append(f"| {label} | {' | '.join(vals)} |")
    md.append("")

    md.append("## Per-Class Test F1\n")
    md.append("| Class | " + " | ".join(r["name"] for r in results) + " |")
    md.append("|-------|" + "|".join("--------|" for _ in results))
    for c in LABEL_NAMES:
        vals = []
        scores = []
        for r in results:
            f1 = r["test"]["report_dict"][c]["f1-score"]
            scores.append(f1)
            vals.append(f"{f1:.4f}")
        best_val = max(scores)
        vals = [f"**{v}**" if float(v) == best_val else v for v, s in zip(vals, scores)]
        md.append(f"| {c} | {' | '.join(vals)} |")
    md.append("")

    md.append("## Training Progress\n")
    for r in results:
        md.append(f"### {r['name']}\n")
        md.append("| Epoch | Loss | Val Accuracy | Val Macro-F1 |")
        md.append("|-------|------|-------------|-------------|")
        for e in r["epoch_log"]:
            md.append(
                f"| {e['epoch']} | {e['loss']:.4f} | "
                f"{e['val_accuracy']:.4f} | {e['val_f1_macro']:.4f} |"
            )
        md.append("")

    best_test = max(results, key=lambda r: r["test"]["f1_macro"])
    best_val = max(results, key=lambda r: r["val"]["f1_macro"])

    md.append("## Summary\n")
    md.append(
        f"**Best model on test set:** {best_test['name']} "
        f"(Macro-F1 = {best_test['test']['f1_macro']:.4f})\n"
    )
    md.append(
        f"**Best model on val set:** {best_val['name']} "
        f"(Macro-F1 = {best_val['val']['f1_macro']:.4f})\n"
    )

    if len(results) >= 2:
        old_f1 = results[0]["test"]["f1_macro"]
        new_f1 = results[1]["test"]["f1_macro"]
        delta = new_f1 - old_f1
        md.append(
            f"**Old vs New data:** Test Macro-F1 changed by "
            f"{'+' if delta >= 0 else ''}{delta:.4f} "
            f"({'+' if delta >= 0 else ''}{delta / old_f1 * 100:.1f}% relative)\n"
        )

    if len(results) >= 3:
        new_f1 = results[1]["test"]["f1_macro"]
        ft_f1 = results[2]["test"]["f1_macro"]
        delta = ft_f1 - new_f1
        md.append(
            f"**Fine-tuning improvement:** Test Macro-F1 changed by "
            f"{'+' if delta >= 0 else ''}{delta:.4f} "
            f"({'+' if delta >= 0 else ''}{delta / new_f1 * 100:.1f}% relative)\n"
        )

    md.append("## Figures\n")
    md.append("- `outputs/figures/bert_dataset_comparison_metrics.png` — Test metrics comparison")
    md.append("- `outputs/figures/bert_dataset_comparison_per_class.png` — Per-class F1 comparison")
    md.append("- `outputs/figures/bert_dataset_comparison_training.png` — Training progress curves")
    md.append("")

    md.append("---\n")
    md.append(f"*Model: `{BERT_MODEL_NAME}` | Seed: {RANDOM_SEED}*\n")

    return "\n".join(md)

def main():
    print("=" * 60)
    print("  BERT Dataset Comparison")
    print("  Old Data vs New Data vs Fine-tuned")
    print("=" * 60)

    device = get_device()
    results = []

    for exp in EXPERIMENTS:
        result = run_experiment(exp, device)
        results.append(result)

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE — Generating comparison")
    print("=" * 60)

    plot_comparison(results)

    report = generate_comparison_report(results)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "bert_dataset_comparison.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Saved: {report_path}")

    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\n" + report.encode("ascii", errors="replace").decode("ascii"))

    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    header = f"  {'Experiment':<25} {'Val F1':>8} {'Test F1':>8} {'Test Acc':>8} {'Time':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(
            f"  {r['name']:<25} "
            f"{r['val']['f1_macro']:>8.4f} "
            f"{r['test']['f1_macro']:>8.4f} "
            f"{r['test']['accuracy']:>8.4f} "
            f"{r['train_time_s']:>7.0f}s"
        )
    best = max(results, key=lambda r: r["test"]["f1_macro"])
    print(f"\n  BEST: {best['name']} (Test Macro-F1 = {best['test']['f1_macro']:.4f})")
    print("=" * 60)

if __name__ == "__main__":
    main()
