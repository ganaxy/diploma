import gc
import os
import re
import glob
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    PROJECT_ROOT, LABEL_NAMES, RANDOM_SEED, MODELS_DIR, REPORTS_DIR,
    FIGURES_DIR, PREDICTIONS_DIR, BERT_MODEL_NAME, BERT_TEXT_COLUMN,
    BERT_BATCH_SIZE,
)
from src.data_prep import load_and_prepare
from src.evaluate import full_evaluation
from src.train_bert import (
    get_device, make_dataloader, evaluate_model, train_one_epoch,
    LABEL2ID, ID2LABEL,
)

NEWLY_CSV = os.path.join(PROJECT_ROOT, "data", "sampled_labeled_fully_newly.csv")

EPOCHS = 10
PATIENCE = 4
LR = 2e-5
MAX_LENGTH = 128
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.0
MODEL_SAVE_DIR = os.path.join(MODELS_DIR, "bert_newly_best")
EXPERIMENT_NAME = "BERT_newly_data"

def tokenize_texts(texts, tokenizer):
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]

def train_bert_newly(device):
    print(f"\n{'='*60}")
    print(f"  Training BERT on newly labeled dataset")
    print(f"  Data: {os.path.basename(NEWLY_CSV)}")
    print(f"  Epochs: {EPOCHS}, LR: {LR}, MaxLen: {MAX_LENGTH}")
    print(f"{'='*60}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    splits = load_and_prepare(NEWLY_CSV)

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
    print(f"[{EXPERIMENT_NAME}] Class weights: {dict(zip(LABEL_NAMES, [f'{w:.3f}' for w in class_weights]))}")

    print(f"[{EXPERIMENT_NAME}] Loading tokenizer: {BERT_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    print(f"[{EXPERIMENT_NAME}] Tokenizing (max_length={MAX_LENGTH}) ...")
    train_ids, train_mask = tokenize_texts(X_train, tokenizer)
    val_ids, val_mask = tokenize_texts(X_val, tokenizer)
    test_ids, test_mask = tokenize_texts(X_test, tokenizer)

    train_labels = torch.tensor([LABEL2ID[l] for l in y_train], dtype=torch.long)
    val_labels = torch.tensor([LABEL2ID[l] for l in y_val], dtype=torch.long)
    test_labels = torch.tensor([LABEL2ID[l] for l in y_test], dtype=torch.long)

    batch_size = BERT_BATCH_SIZE
    train_loader = make_dataloader(train_ids, train_mask, train_labels, batch_size, shuffle=True)
    val_loader = make_dataloader(val_ids, val_mask, val_labels, batch_size * 2)
    test_loader = make_dataloader(test_ids, test_mask, test_labels, batch_size * 2)

    print(f"[{EXPERIMENT_NAME}] Loading model: {BERT_MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[{EXPERIMENT_NAME}] Steps/epoch: {len(train_loader)}, total: {total_steps}, warmup: {warmup_steps}")

    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=LABEL_SMOOTHING)

    best_f1 = 0.0
    patience_counter = 0
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    epoch_log = []

    print(f"\n[{EXPERIMENT_NAME}] Starting training ...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n--- {EXPERIMENT_NAME} | Epoch {epoch + 1}/{EPOCHS} ---")

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
            model_for_save.save_pretrained(MODEL_SAVE_DIR)
            tokenizer.save_pretrained(MODEL_SAVE_DIR)
            del model_for_save, cpu_state
            print(f"  >> New best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    train_time = time.time() - start_time
    print(f"\n[{EXPERIMENT_NAME}] Training done in {train_time:.0f}s")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[{EXPERIMENT_NAME}] Loading best model from {MODEL_SAVE_DIR}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_DIR)
    model.to(device)

    print(f"\n[{EXPERIMENT_NAME}] Evaluating on validation ...")
    val_raw = evaluate_model(model, val_loader, device)
    val_pred_labels = np.array([ID2LABEL[p] for p in val_raw["preds"]])
    val_result = full_evaluation(X_val, y_val, val_pred_labels, EXPERIMENT_NAME, "val")

    print(f"\n[{EXPERIMENT_NAME}] Evaluating on test ...")
    test_raw = evaluate_model(model, test_loader, device)
    test_pred_labels = np.array([ID2LABEL[p] for p in test_raw["preds"]])
    test_result = full_evaluation(X_test, y_test, test_pred_labels, EXPERIMENT_NAME, "test")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": EXPERIMENT_NAME,
        "epochs_trained": len(epoch_log),
        "best_val_f1": best_f1,
        "train_time_s": train_time,
        "epoch_log": epoch_log,
        "val": val_result,
        "test": test_result,
    }

def load_previous_results():
    results = []

    pattern = os.path.join(PREDICTIONS_DIR, "*_test_preds.csv")
    pred_files = sorted(glob.glob(pattern))

    for pred_path in pred_files:
        model_name = os.path.basename(pred_path).replace("_test_preds.csv", "")

        preds_df = pd.read_csv(pred_path, encoding="utf-8-sig")
        y_true = preds_df["true_label"].values
        y_pred = preds_df["pred_label"].values

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        report_path = os.path.join(REPORTS_DIR, f"{model_name}_test_report.csv")
        report_dict = None
        if os.path.isfile(report_path):
            report_df = pd.read_csv(report_path, index_col=0)
            per_class_f1 = {}
            for label in LABEL_NAMES:
                if label in report_df.index:
                    per_class_f1[label] = float(report_df.loc[label, "f1-score"])
            report_dict = per_class_f1

        results.append({
            "name": model_name,
            "metrics": metrics,
            "per_class_f1": report_dict or {},
        })

    return results

def generate_comparison_report(new_result, previous_results):
    all_results = list(previous_results)

    new_entry = {
        "name": new_result["name"],
        "metrics": {
            "accuracy": new_result["test"]["accuracy"],
            "precision_macro": new_result["test"]["precision_macro"],
            "recall_macro": new_result["test"]["recall_macro"],
            "f1_macro": new_result["test"]["f1_macro"],
            "f1_weighted": new_result["test"]["f1_weighted"],
        },
        "per_class_f1": {
            c: new_result["test"]["report_dict"][c]["f1-score"]
            for c in LABEL_NAMES
            if c in new_result["test"]["report_dict"]
        },
    }
    all_results.append(new_entry)

    all_results.sort(key=lambda r: r["metrics"]["f1_macro"], reverse=True)

    md = []
    md.append("# BERT Newly Dataset — Training & Comparison Report\n")
    md.append(f"**Model:** `{BERT_MODEL_NAME}`  ")
    md.append(f"**Dataset:** `{os.path.basename(NEWLY_CSV)}`  ")
    md.append(f"**Epochs trained:** {new_result['epochs_trained']}  ")
    md.append(f"**Training time:** {new_result['train_time_s']:.0f}s  ")
    md.append(f"**Best val Macro-F1:** {new_result['best_val_f1']:.4f}\n")
    md.append("---\n")

    md.append("## Training Configuration\n")
    md.append("| Setting | Value |")
    md.append("|---------|-------|")
    md.append(f"| Model | `{BERT_MODEL_NAME}` |")
    md.append(f"| Text column | `{BERT_TEXT_COLUMN}` |")
    md.append(f"| Dataset | `{os.path.basename(NEWLY_CSV)}` |")
    md.append(f"| Max epochs | {EPOCHS} |")
    md.append(f"| Epochs trained | {new_result['epochs_trained']} |")
    md.append(f"| Learning rate | {LR} |")
    md.append(f"| Max length | {MAX_LENGTH} |")
    md.append(f"| Batch size | {BERT_BATCH_SIZE} |")
    md.append(f"| Warmup ratio | {WARMUP_RATIO} |")
    md.append(f"| Weight decay | {WEIGHT_DECAY} |")
    md.append(f"| Label smoothing | {LABEL_SMOOTHING} |")
    md.append(f"| Early stopping patience | {PATIENCE} |")
    md.append(f"| Seed | {RANDOM_SEED} |")
    md.append("")

    md.append("## Training Progress\n")
    md.append("| Epoch | Loss | Val Accuracy | Val Macro-F1 | Val Weighted-F1 |")
    md.append("|-------|------|-------------|-------------|----------------|")
    for e in new_result["epoch_log"]:
        md.append(
            f"| {e['epoch']} | {e['loss']:.4f} | "
            f"{e['val_accuracy']:.4f} | {e['val_f1_macro']:.4f} | "
            f"{e['val_f1_weighted']:.4f} |"
        )
    md.append("")

    md.append("---\n")
    md.append("## Comparison with All Previous Models (Test Set)\n")
    md.append("| Rank | Model | Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 |")
    md.append("|------|-------|----------|-----------|--------|----------|-------------|")
    for i, r in enumerate(all_results):
        m = r["metrics"]
        marker = " **" if r["name"] == EXPERIMENT_NAME else ""
        marker_end = "**" if r["name"] == EXPERIMENT_NAME else ""
        star = " ★" if r["name"] == EXPERIMENT_NAME else ""
        md.append(
            f"| {i+1} | {marker}{r['name']}{star}{marker_end} | "
            f"{m['accuracy']:.4f} | {m['precision_macro']:.4f} | "
            f"{m['recall_macro']:.4f} | {marker}{m['f1_macro']:.4f}{marker_end} | "
            f"{m['f1_weighted']:.4f} |"
        )
    md.append("")

    md.append("## Per-Class Test F1\n")
    header = "| Class | " + " | ".join(r["name"] for r in all_results) + " |"
    sep = "|-------|" + "|".join("--------|" for _ in all_results)
    md.append(header)
    md.append(sep)
    for c in LABEL_NAMES:
        vals = []
        scores = []
        for r in all_results:
            f1_val = r["per_class_f1"].get(c, 0.0)
            scores.append(f1_val)
            vals.append(f"{f1_val:.4f}")
        best_val = max(scores) if scores else 0
        formatted = []
        for v, s in zip(vals, scores):
            if s == best_val and best_val > 0:
                formatted.append(f"**{v}**")
            else:
                formatted.append(v)
        md.append(f"| {c} | " + " | ".join(formatted) + " |")
    md.append("")

    md.append("---\n")
    md.append("## Validation Results\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Accuracy | {new_result['val']['accuracy']:.4f} |")
    md.append(f"| Precision (macro) | {new_result['val']['precision_macro']:.4f} |")
    md.append(f"| Recall (macro) | {new_result['val']['recall_macro']:.4f} |")
    md.append(f"| Macro-F1 | {new_result['val']['f1_macro']:.4f} |")
    md.append(f"| Weighted-F1 | {new_result['val']['f1_weighted']:.4f} |")
    md.append("")

    best_model = all_results[0]
    new_rank = next(i + 1 for i, r in enumerate(all_results) if r["name"] == EXPERIMENT_NAME)
    md.append("## Summary\n")
    md.append(f"**Best model (test Macro-F1):** {best_model['name']} ({best_model['metrics']['f1_macro']:.4f})\n")
    md.append(f"**{EXPERIMENT_NAME} rank:** #{new_rank} out of {len(all_results)} models\n")
    md.append(f"**{EXPERIMENT_NAME} test Macro-F1:** {new_entry['metrics']['f1_macro']:.4f}\n")

    prev_bert = [r for r in all_results if r["name"] != EXPERIMENT_NAME and "BERT" in r["name"]]
    if prev_bert:
        best_prev_bert = max(prev_bert, key=lambda r: r["metrics"]["f1_macro"])
        delta = new_entry["metrics"]["f1_macro"] - best_prev_bert["metrics"]["f1_macro"]
        sign = "+" if delta >= 0 else ""
        md.append(
            f"**vs best previous BERT ({best_prev_bert['name']}):** "
            f"{sign}{delta:.4f} Macro-F1 "
            f"({sign}{delta / best_prev_bert['metrics']['f1_macro'] * 100:.1f}% relative)\n"
        )

    prev_baseline = [r for r in all_results if "BERT" not in r["name"]]
    if prev_baseline:
        best_baseline = max(prev_baseline, key=lambda r: r["metrics"]["f1_macro"])
        delta = new_entry["metrics"]["f1_macro"] - best_baseline["metrics"]["f1_macro"]
        sign = "+" if delta >= 0 else ""
        md.append(
            f"**vs best baseline ({best_baseline['name']}):** "
            f"{sign}{delta:.4f} Macro-F1 "
            f"({sign}{delta / best_baseline['metrics']['f1_macro'] * 100:.1f}% relative)\n"
        )

    md.append("---\n")
    md.append(f"*Model: `{BERT_MODEL_NAME}` | Seed: {RANDOM_SEED}*\n")

    return "\n".join(md), all_results

def plot_comparison(all_results, new_name):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    names = [r["name"] for r in all_results]
    f1s = [r["metrics"]["f1_macro"] for r in all_results]
    accs = [r["metrics"]["accuracy"] for r in all_results]
    short_names = [
        n.replace("LogisticRegression", "LogReg")
         .replace("MultinomialNB", "MNB")
        for n in names
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(names) * 0.6)))
    fig.suptitle(
        f"BERT Newly Dataset — Comparison with All Models (Test Set)",
        fontsize=15, fontweight="bold",
    )

    colors = ["#FF5722" if n == new_name else ("#2196F3" if "BERT" in n else "#9E9E9E")
              for n in names]

    ax = axes[0]
    bars = ax.barh(range(len(names)), f1s, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Test Macro-F1", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(f1s) * 1.15 if f1s else 1.0)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    ax.invert_yaxis()

    ax = axes[1]
    bars = ax.barh(range(len(names)), accs, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Accuracy")
    ax.set_title("Test Accuracy", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(accs) * 1.15 if accs else 1.0)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FF5722", alpha=0.85, label=f"{new_name} (new)"),
        Patch(facecolor="#2196F3", alpha=0.85, label="Previous BERT"),
        Patch(facecolor="#9E9E9E", alpha=0.85, label="Baseline"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "bert_newly_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    if any(r["per_class_f1"] for r in all_results):
        bert_results = [r for r in all_results if "BERT" in r["name"] and r["per_class_f1"]]
        if bert_results:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(LABEL_NAMES))
            width = 0.8 / max(len(bert_results), 1)
            bert_colors = ["#FF5722", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

            for i, r in enumerate(bert_results):
                f1_vals = [r["per_class_f1"].get(c, 0) for c in LABEL_NAMES]
                short = r["name"].replace("BERT_", "")
                color = "#FF5722" if r["name"] == new_name else bert_colors[min(i, len(bert_colors) - 1)]
                bars = ax.bar(x + i * width, f1_vals, width, label=short, color=color, alpha=0.85)
                for bar, val in zip(bars, f1_vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x + width * (len(bert_results) - 1) / 2)
            ax.set_xticklabels(LABEL_NAMES, fontsize=10)
            ax.set_ylabel("F1-Score")
            ax.set_title("Per-Class Test F1 — BERT Models Comparison", fontsize=13, fontweight="bold")
            ax.set_ylim(0, 1.0)
            ax.legend(loc="upper right", fontsize=9)
            plt.tight_layout()

            path = os.path.join(FIGURES_DIR, "bert_newly_per_class_f1.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path}")

def main():
    print("=" * 60)
    print("  BERT Training on Newly Labeled Dataset")
    print(f"  Dataset: {os.path.basename(NEWLY_CSV)}")
    print("=" * 60)

    device = get_device()

    new_result = train_bert_newly(device)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — Loading previous results for comparison")
    print("=" * 60)

    previous_results = load_previous_results()
    previous_results = [r for r in previous_results if r["name"] != EXPERIMENT_NAME]
    print(f"[compare] Found {len(previous_results)} previous model(s):")
    for r in previous_results:
        print(f"          - {r['name']} (Macro-F1: {r['metrics']['f1_macro']:.4f})")

    print("\n[compare] Generating comparison report ...")
    report, all_results = generate_comparison_report(new_result, previous_results)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "bert_newly_comparison.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    print("\n[compare] Generating comparison figures ...")
    plot_comparison(all_results, EXPERIMENT_NAME)

    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\n" + report.encode("ascii", errors="replace").decode("ascii"))

    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    header = f"  {'Model':<35} {'Macro-F1':>9} {'Accuracy':>9} {'W-F1':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in all_results:
        marker = " *" if r["name"] == EXPERIMENT_NAME else ""
        print(
            f"  {r['name'] + marker:<35} "
            f"{r['metrics']['f1_macro']:>9.4f} "
            f"{r['metrics']['accuracy']:>9.4f} "
            f"{r['metrics']['f1_weighted']:>9.4f}"
        )
    best = all_results[0]
    print(f"\n  BEST: {best['name']} (Test Macro-F1 = {best['metrics']['f1_macro']:.4f})")
    print("=" * 60)

if __name__ == "__main__":
    main()
