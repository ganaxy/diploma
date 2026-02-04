import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from src.config import LABEL_NAMES, REPORTS_DIR, FIGURES_DIR, PREDICTIONS_DIR

def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

def print_metrics(metrics: dict, prefix: str = ""):
    tag = f"[{prefix}] " if prefix else ""
    print(f"{tag}Accuracy:      {metrics['accuracy']:.4f}")
    print(f"{tag}Precision (M): {metrics['precision_macro']:.4f}")
    print(f"{tag}Recall (M):    {metrics['recall_macro']:.4f}")
    print(f"{tag}Macro-F1:      {metrics['f1_macro']:.4f}")
    print(f"{tag}Weighted-F1:   {metrics['f1_weighted']:.4f}")

def save_classification_report(y_true, y_pred, model_name: str, split_name: str):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    report_txt = classification_report(
        y_true, y_pred, labels=LABEL_NAMES, zero_division=0
    )
    txt_path = os.path.join(REPORTS_DIR, f"{model_name}_{split_name}_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report: {model_name} ({split_name})\n")
        f.write("=" * 60 + "\n")
        f.write(report_txt)
    print(f"  Saved: {txt_path}")

    report_dict = classification_report(
        y_true, y_pred, labels=LABEL_NAMES, output_dict=True, zero_division=0
    )
    csv_path = os.path.join(REPORTS_DIR, f"{model_name}_{split_name}_report.csv")
    pd.DataFrame(report_dict).T.to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    return report_dict

def save_confusion_matrix(y_true, y_pred, model_name: str, split_name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — {split_name}")
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, f"{model_name}_{split_name}_cm.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    return cm

def save_predictions(X_texts, y_true, y_pred, model_name: str, split_name: str):
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    pred_df = pd.DataFrame({
        "text": X_texts,
        "true_label": y_true,
        "pred_label": y_pred,
        "correct": np.array(y_true) == np.array(y_pred),
    })
    csv_path = os.path.join(PREDICTIONS_DIR, f"{model_name}_{split_name}_preds.csv")
    pred_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {csv_path}")

def full_evaluation(X_texts, y_true, y_pred, model_name: str, split_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name} on {split_name}")
    print(f"{'='*60}")

    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, prefix=f"{model_name}/{split_name}")

    print(classification_report(y_true, y_pred, labels=LABEL_NAMES, zero_division=0))

    report_dict = save_classification_report(y_true, y_pred, model_name, split_name)
    cm = save_confusion_matrix(y_true, y_pred, model_name, split_name)
    save_predictions(X_texts, y_true, y_pred, model_name, split_name)

    return {
        "model": model_name,
        "split": split_name,
        **metrics,
        "report_dict": report_dict,
        "confusion_matrix": cm,
    }
