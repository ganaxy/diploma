import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import LABEL_NAMES, PREDICTIONS_DIR, REPORTS_DIR, FIGURES_DIR

MAX_EXAMPLES_PER_PAIR = 15
TEXT_PREVIEW_LEN = 120

def find_test_predictions() -> list[tuple[str, pd.DataFrame]]:
    pattern = os.path.join(PREDICTIONS_DIR, "*_test_preds.csv")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    results = []
    for f in files:
        basename = os.path.basename(f).replace("_test_preds.csv", "")
        df = pd.read_csv(f, encoding="utf-8-sig")
        results.append((basename, df))
    return results

def compute_error_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    correct = df["correct"].sum()
    wrong = total - correct

    per_class = []
    for label in LABEL_NAMES:
        mask = df["true_label"] == label
        cls_total = mask.sum()
        if cls_total == 0:
            continue
        cls_correct = (df.loc[mask, "correct"]).sum()
        cls_wrong = cls_total - cls_correct
        per_class.append({
            "class": label,
            "total": int(cls_total),
            "correct": int(cls_correct),
            "wrong": int(cls_wrong),
            "error_rate": cls_wrong / cls_total,
        })

    return {
        "total": int(total),
        "correct": int(correct),
        "wrong": int(wrong),
        "accuracy": correct / total,
        "error_rate": wrong / total,
        "per_class": per_class,
    }

def compute_confusion_pairs(df: pd.DataFrame) -> pd.DataFrame:
    errors = df[~df["correct"]].copy()
    if len(errors) == 0:
        return pd.DataFrame(columns=["true_label", "pred_label", "count", "pct_of_true_class"])

    pairs = (
        errors.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    true_counts = df["true_label"].value_counts().to_dict()
    pairs["pct_of_true_class"] = pairs.apply(
        lambda r: r["count"] / true_counts.get(r["true_label"], 1) * 100, axis=1
    )
    return pairs

def get_examples_for_pair(df: pd.DataFrame, true_label: str, pred_label: str, n: int) -> list[str]:
    mask = (~df["correct"]) & (df["true_label"] == true_label) & (df["pred_label"] == pred_label)
    examples = df.loc[mask, "text"].head(n).tolist()
    return [str(t)[:TEXT_PREVIEW_LEN] + ("..." if len(str(t)) > TEXT_PREVIEW_LEN else "") for t in examples]

def plot_error_rates(summary: dict, model_name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    per_class = summary["per_class"]
    classes = [r["class"] for r in per_class]
    rates = [r["error_rate"] * 100 for r in per_class]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(classes, rates, color=["#4CAF50", "#2196F3", "#F44336", "#FF9800"])
    ax.set_xlabel("Error Rate (%)")
    ax.set_title(f"Per-Class Error Rate — {model_name} (test)")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%", va="center", fontsize=10)
    ax.set_xlim(0, max(rates) * 1.2 + 5)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, f"{model_name}_error_rate_by_class.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path

def plot_text_length_vs_errors(df: pd.DataFrame, model_name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = df.copy()
    df["char_len"] = df["text"].astype(str).str.len()
    df["status"] = df["correct"].map({True: "Correct", False: "Misclassified"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=df, x="status", y="char_len", ax=axes[0], palette=["#4CAF50", "#F44336"])
    axes[0].set_title("Text Length by Prediction Status")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Character Length")

    sns.boxplot(data=df, x="true_label", y="char_len", hue="status",
                ax=axes[1], palette=["#4CAF50", "#F44336"],
                order=LABEL_NAMES)
    axes[1].set_title("Text Length by Class & Status")
    axes[1].set_xlabel("True Label")
    axes[1].set_ylabel("Character Length")
    axes[1].legend(title="", loc="upper right")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"{model_name}_text_length_vs_errors.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path

def generate_report(
    model_name: str,
    df: pd.DataFrame,
    summary: dict,
    pairs_df: pd.DataFrame,
    all_models: list[tuple[str, dict]],
) -> str:
    md = []
    md.append("# Misclassification Analysis Report\n")
    md.append(f"**Model:** {model_name}  ")
    md.append(f"**Split:** test  ")
    md.append(f"**Total samples:** {summary['total']}\n")

    if len(all_models) > 1:
        md.append("## Model Comparison (Test Error Rates)\n")
        md.append("| Model | Accuracy | Error Rate | Errors |")
        md.append("|-------|----------|------------|--------|")
        for name, s in all_models:
            marker = " *" if name == model_name else ""
            md.append(
                f"| {name}{marker} | {s['accuracy']:.4f} | "
                f"{s['error_rate']*100:.1f}% | {s['wrong']}/{s['total']} |"
            )
        md.append("")

    md.append("## 1. Overall Error Summary\n")
    md.append(f"- **Correct:** {summary['correct']} ({summary['accuracy']*100:.1f}%)")
    md.append(f"- **Misclassified:** {summary['wrong']} ({summary['error_rate']*100:.1f}%)\n")

    md.append("## 2. Per-Class Error Rates\n")
    md.append("| Class | Total | Correct | Wrong | Error Rate |")
    md.append("|-------|-------|---------|-------|------------|")
    for r in summary["per_class"]:
        md.append(
            f"| {r['class']} | {r['total']} | {r['correct']} | "
            f"{r['wrong']} | {r['error_rate']*100:.1f}% |"
        )
    md.append("")

    md.append("## 3. Top Confusion Pairs\n")
    md.append("| True Label | Predicted As | Count | % of True Class |")
    md.append("|------------|-------------|-------|-----------------|")
    for _, row in pairs_df.head(10).iterrows():
        md.append(
            f"| {row['true_label']} | {row['pred_label']} | "
            f"{row['count']} | {row['pct_of_true_class']:.1f}% |"
        )
    md.append("")

    md.append("## 4. TOXIC <-> CONSTRUCTIVE Deep Dive\n")

    toxic_as_constr = pairs_df[
        (pairs_df["true_label"] == "TOXIC") & (pairs_df["pred_label"] == "CONSTRUCTIVE")
    ]
    constr_as_toxic = pairs_df[
        (pairs_df["true_label"] == "CONSTRUCTIVE") & (pairs_df["pred_label"] == "TOXIC")
    ]

    tc_count = int(toxic_as_constr["count"].sum()) if len(toxic_as_constr) > 0 else 0
    ct_count = int(constr_as_toxic["count"].sum()) if len(constr_as_toxic) > 0 else 0
    toxic_total = sum(r["total"] for r in summary["per_class"] if r["class"] == "TOXIC")
    constr_total = sum(r["total"] for r in summary["per_class"] if r["class"] == "CONSTRUCTIVE")

    md.append(f"### TOXIC -> CONSTRUCTIVE: {tc_count} errors "
              f"({tc_count/max(toxic_total,1)*100:.1f}% of TOXIC)\n")
    examples = get_examples_for_pair(df, "TOXIC", "CONSTRUCTIVE", MAX_EXAMPLES_PER_PAIR)
    if examples:
        for i, ex in enumerate(examples, 1):
            md.append(f"{i}. `{ex}`")
    else:
        md.append("No examples found.")
    md.append("")

    md.append(f"### CONSTRUCTIVE -> TOXIC: {ct_count} errors "
              f"({ct_count/max(constr_total,1)*100:.1f}% of CONSTRUCTIVE)\n")
    examples = get_examples_for_pair(df, "CONSTRUCTIVE", "TOXIC", MAX_EXAMPLES_PER_PAIR)
    if examples:
        for i, ex in enumerate(examples, 1):
            md.append(f"{i}. `{ex}`")
    else:
        md.append("No examples found.")
    md.append("")

    md.append("## 5. Text Length Analysis\n")
    df_copy = df.copy()
    df_copy["char_len"] = df_copy["text"].astype(str).str.len()
    correct_len = df_copy.loc[df_copy["correct"], "char_len"]
    wrong_len = df_copy.loc[~df_copy["correct"], "char_len"]
    md.append(f"- **Correct predictions** — median length: {correct_len.median():.0f} chars, "
              f"mean: {correct_len.mean():.0f} chars")
    md.append(f"- **Misclassified** — median length: {wrong_len.median():.0f} chars, "
              f"mean: {wrong_len.mean():.0f} chars\n")

    if wrong_len.mean() > correct_len.mean() * 1.15:
        md.append("Longer texts tend to be misclassified more often. "
                  "This may indicate that longer comments contain mixed signals.\n")
    elif wrong_len.mean() < correct_len.mean() * 0.85:
        md.append("Shorter texts tend to be misclassified more often. "
                  "This may indicate that very short comments lack enough context for classification.\n")
    else:
        md.append("Text length does not show a strong pattern between correct and misclassified samples.\n")

    md.append("## 6. Sample Misclassifications by Confusion Pair\n")
    shown_pairs = set()
    shown_pairs.add(("TOXIC", "CONSTRUCTIVE"))
    shown_pairs.add(("CONSTRUCTIVE", "TOXIC"))

    for _, row in pairs_df.head(8).iterrows():
        pair_key = (row["true_label"], row["pred_label"])
        if pair_key in shown_pairs:
            continue
        shown_pairs.add(pair_key)

        md.append(f"### True: {row['true_label']} -> Predicted: {row['pred_label']} "
                  f"({int(row['count'])} errors)\n")
        examples = get_examples_for_pair(
            df, row["true_label"], row["pred_label"], MAX_EXAMPLES_PER_PAIR
        )
        if examples:
            for i, ex in enumerate(examples, 1):
                md.append(f"{i}. `{ex}`")
        md.append("")

    return "\n".join(md)

def main():
    print("=" * 60)
    print("  Misclassification Analysis")
    print("=" * 60)

    predictions = find_test_predictions()
    if not predictions:
        print("[error] No test prediction files found in:")
        print(f"        {PREDICTIONS_DIR}")
        print("[error] Run the baseline experiment first:  python -m src.run_experiment")
        return

    print(f"[analysis] Found {len(predictions)} test prediction file(s):")
    for name, df in predictions:
        print(f"           - {name} ({len(df)} rows)")

    all_summaries = []
    for name, df in predictions:
        s = compute_error_summary(df)
        all_summaries.append((name, s))
        print(f"[analysis] {name}: accuracy={s['accuracy']:.4f}, errors={s['wrong']}/{s['total']}")

    primary_name, primary_summary = max(all_summaries, key=lambda x: x[1]["accuracy"])
    primary_df = next(df for name, df in predictions if name == primary_name)
    print(f"\n[analysis] Primary model for detailed analysis: {primary_name}")

    pairs_df = compute_confusion_pairs(primary_df)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    pairs_path = os.path.join(REPORTS_DIR, f"{primary_name}_confusion_pairs.csv")
    pairs_df.to_csv(pairs_path, index=False)
    print(f"  Saved: {pairs_path}")

    misclassified = primary_df[~primary_df["correct"]].copy()
    misc_path = os.path.join(PREDICTIONS_DIR, f"{primary_name}_misclassified.csv")
    misclassified.to_csv(misc_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {misc_path} ({len(misclassified)} rows)")

    print("\n[analysis] Generating plots ...")
    plot_error_rates(primary_summary, primary_name)
    plot_text_length_vs_errors(primary_df, primary_name)

    print("\n[analysis] Generating report ...")
    report = generate_report(
        primary_name, primary_df, primary_summary, pairs_df, all_summaries
    )
    report_path = os.path.join(REPORTS_DIR, "error_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\n" + report.encode("ascii", errors="replace").decode("ascii"))

    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
