import os
import pandas as pd
from src.config import (
    LABEL_NAMES, REPORTS_DIR, FIGURES_DIR, PREDICTIONS_DIR, MODELS_DIR,
)
from src.data_prep import load_and_prepare
from src.train_baselines import train_all_models, save_model
from src.evaluate import full_evaluation

def select_best_model(val_results: list) -> tuple:
    best = max(val_results, key=lambda r: r["f1_macro"])
    return best["model"], best

def build_comparison_table(val_results: list) -> pd.DataFrame:
    rows = []
    for r in val_results:
        rows.append({
            "model": r["model"],
            "accuracy": round(r["accuracy"], 4),
            "precision_macro": round(r["precision_macro"], 4),
            "recall_macro": round(r["recall_macro"], 4),
            "f1_macro": round(r["f1_macro"], 4),
            "f1_weighted": round(r["f1_weighted"], 4),
        })
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False)

def generate_interpretation(
    best_name: str,
    val_result: dict,
    test_result: dict,
    comparison_df: pd.DataFrame,
) -> str:

    report = test_result["report_dict"]
    class_f1 = {c: report[c]["f1-score"] for c in LABEL_NAMES}
    weakest_class = min(class_f1, key=class_f1.get)
    weakest_f1 = class_f1[weakest_class]

    cm = test_result["confusion_matrix"]
    toxic_idx = LABEL_NAMES.index("TOXIC")
    constr_idx = LABEL_NAMES.index("CONSTRUCTIVE")
    toxic_as_constr = cm[toxic_idx][constr_idx]
    constr_as_toxic = cm[constr_idx][toxic_idx]
    toxic_total = cm[toxic_idx].sum()

    class_support = {c: int(report[c]["support"]) for c in LABEL_NAMES}
    smallest_class = min(class_support, key=class_support.get)

    md = []
    md.append("# Experiment Interpretation Report\n")

    md.append("## 1. Best model on validation\n")
    md.append(
        f"**{best_name}** achieved the highest validation Macro-F1 of "
        f"**{val_result['f1_macro']:.4f}**.\n"
    )

    md.append("### Validation comparison\n")
    md.append(comparison_df.to_markdown(index=False))
    md.append("")

    md.append("## 2. Final test scores\n")
    md.append(f"| Metric | Score |")
    md.append(f"|--------|-------|")
    md.append(f"| Accuracy | {test_result['accuracy']:.4f} |")
    md.append(f"| Precision (macro) | {test_result['precision_macro']:.4f} |")
    md.append(f"| Recall (macro) | {test_result['recall_macro']:.4f} |")
    md.append(f"| Macro-F1 | {test_result['f1_macro']:.4f} |")
    md.append(f"| Weighted-F1 | {test_result['f1_weighted']:.4f} |")
    md.append("")

    md.append("### Per-class F1 on test\n")
    md.append("| Class | F1 | Support |")
    md.append("|-------|-----|---------|")
    for c in LABEL_NAMES:
        md.append(f"| {c} | {class_f1[c]:.4f} | {class_support[c]} |")
    md.append("")

    md.append("## 3. Weakest classes\n")
    md.append(
        f"**{weakest_class}** is the weakest class with test F1 = **{weakest_f1:.4f}**.\n"
    )
    second_weakest = sorted(class_f1.items(), key=lambda x: x[1])
    if len(second_weakest) > 1:
        sw_name, sw_f1 = second_weakest[1]
        md.append(f"Second weakest: **{sw_name}** with F1 = {sw_f1:.4f}.\n")

    md.append("## 4. Is TOXIC confused with CONSTRUCTIVE?\n")
    md.append(
        f"Out of **{toxic_total}** true TOXIC samples in the test set, "
        f"**{toxic_as_constr}** were misclassified as CONSTRUCTIVE "
        f"({toxic_as_constr / max(toxic_total, 1) * 100:.1f}%).\n"
    )
    md.append(
        f"Conversely, **{constr_as_toxic}** CONSTRUCTIVE samples were "
        f"misclassified as TOXIC.\n"
    )
    if toxic_as_constr / max(toxic_total, 1) > 0.10:
        md.append(
            "⚠️ Yes — a notable portion of TOXIC comments are being confused "
            "with CONSTRUCTIVE. This is a priority issue to address.\n"
        )
    else:
        md.append(
            "✅ The confusion between TOXIC and CONSTRUCTIVE is relatively low.\n"
        )

    md.append("## 5. Should I label more data?\n")
    macro_f1_test = test_result["f1_macro"]
    if macro_f1_test < 0.60:
        md.append(
            f"**Yes, strongly recommended.** Macro-F1 = {macro_f1_test:.4f} is below 0.60, "
            "indicating the model struggles significantly. More labeled data — "
            "especially for minority classes — should improve performance.\n"
        )
    elif macro_f1_test < 0.75:
        md.append(
            f"**Yes, recommended.** Macro-F1 = {macro_f1_test:.4f} is decent but below 0.75. "
            "Labeling more data for the weakest classes will likely push scores higher.\n"
        )
    else:
        md.append(
            f"**Optional.** Macro-F1 = {macro_f1_test:.4f} is already reasonable. "
            "You could focus on improving the weakest class(es) with targeted labeling, "
            "or move on to stronger models (e.g., fine-tuned BERT).\n"
        )

    md.append("## 6. Which class should I label more of first?\n")
    md.append(
        f"**{weakest_class}** — it has the lowest F1 ({weakest_f1:.4f}) "
        f"and only {class_support[weakest_class]} test samples.\n"
    )
    if smallest_class != weakest_class:
        md.append(
            f"Also consider **{smallest_class}**, which has the smallest "
            f"support ({class_support[smallest_class]} test samples).\n"
        )
    md.append(
        "Focus your next labeling round on collecting more examples of "
        f"**{weakest_class}** comments to improve class balance and model performance.\n"
    )

    return "\n".join(md)

def main():
    print("=" * 60)
    print("  Mongolian Comment Classification — Baseline Experiment")
    print("=" * 60)

    splits = load_and_prepare()
    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]
    X_test, y_test = splits["test"]["X"], splits["test"]["y"]

    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)
    trained_models = train_all_models(X_train, y_train)

    print("\n" + "=" * 60)
    print("  VALIDATION EVALUATION")
    print("=" * 60)
    val_results = []
    model_lookup = {}
    for name, pipe in trained_models:
        y_pred_val = pipe.predict(X_val)
        result = full_evaluation(X_val, y_val, y_pred_val, name, "val")
        val_results.append(result)
        model_lookup[name] = pipe

    comparison_df = build_comparison_table(val_results)
    print("\n\n  Model Comparison (Validation)")
    print("-" * 60)
    print(comparison_df.to_string(index=False))

    os.makedirs(REPORTS_DIR, exist_ok=True)
    comp_path = os.path.join(REPORTS_DIR, "model_comparison_val.csv")
    comparison_df.to_csv(comp_path, index=False)
    print(f"\n  Saved: {comp_path}")

    best_name, best_val_result = select_best_model(val_results)
    print(f"\n{'='*60}")
    print(f"  BEST MODEL: {best_name}  (val Macro-F1 = {best_val_result['f1_macro']:.4f})")
    print(f"{'='*60}")

    best_pipe = model_lookup[best_name]
    y_pred_test = best_pipe.predict(X_test)
    test_result = full_evaluation(X_test, y_test, y_pred_test, best_name, "test")

    save_model(best_pipe, best_name)

    print("\n" + "=" * 60)
    print("  GENERATING INTERPRETATION REPORT")
    print("=" * 60)

    interpretation = generate_interpretation(
        best_name, best_val_result, test_result, comparison_df
    )
    report_path = os.path.join(REPORTS_DIR, "interpretation.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(interpretation)
    print(f"\n  Saved: {report_path}")
    try:
        print("\n" + interpretation)
    except UnicodeEncodeError:
        print("\n" + interpretation.encode("ascii", errors="replace").decode("ascii"))

    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Reports:     {REPORTS_DIR}")
    print(f"  Figures:     {FIGURES_DIR}")
    print(f"  Predictions: {PREDICTIONS_DIR}")
    print(f"  Models:      {MODELS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
